#!/usr/bin/env python3
"""
诊断脚本: 用于分析 RL controller 的 reward 平衡性问题。
运行示例:
    python diagnose_policy.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --dataset aime_2025.jsonl
"""

import os, json, math, argparse
import numpy as np
from tqdm import tqdm

import torch
from deepconf import DeepThinkLLM
from vllm import SamplingParams
from dynasor.core.evaluator import math_equal as equal_func
from deepconf.utils import extract_answer

# ---------------------------
# ConfTracker (和训练一致)
# ---------------------------
from collections import deque

class ConfTracker:
    def __init__(self, window_size=2048):
        self.window_size = window_size
        self.buf = deque()
        self.running_sum = 0.0
        self.conf_history = []

    def push_token_logprob(self, token_logprob: float):
        self.buf.append(token_logprob)
        self.running_sum += token_logprob
        if len(self.buf) > self.window_size:
            old = self.buf.popleft()
            self.running_sum -= old
        denom = len(self.buf) if len(self.buf) > 0 else 1
        group_conf = - (self.running_sum / denom)
        self.conf_history.append(group_conf)
        return group_conf

# ---------------------------
# Helper: Normalize confidence
# ---------------------------
def normalize_confidence(conf_value: float, global_stats: dict):
    g_min = global_stats.get('min', 0.0)
    g_max = global_stats.get('max', 1.0)
    g_range = g_max - g_min
    if g_range > 1e-6:
        norm_val = (conf_value - g_min) / g_range
        return float(np.clip(norm_val, 0.0, 1.0))
    else:
        return 0.5

# ---------------------------
# Reward functions (和训练一致)
# ---------------------------
def h_sig(c, c0=0.5, k=8.0, s=4.0):
    x = k * (c0 - c)
    sig = 1.0 / (1.0 + math.exp(-x))
    return s * (sig - 0.5)

def compute_stop_reward(tokens_used, max_tokens, c_t, trend, cfg):
    token_term = cfg["alpha_token"] * (1.0 - (tokens_used / float(max_tokens)))
    conf_term  = cfg["confidence_scale"] * h_sig(
        c_t, c0=cfg["confidence_mid"], k=cfg["confidence_k"], s=cfg["confidence_s"]
    )
    trend_term = cfg["trend_scale"] * (-trend)
    return token_term + conf_term + trend_term, (token_term, conf_term, trend_term)

def compute_final_noop_reward(is_correct, final_conf_norm, cfg):
    if is_correct:
        return cfg["correctness_weight"] + cfg.get("confidence_weight", 0.0) * float(final_conf_norm)
    else:
        return - float(cfg["wrong_penalty"])

# ---------------------------
# Main diagnostic routine
# ---------------------------
def diagnose(args):
    llm = DeepThinkLLM(model=args.model, tensor_parallel_size=1)
    sampling_params = SamplingParams(
        temperature=0.6, top_p=0.95, max_tokens=args.max_tokens, logprobs=20
    )

    with open(args.dataset, "r", encoding="utf-8") as f:
        all_data = [json.loads(line.strip()) for line in f]

    subset = all_data[:args.n_samples]

    # Global stats for confidence normalization
    global_stats = {"min":0.0,"max":1.0,"median":0.5}

    # Step 1: Baseline correctness
    corrects = 0
    for item in tqdm(subset, desc="Baseline test"):
        q, gt = item["question"], str(item["answer"]).strip()
        full_prompt = llm.tokenizer.apply_chat_template(
            [{"role":"user","content":q}], tokenize=False, add_generation_prompt=True
        )
        out = llm.generate([full_prompt], sampling_params, use_tqdm=False)[0].outputs[0]
        text = getattr(out, "text", "")
        pred = extract_answer(text)
        if equal_func(pred, gt):
            corrects += 1
    print("\n=== Step 1: Baseline Correctness ===")
    print(f"Baseline correct rate: {corrects}/{len(subset)} = {corrects/len(subset):.2%}")

    # Step 2 & 3: Rewards analysis
    stop_rewards, natural_rewards = [], []
    stop_decomp = []

    for item in tqdm(subset, desc="Reward test"):
        q, gt = item["question"], str(item["answer"]).strip()
        full_prompt = llm.tokenizer.apply_chat_template(
            [{"role":"user","content":q}], tokenize=False, add_generation_prompt=True
        )
        trace = llm.generate([full_prompt], sampling_params, use_tqdm=False)[0].outputs[0]
        if not trace or not getattr(trace, "logprobs", None): continue
        logprobs = [list(lpd.values())[0].logprob for lpd in trace.logprobs if lpd]
        if not logprobs: continue

        tracker = ConfTracker(window_size=args.window_size)
        for lp in logprobs: tracker.push_token_logprob(lp)

        # --- Stop at mid-point
        mid = len(logprobs)//2 or 1
        group_conf_mid = tracker.conf_history[mid-1]
        c_norm = normalize_confidence(group_conf_mid, global_stats)
        stop_r, decomp = compute_stop_reward(mid, args.max_tokens, c_norm, 0.0, vars(args))
        stop_rewards.append(stop_r); stop_decomp.append(decomp)

        # --- Natural completion
        final_ans = extract_answer(getattr(trace,"text",""))
        is_corr = equal_func(final_ans, gt)
        final_conf = tracker.conf_history[-1]
        final_c_norm = normalize_confidence(final_conf, global_stats)
        natural_r = compute_final_noop_reward(is_corr, final_c_norm, vars(args))
        natural_rewards.append(natural_r)

    print("\n=== Step 2: Stop Reward Components (avg) ===")
    if stop_decomp:
        token_t, conf_t, trend_t = np.mean([d[0] for d in stop_decomp]), np.mean([d[1] for d in stop_decomp]), np.mean([d[2] for d in stop_decomp])
        print(f"token_term={token_t:.2f}, conf_term={conf_t:.2f}, trend_term={trend_t:.2f}")

    print("\n=== Step 3: Reward Comparison ===")
    print(f"Avg stop reward:    {np.mean(stop_rewards):.2f}")
    print(f"Avg natural reward: {np.mean(natural_rewards):.2f}")

# ---------------------------
# Argument parsing
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--n_samples", type=int, default=50)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--window_size", type=int, default=2048)
    # reward config
    p.add_argument("--alpha_token", type=float, default=3.0)
    p.add_argument("--confidence_scale", type=float, default=4.0)
    p.add_argument("--confidence_mid", type=float, default=0.5)
    p.add_argument("--confidence_k", type=float, default=8.0)
    p.add_argument("--confidence_s", type=float, default=4.0)
    p.add_argument("--trend_scale", type=float, default=2.0)
    p.add_argument("--correctness_weight", type=float, default=10.0)
    p.add_argument("--wrong_penalty", type=float, default=10.0)
    p.add_argument("--confidence_weight", type=float, default=1.0)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    diagnose(args)

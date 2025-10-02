#!/usr/bin/env python3
"""
Sanity Check: Compare stop vs natural rewards on a few samples.
确保自然完成的奖励 > 提前stop的奖励。
"""

import os, json, math, argparse
import numpy as np
from tqdm import tqdm

from deepconf import DeepThinkLLM
from vllm import SamplingParams
from dynasor.core.evaluator import math_equal as equal_func
from deepconf.utils import extract_answer, compute_confidence

from collections import deque

# ---------------------------
# ConfTracker (与训练代码保持一致)
# ---------------------------
class ConfTracker:
    def __init__(self, window_size=2048):
        self.window_size = window_size
        self.buf = deque()
        self.running_sum = 0.0
        self.conf_history = []

    def push_token_confidence(self, token_conf: float):
        self.buf.append(token_conf)
        self.running_sum += token_conf
        if len(self.buf) > self.window_size:
            old = self.buf.popleft()
            self.running_sum -= old
        denom = len(self.buf) if len(self.buf) > 0 else 1
        group_conf = self.running_sum / denom
        self.conf_history.append(group_conf)
        return group_conf


def normalize_confidence(conf_value: float, global_stats: dict):
    g_min = global_stats.get('min', 0.0)
    g_max = global_stats.get('max', 1.0)
    g_range = g_max - g_min
    if g_range > 1e-6:
        return float(np.clip((conf_value - g_min) / g_range, 0.0, 1.0))
    else:
        return 0.5

def h_sig(c, c0=0.5, k=8.0, s=4.0):
    x = k * (c0 - c)
    sig = 1.0 / (1.0 + math.exp(-x))
    return s * (sig - 0.5)

def compute_stop_reward(tokens_used, max_tokens, c_t, trend, cfg):
    token_term = cfg.alpha_token * (1.0 - (tokens_used / float(max_tokens)))
    conf_term  = cfg.confidence_scale * h_sig(
        c_t, c0=cfg.confidence_mid, k=cfg.confidence_k, s=cfg.confidence_s
    )
    trend_term = cfg.trend_scale * (-trend)
    return token_term + conf_term + trend_term

def compute_final_noop_reward(is_correct, final_conf_norm, cfg):
    if is_correct:
        return cfg.correctness_weight + cfg.confidence_weight * float(final_conf_norm)
    else:
        return - float(cfg.wrong_penalty)

# ---------------------------
# Main sanity check
# ---------------------------
def sanity_check(args):
    print("Initializing model...")
    llm = DeepThinkLLM(model=args.model, tensor_parallel_size=1)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        logprobs=20
    )

    print(f"Loading dataset {args.dataset}...")
    with open(args.dataset, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f]

    subset = data[:args.n_samples]

    # 这里为了简化，global_stats 用默认；更严谨的做法是从 warmup 导入
    global_stats = {"min":0.0,"max":1.0,"median":0.5}

    stop_rewards, natural_rewards = [], []

    for item in tqdm(subset, desc="SanityCheck"):
        q, gt = item["question"], str(item.get("answer", "")).strip()
        messages = [
            {"role": "system", "content": "该助手为DeepSeek-R1，由深度求索公司创造。\n今天是2025年5月28日，星期一。\n"},
            {"role": "user", "content": q}
        ]
        full_prompt = llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        out = llm.generate([full_prompt], sampling_params, use_tqdm=False)[0].outputs[0]
        if not out or not getattr(out, "logprobs", None): 
            continue

        # ✅ 使用官方 compute_confidence
        token_confs = compute_confidence(out.logprobs)
        tracker = ConfTracker(window_size=args.window_size)
        for conf in token_confs: 
            tracker.push_token_confidence(conf)

        # Stop at mid-trace
        mid = len(token_confs)//2 or 1
        group_conf_mid = tracker.conf_history[mid-1]
        c_norm = normalize_confidence(group_conf_mid, global_stats)
        stop_r = compute_stop_reward(mid, args.max_tokens, c_norm, 0.0, args)
        stop_rewards.append(stop_r)

        # Natural completion
        pred = extract_answer(getattr(out,"text",""))
        is_corr = pred is not None and equal_func(pred, gt)
        final_conf = tracker.conf_history[-1]
        final_c_norm = normalize_confidence(final_conf, global_stats)
        natural_r = compute_final_noop_reward(is_corr, final_c_norm, args)
        natural_rewards.append(natural_r)

    print("\n================ Sanity Check Result ================")
    print(f"Avg stop reward (mid-trace): {np.mean(stop_rewards):.2f}")
    print(f"Avg natural completion reward: {np.mean(natural_rewards):.2f}")
    if np.mean(natural_rewards) > np.mean(stop_rewards):
        print("✅ Good: Natural > Stop (reward function is consistent)")
    else:
        print("⚠️ Warning: Stop >= Natural (reward may cause premature stopping)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--n_samples", type=int, default=5)
    p.add_argument("--max_tokens", type=int, default=32000)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--window_size", type=int, default=2048)

    # Reward config
    p.add_argument("--alpha_token", type=float, default=1.0)
    p.add_argument("--confidence_scale", type=float, default=1.0)
    p.add_argument("--confidence_mid", type=float, default=0.5)
    p.add_argument("--confidence_k", type=float, default=8.0)
    p.add_argument("--confidence_s", type=float, default=4.0)
    p.add_argument("--trend_scale", type=float, default=1.0)
    p.add_argument("--correctness_weight", type=float, default=20.0)
    p.add_argument("--wrong_penalty", type=float, default=12.0)
    p.add_argument("--confidence_weight", type=float, default=1.0)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    sanity_check(args)

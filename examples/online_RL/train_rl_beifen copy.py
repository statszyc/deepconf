#!/usr/bin/env python3
"""
Train an RL policy (REINFORCE) that decides Stop / No-op periodically during
generation based on sliding-window group confidences. Includes built-in sanity
checks and corrected REINFORCE returns alignment so training is meaningful.

Usage:
    python train_rl_controller_with_check.py --dataset aime_2025.jsonl --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
        --max_tokens 64000 --epochs 10
"""
import os
import json
import math
import time
import random
import argparse
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from deepconf import DeepThinkLLM
from vllm import SamplingParams
from dynasor.core.evaluator import math_equal as equal_func
from deepconf.utils import extract_answer

# ---------------------------
# Utilities / Seed
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except:
        pass

# ---------------------------
# Robust logprob extraction (vllm output can vary)
# ---------------------------
def safe_extract_logprob(lpd):
    """Given a vllm logprobs dict/list element, try to extract a float logprob."""
    if lpd is None:
        return 0.0
    # lpd typically is a dict mapping token_str -> object with .logprob or numeric
    try:
        v = list(lpd.values())[0]
        if hasattr(v, "logprob"):
            return float(v.logprob)
        else:
            return float(v)
    except Exception:
        try:
            return float(lpd)
        except Exception:
            return 0.0

# ---------------------------
# Helper function for confidence normalization
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
# ConfTracker
# ---------------------------
class ConfTracker:
    def __init__(self, window_size=2048, recent_raw=5):
        self.window_size = window_size
        self.buf = deque()
        self.running_sum = 0.0
        self.conf_history = []
        self.global_min = float("inf")

    def push_token_logprob(self, token_logprob: float):
        self.buf.append(token_logprob)
        self.running_sum += token_logprob
        if len(self.buf) > self.window_size:
            old = self.buf.popleft()
            self.running_sum -= old
        denom = len(self.buf) if len(self.buf) > 0 else 1
        group_conf = - (self.running_sum / denom)
        self.conf_history.append(group_conf)
        if group_conf < self.global_min:
            self.global_min = group_conf
        return group_conf

    def last_k_raw(self, K=5):
        if not self.conf_history: return [0.0] * K
        arr = self.conf_history[-K:]
        if len(arr) < K: arr = [0.0] * (K - len(arr)) + arr
        return list(arr)

    def last_k_stats(self, K=5):
        last = self.conf_history[-K:] if len(self.conf_history) >= K else self.conf_history
        if not last: return 0.0, 0.0, 0.0, 0.0
        a = np.array(last, dtype=float)
        mean, mn, std = float(np.mean(a)), float(np.min(a)), float(np.std(a))
        slope = float(np.polyfit(np.arange(len(a)), a, 1)[0]) if len(a) >= 2 else 0.0
        return mean, mn, std, slope

    def get_global_min(self):
        return float(self.global_min) if self.global_min != float("inf") else 0.0

# ---------------------------
# Policy network (small MLP)
# ---------------------------
class PolicyNet(nn.Module):
    def __init__(self, input_dim, hidden=(128,64), action_dim=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden[0]), nn.ReLU()]
        for i in range(len(hidden) - 1):
            layers.extend([nn.Linear(hidden[i], hidden[i+1]), nn.ReLU()])
        layers.append(nn.Linear(hidden[-1], action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ---------------------------
# Reward calculation
# ---------------------------
def h_sig(c, c0=0.5, k=8.0, s=4.0):
    x = k * (c0 - c)
    sig = 1.0 / (1.0 + math.exp(-x))
    return s * (sig - 0.5)

def compute_stop_reward(tokens_used, max_tokens, c_t, trend, reward_config):
    alpha = reward_config["alpha_token"]
    lam_c = reward_config["confidence_scale"]
    lam_trend = reward_config["trend_scale"]
    c0 = reward_config["confidence_mid"]
    k = reward_config["confidence_k"]
    s = reward_config["confidence_s"]

    token_term = alpha * (1.0 - (tokens_used / float(max_tokens)))
    conf_term = lam_c * h_sig(c_t, c0=c0, k=k, s=s)
    trend_term = lam_trend * (-trend)
    total_reward = float(token_term + conf_term + trend_term)

    if reward_config.get("debug_rewards", False):
        print(f"[Stop Reward Components] token={token_term:.2f}, conf={conf_term:.2f}, trend={trend_term:.2f}, total={total_reward:.2f}")

    return total_reward

def compute_noop_immediate(decision_stride, reward_config):
    return - reward_config["token_cost_weight"] * float(decision_stride)

def compute_final_noop_reward(is_correct, final_conf_norm, reward_config):
    if is_correct:
        return reward_config["correctness_weight"] + reward_config.get("confidence_weight", 0.0) * float(final_conf_norm)
    else:
        return - float(reward_config["wrong_penalty"])

# ---------------------------
# Environment loop helper
# ---------------------------
def run_episode_on_trace(llm, prompt, ground_truth, sampling_params, controller, device,
                         reward_config, max_tokens, decision_stride, K, gamma, global_stats):
    gen_obj = llm.generate([prompt], sampling_params, use_tqdm=False)[0].outputs[0]

    token_logprobs = []
    if hasattr(gen_obj, "logprobs") and gen_obj.logprobs:
        for lpd in gen_obj.logprobs:
            token_logprobs.append(safe_extract_logprob(lpd))

    text = getattr(gen_obj, "text", "")
    conf_tracker = ConfTracker(window_size=reward_config.get("window_size", 2048))
    tokens_used = 0
    saved_logprobs, rewards, avg_conf_history = [], [], []
    actions_counts = {0:0, 1:0}

    for idx, lp in enumerate(token_logprobs):
        group_conf = conf_tracker.push_token_logprob(lp)
        tokens_used += 1
        avg_conf_history.append(group_conf)

        if tokens_used > 0 and tokens_used % decision_stride == 0:
            raw_lastK = conf_tracker.last_k_raw(K=K)
            c_mean, c_min, c_std, c_slope = conf_tracker.last_k_stats(K=K)
            global_min_local, tokens_ratio = conf_tracker.get_global_min(), tokens_used / float(max_tokens)
            recent_token_conf = token_logprobs[idx]

            state_list = raw_lastK + [c_mean, c_min, c_std, c_slope, global_min_local, tokens_ratio, recent_token_conf]
            state = torch.FloatTensor(state_list).unsqueeze(0).to(device)

            logits = controller(state)
            probs = torch.softmax(logits, dim=1)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            saved_logprobs.append(m.log_prob(action))
            actions_counts[action.item()] += 1

            if action.item() == 1: # Stop
                final_answer = extract_answer(text)
                is_correct_future = equal_func(final_answer, ground_truth)

                if is_correct_future:
                    stop_r = -reward_config["early_stop_penalty"]
                else:
                    c_norm = normalize_confidence(group_conf, global_stats)
                    stop_r = compute_stop_reward(tokens_used, max_tokens, c_norm, c_slope, reward_config)

                rewards.append(stop_r)
                return {
                    "done_reason":"stop",
                    "tokens_used": tokens_used,
                    "decisions": len(saved_logprobs),
                    "actions_counts": actions_counts,
                    "avg_conf": np.mean(avg_conf_history) if avg_conf_history else 0.0,
                    "saved_logprobs": saved_logprobs,
                    "rewards": rewards,
                    "future_correct": is_correct_future,
                }
            else: # No-op
                rewards.append(compute_noop_immediate(decision_stride, reward_config))

        if tokens_used >= max_tokens:
            break

    # natural completion
    return {
        "done_reason":"natural",
        "tokens_used": tokens_used,
        "decisions": len(saved_logprobs),
        "actions_counts": actions_counts,
        "avg_conf": np.mean(avg_conf_history) if avg_conf_history else 0.0,
        "raw_text": text,
        "saved_logprobs": saved_logprobs,
        "rewards": rewards,
        "conf_tracker": conf_tracker,
    }

# ---------------------------
# Sanity Check (new)
# ---------------------------
def sanity_check(llm, dataset, sampling_params, global_stats, reward_config, n_samples=5, device="cpu"):
    print("\n[Sanity Check] Comparing Stop vs Natural Rewards ...")
    subset = random.sample(dataset, min(n_samples, len(dataset)))

    for i, item in enumerate(subset):
        q, gt = item["question"], str(item.get("answer", "")).strip()
        messages = [
            {"role": "system", "content": "该助手为DeepSeek-R1，由深度求索公司创造。\n今天是2025年5月28日，星期一。\n"},
            {"role": "user", "content": q}
        ]
        full_prompt = llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        trace = llm.generate([full_prompt], sampling_params, use_tqdm=False)[0].outputs[0]
        if not trace or not getattr(trace, "logprobs", None):
            print(f"[Sanity {i+1}] no logprobs; skipping")
            continue
        logprobs = [safe_extract_logprob(lpd) for lpd in trace.logprobs if lpd]
        if not logprobs:
            print(f"[Sanity {i+1}] empty logprobs; skipping")
            continue

        tracker = ConfTracker(window_size=reward_config["window_size"])
        for lp in logprobs: tracker.push_token_logprob(lp)

        mid = max(1, len(logprobs)//2)
        group_conf_mid = tracker.conf_history[mid-1]
        c_norm = normalize_confidence(group_conf_mid, global_stats)
        stop_r = compute_stop_reward(mid, reward_config["max_tokens"], c_norm, 0.0, reward_config)

        final_ans = extract_answer(getattr(trace, "text", ""))
        is_corr = final_ans is not None and equal_func(final_ans, gt)
        final_conf = tracker.conf_history[-1]
        final_c_norm = normalize_confidence(final_conf, global_stats)
        natural_r = compute_final_noop_reward(is_corr, final_c_norm, reward_config)

        print(f"[Sanity {i+1}] stop_r={stop_r:.2f}, natural_r={natural_r:.2f}, final_conf={final_c_norm:.3f}, correct={is_corr}")

    print("[Sanity Check] Done.\n")

# ---------------------------
# Training loop (REINFORCE)
# ---------------------------
def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Initializing model...")
    llm = DeepThinkLLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size)
    # NOTE: keep logprobs <= engine max (20). Set to 20 here.
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                                     max_tokens=args.max_tokens, logprobs=20)

    state_dim = args.K + 7
    policy = PolicyNet(input_dim=state_dim, hidden=(128,64), action_dim=2).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    reward_config = vars(args)

    with open(args.dataset, "r", encoding="utf-8") as f:
        all_data = [json.loads(line.strip()) for line in f]

    # --- Warmup Phase ---
    print(f"Using all {len(all_data)} questions for warmup.")
    print("\n--- Starting Warmup Phase ---")
    all_conf_values = []
    for item in tqdm(all_data, desc="Warmup"):
        full_prompt = llm.tokenizer.apply_chat_template([{"role":"user", "content": item["question"]}],
                                                        tokenize=False, add_generation_prompt=True)
        trace = llm.generate([full_prompt], sampling_params, use_tqdm=False)[0].outputs[0]
        if not trace or not getattr(trace, "logprobs", None):
            continue
        logprobs = [safe_extract_logprob(lpd) for lpd in trace.logprobs if lpd]
        if not logprobs:
            continue
        tracker = ConfTracker(window_size=reward_config["window_size"])
        for lp in logprobs: tracker.push_token_logprob(lp)
        all_conf_values.extend(tracker.conf_history)

    if all_conf_values:
        conf_array = np.array(all_conf_values)
        g_min, g_max, g_median = np.percentile(conf_array, 1), np.percentile(conf_array, 99), np.median(conf_array)
        global_stats = {'min': float(g_min), 'max': float(g_max), 'median': float(g_median)}
        original_c0 = reward_config["confidence_mid"]
        reward_config["confidence_mid"] = normalize_confidence(g_median, global_stats)
        print("--- Warmup Finished ---")
        print(f"Global Conf Stats: min(1%): {g_min:.4f}, max(99%): {g_max:.4f}, median: {g_median:.4f}")
        print(f"Reward 'confidence_mid' (c0) updated from {original_c0:.4f} to {reward_config['confidence_mid']:.4f}\n")
    else:
        print("Warning: No confidence values collected. Using default normalization.")
        global_stats = {'min': 0.0, 'max': 1.0, 'median': 0.5}

    # --- Sanity check after warmup ---
    sanity_check(llm, all_data, sampling_params, global_stats, reward_config, n_samples=args.sanity_n, device=device)

    # --- Main Training Loop ---
    global_step, episode_stats = 0, deque(maxlen=100)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        total_epoch_reward, total_tokens, total_decisions, total_stops, total_correct = 0.0, 0, 0, 0, 0
        train_data = random.choices(all_data, k=len(all_data))
        n_samples = len(train_data)

        for idx, item in enumerate(tqdm(train_data, desc=f"Epoch {epoch}")):
            question, ground_truth = item["question"], str(item.get("answer","")).strip()
            full_prompt = llm.tokenizer.apply_chat_template([{"role":"user", "content":question}], tokenize=False, add_generation_prompt=True)

            out = run_episode_on_trace(llm, full_prompt, ground_truth, sampling_params,
                                       policy, device, reward_config, args.max_tokens,
                                       args.decision_stride, args.K, args.gamma, global_stats)

            saved_logprobs = out.get("saved_logprobs", [])
            imm_rewards = out.get("rewards", [])  # immediate rewards for each decision
            if not saved_logprobs:
                # no decisions made (maybe very short trace) => skip update
                continue

            total_decisions += out["decisions"]
            total_tokens += out["tokens_used"]

            # For natural completion, compute final reward appended to imm_rewards for analysis,
            # but for policy update we need returns aligned with decisions.
            if out["done_reason"] == "stop":
                total_stops += 1
                final_r = 0.0
                # imm_rewards already includes the stop reward (length == decisions)
            else:
                final_text = out.get("raw_text","")
                final_answer = extract_answer(final_text)
                is_correct = equal_func(final_answer, ground_truth)
                if is_correct:
                    total_correct += 1
                conf_tracker = out.get("conf_tracker")
                final_conf = conf_tracker.conf_history[-1] if (conf_tracker and conf_tracker.conf_history) else 0.0
                final_c_norm = normalize_confidence(final_conf, global_stats)
                final_r = compute_final_noop_reward(is_correct, final_c_norm, reward_config)
                # NOTE: do NOT append final_r to imm_rewards here (we'll include it in return calc)

            # --- Build returns aligned with saved_logprobs (decisions) ---
            # Strategy: set R = final_r, then iter over imm_rewards reversed:
            # R = r_i + gamma * R ; G_i = R
            returns = []
            R = float(final_r)
            for r in reversed(imm_rewards):
                R = float(r) + args.gamma * R
                returns.insert(0, R)

            ep_reward = sum(returns)  # episode metric (discounted returns sum over decisions)
            total_epoch_reward += ep_reward
            episode_stats.append(ep_reward)

            # Convert returns -> tensor and normalize
            returns_t = torch.tensor(returns, dtype=torch.float32).to(device)
            if len(returns_t) > 1 and returns_t.std() > 1e-6:
                returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

            logps = torch.stack(saved_logprobs).squeeze()
            if logps.dim() == 0:
                logps = logps.unsqueeze(0)

            # policy loss: - sum_i logpi(a_i) * G_i
            policy_loss = - (logps * returns_t).sum()

            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()
            global_step += 1

            if (idx + 1) % args.print_every == 0 or idx == 0:
                actions_str = f"N{out['actions_counts'][0]}:S{out['actions_counts'][1]}"
                if out['done_reason'] == 'stop':
                    correct_str = f"future_correct:{int(out.get('future_correct', False))}"
                else:
                    correct_str = f"correct:{int(is_correct)}"
                print(f"[Epoch {epoch} | {idx+1}/{n_samples}] tokens:{out['tokens_used']:<4d} "
                      f"actions:{actions_str:<6} reason:{out['done_reason']:<7s} avg_conf:{out['avg_conf']:.3f} | "
                      f"ep_r:{ep_reward:6.2f} avg_r_100:{np.mean(episode_stats):6.2f} {correct_str}")

            if global_step > 0 and global_step % args.save_every == 0:
                torch.save(policy.state_dict(), args.save_path)
                print(f"[Checkpoint] saved policy to {args.save_path} at step {global_step}")

        # per-epoch summary
        avg_ep_reward = total_epoch_reward / n_samples if n_samples > 0 else 0.0
        print("="*80)
        print(f"Epoch {epoch} finished. time: {(time.time()-t0):.1f}s | avg_ep_reward: {avg_ep_reward:.3f} | "
              f"avg_tokens: {total_tokens / n_samples:.1f} | stops: {total_stops} | corrects: {total_correct}")
        print("="*80)

        # optional sanity check after epoch
        if args.check_every_epoch:
            sanity_check(llm, all_data, sampling_params, global_stats, reward_config, n_samples=args.sanity_n, device=device)

    torch.save(policy.state_dict(), args.save_path)
    print(f"Training finished. Saved final policy to {args.save_path}")

# ---------------------------
# Argument parsing
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train a REINFORCE policy for dynamic stop/continue decisions (with sanity checks).")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="rl_policy_with_check.pt")
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--print_every", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--decision_stride", type=int, default=128)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--window_size", type=int, default=2048)

    # Reward hyperparameters
    parser.add_argument("--alpha_token", type=float, default=4.0)
    parser.add_argument("--confidence_scale", type=float, default=5.0)
    parser.add_argument("--confidence_mid", type=float, default=0.5)
    parser.add_argument("--confidence_k", type=float, default=8.0)
    parser.add_argument("--confidence_s", type=float, default=4.0)
    parser.add_argument("--trend_scale", type=float, default=2.0)
    parser.add_argument("--token_cost_weight", type=float, default=0.01)
    parser.add_argument("--correctness_weight", type=float, default=10.0)
    parser.add_argument("--wrong_penalty", type=float, default=10.0)
    parser.add_argument("--confidence_weight", type=float, default=1.0)
    parser.add_argument("--early_stop_penalty", type=float, default=3.0)
    parser.add_argument("--debug_rewards", action="store_true", help="Print reward components")
    parser.add_argument("--sanity_n", type=int, default=5, help="How many samples in sanity check")
    parser.add_argument("--check_every_epoch", action="store_true", help="Run sanity check after each epoch")

    return parser.parse_args()

# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    args = parse_args()
    train(args)

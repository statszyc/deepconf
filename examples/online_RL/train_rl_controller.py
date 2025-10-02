#!/usr/bin/env python3
"""
Train an RL policy (REINFORCE) that decides Stop / No-op periodically during
generation. This version is fully aligned with the deepconf library's internal
confidence calculation utilities for maximum consistency.
Usage:
    python train_rl_controller_final.py --dataset aime_2025.jsonl --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
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
# ### MODIFIED: Import official utility functions ###
from deepconf.utils import extract_answer, compute_confidence

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
# ### MODIFIED: ConfTracker now works with confidence scores ###
# ---------------------------
class ConfTracker:
    def __init__(self, window_size=2048):
        self.window_size = window_size
        self.buf = deque()
        self.running_sum = 0.0
        self.conf_history = []
        self.global_min = float("inf")

    def push_token_confidence(self, token_confidence: float):
        """Accepts a positive confidence score, not a logprob."""
        self.buf.append(token_confidence)
        self.running_sum += token_confidence
        if len(self.buf) > self.window_size:
            old = self.buf.popleft()
            self.running_sum -= old
        
        denom = len(self.buf) if len(self.buf) > 0 else 1
        # The group confidence is now a direct average of token confidences
        group_conf = self.running_sum / denom
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
    # This function's logic remains the same as it operates on normalized confidence
    # ... (implementation is unchanged)
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

    # ### MODIFIED: Use official `compute_confidence` function ###
    token_confs = []
    if hasattr(gen_obj, "logprobs") and gen_obj.logprobs:
        token_confs = compute_confidence(gen_obj.logprobs)

    text = getattr(gen_obj, "text", "")
    conf_tracker = ConfTracker(window_size=reward_config.get("window_size", 2048))
    tokens_used = 0
    saved_logprobs, rewards = [], []
    actions_counts = {0:0, 1:0}

    for idx, conf in enumerate(token_confs):
        group_conf = conf_tracker.push_token_confidence(conf)
        tokens_used += 1

        if tokens_used > 0 and tokens_used % decision_stride == 0:
            raw_lastK = conf_tracker.last_k_raw(K=K)
            c_mean, c_min, c_std, c_slope = conf_tracker.last_k_stats(K=K)
            global_min_local, tokens_ratio = conf_tracker.get_global_min(), tokens_used / float(max_tokens)
            recent_token_conf = conf

            state_list = raw_lastK + [c_mean, c_min, c_std, c_slope, global_min_local, tokens_ratio, recent_token_conf]
            state = torch.FloatTensor(state_list).unsqueeze(0).to(device)

            logits = controller(state)
            m = torch.distributions.Categorical(logits=logits)
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
                    "done_reason":"stop", "tokens_used": tokens_used, "decisions": len(saved_logprobs),
                    "actions_counts": actions_counts, "avg_conf": np.mean(conf_tracker.conf_history) if conf_tracker.conf_history else 0.0,
                    "saved_logprobs": saved_logprobs, "rewards": rewards, "future_correct": is_correct_future,
                }
            else: # No-op
                rewards.append(compute_noop_immediate(decision_stride, reward_config))

        if tokens_used >= max_tokens:
            break

    # natural completion
    return {
        "done_reason":"natural", "tokens_used": tokens_used, "decisions": len(saved_logprobs),
        "actions_counts": actions_counts, "avg_conf": np.mean(conf_tracker.conf_history) if conf_tracker.conf_history else 0.0,
        "raw_text": text, "saved_logprobs": saved_logprobs, "rewards": rewards, "conf_tracker": conf_tracker,
    }


# ---------------------------
# Training loop (REINFORCE)
# ---------------------------
def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Initializing model...")
    llm = DeepThinkLLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size)
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
    all_group_conf_values = []
    for item in tqdm(all_data, desc="Warmup"):
        full_prompt = llm.tokenizer.apply_chat_template([{"role":"user", "content": item["question"]}],
                                                        tokenize=False, add_generation_prompt=True)
        trace = llm.generate([full_prompt], sampling_params, use_tqdm=False)[0].outputs[0]
        if not trace or not getattr(trace, "logprobs", None):
            continue
            
        # ### MODIFIED: Use official `compute_confidence` and new `ConfTracker` ###
        token_confs = compute_confidence(trace.logprobs)
        if not token_confs:
            continue
        
        tracker = ConfTracker(window_size=reward_config["window_size"])
        for conf in token_confs:
            tracker.push_token_confidence(conf)
        all_group_conf_values.extend(tracker.conf_history)

    if all_group_conf_values:
        conf_array = np.array(all_group_conf_values)
        g_min, g_max, g_median = np.percentile(conf_array, 1), np.percentile(conf_array, 99), np.median(conf_array)
        global_stats = {'min': float(g_min), 'max': float(g_max), 'median': float(g_median)}
        original_c0 = reward_config["confidence_mid"]
        reward_config["confidence_mid"] = normalize_confidence(g_median, global_stats)
        print("--- Warmup Finished ---")
        print(f"Global Group-Conf Stats: min(1%): {g_min:.4f}, max(99%): {g_max:.4f}, median: {g_median:.4f}")
        print(f"Reward 'confidence_mid' (c0) updated from {original_c0:.4f} to {reward_config['confidence_mid']:.4f}\n")
    else:
        print("Warning: No confidence values collected. Using default normalization.")
        global_stats = {'min': 0.0, 'max': 1.0, 'median': 0.5}
        
    # --- Main Training Loop ---
    # ... (The rest of the training loop remains unchanged as it's already robust) ...
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

            saved_logprobs, imm_rewards = out.get("saved_logprobs", []), out.get("rewards", [])
            if not saved_logprobs: continue

            total_decisions += out["decisions"]; total_tokens += out["tokens_used"]
            is_correct = False
            
            if out["done_reason"] == "stop":
                total_stops += 1; rewards_all = imm_rewards
            else:
                final_answer = extract_answer(out.get("raw_text",""))
                is_correct = equal_func(final_answer, ground_truth)
                if is_correct:
                    total_correct += 1

                conf_tracker = out.get("conf_tracker")
                final_conf = conf_tracker.conf_history[-1] if (conf_tracker and conf_tracker.conf_history) else 0.0
                final_c_norm = normalize_confidence(final_conf, global_stats)
                final_r = compute_final_noop_reward(is_correct, final_c_norm, reward_config)

                # ✅ 把 final reward 加到最后一个 No-op 的即时 reward 上
                if len(imm_rewards) > 0:
                    imm_rewards[-1] += final_r
                else:
                    imm_rewards = [final_r]

                rewards_all = imm_rewards
                
            returns = []
            R = 0.0
            for r in reversed(rewards_all):
                R = r + args.gamma * R
                returns.insert(0, R)
            
            ep_reward = sum(returns)
            total_epoch_reward += ep_reward; episode_stats.append(ep_reward)

            returns_t = torch.tensor(returns, dtype=torch.float32).to(device)
            if len(returns_t) > 1 and returns_t.std() > 1e-6:
                returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

            logps = torch.stack(saved_logprobs).squeeze()
            policy_loss = -(logps * returns_t).sum()

            optimizer.zero_grad(); policy_loss.backward(); optimizer.step()
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

        avg_ep_reward = total_epoch_reward / n_samples if n_samples > 0 else 0.0
        print("="*80)
        print(f"Epoch {epoch} finished. time: {(time.time()-t0):.1f}s | avg_ep_reward: {avg_ep_reward:.3f} | "
              f"avg_tokens: {total_tokens / n_samples:.1f} | stops: {total_stops} | corrects: {total_correct}")
        print("="*80)
        # --- Reward Landscape Check ---
        if total_stops > 0 and total_correct >= 0:
            # 简单统计：平均自然完成奖励 vs 平均提前停止奖励
            sample_size = min(5, len(all_data))
            subset = random.sample(all_data, sample_size)

            stop_rewards, natural_rewards = [], []
            for item in subset:
                q, gt = item["question"], str(item.get("answer","")).strip()
                full_prompt = llm.tokenizer.apply_chat_template([{"role":"user","content":q}],
                                                                tokenize=False, add_generation_prompt=True)
                out = llm.generate([full_prompt], sampling_params, use_tqdm=False)[0].outputs[0]
                if not out or not getattr(out,"logprobs",None): 
                    continue
                # 取logprobs → conf tracker
                from deepconf.utils import compute_confidence
                token_confs = compute_confidence(out.logprobs)
                if not token_confs:
                    continue
                tracker = ConfTracker(window_size=reward_config["window_size"])
                for conf in token_confs: 
                    tracker.push_token_confidence(conf)
                
                # mid-stop
                mid = len(token_confs)//2 or 1
                c_norm = normalize_confidence(tracker.conf_history[mid-1], global_stats)
                stop_r = compute_stop_reward(mid, args.max_tokens, c_norm, 0.0, reward_config)
                stop_rewards.append(stop_r)

                # natural
                pred = extract_answer(getattr(out,"text",""))
                is_corr = pred is not None and equal_func(pred, gt)
                final_c_norm = normalize_confidence(tracker.conf_history[-1], global_stats)
                nat_r = compute_final_noop_reward(is_corr, final_c_norm, reward_config)
                natural_rewards.append(nat_r)

            if stop_rewards and natural_rewards:
                print("\n=== Reward Landscape Check (epoch summary) ===")
                print(f"Avg stop reward (mid-trace): {np.mean(stop_rewards):.2f}")
                print(f"Avg natural completion reward: {np.mean(natural_rewards):.2f}")
                if np.mean(natural_rewards) > np.mean(stop_rewards):
                    print("✅ Natural > Stop (good balance)\n")
                else:
                    print("⚠️ Stop >= Natural (risk of premature stopping)\n")

    torch.save(policy.state_dict(), args.save_path)
    print(f"Training finished. Saved final policy to {args.save_path}")

# ---------------------------
# Argument parsing
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train a REINFORCE policy for dynamic stop/continue decisions.")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    parser.add_argument("--dataset", type=str, required=True)
    # ... (rest of the arguments are the same as the final recommended version)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="rl_policy_final.pt")
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--print_every", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=64000)
    parser.add_argument("--decision_stride", type=int, default=128)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--window_size", type=int, default=2048)
    
    # Re-balanced rewards based on diagnostic
    parser.add_argument("--alpha_token", type=float, default=2.5, help="Weight for token saving reward.")
    parser.add_argument("--confidence_scale", type=float, default=3.0, help="Weight for confidence shaping reward.")
    parser.add_argument("--early_stop_penalty", type=float, default=6.0, help="Penalty for stopping a correct trace early.")
    
    # Other reward hyperparameters
    parser.add_argument("--confidence_mid", type=float, default=0.5, help="Updated by warmup.")
    parser.add_argument("--confidence_k", type=float, default=8.0)
    parser.add_argument("--confidence_s", type=float, default=4.0)
    parser.add_argument("--trend_scale", type=float, default=2.0)
    parser.add_argument("--token_cost_weight", type=float, default=0.01)
    parser.add_argument("--correctness_weight", type=float, default=10.0)
    parser.add_argument("--wrong_penalty", type=float, default=10.0)
    parser.add_argument("--confidence_weight", type=float, default=1.0)
    parser.add_argument("--debug_rewards", action="store_true", help="Print reward components.")
    
    return parser.parse_args()

# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    args = parse_args()
    train(args)
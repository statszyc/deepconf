# train_deepconf_rl.py
import os
import json
import random
import math
from collections import Counter, defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange, tqdm
from sklearn.model_selection import train_test_split

# ========== User config ==========
# Paths: you can alternatively build `questions` list directly in memory
QUESTION_FILES_DIR = "trace_data"  # each file is one question jsonl merged format
OUPUT_DATA_DIR = "examples/online_RL/RL_exploration/output"
NUM_CALIBRATION_TRACES = 16
USE_LOW_THRESHOLD = False  # False -> 90th percentile strict baseline
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# RL hyperparams
NUM_TRAIN_EPISODES = 5000
PPO_EPOCHS = 4
PPO_BATCH_STEPS = 2048
GAMMA = 0.99
LR = 3e-4
CLIP_EPS = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_TRACE_STEPS = 2048  # safety cap per trace
BC_PRETRAIN_STEPS = 2000  # behavioral cloning steps (supervised pretrain)

# Reward design (conservative)
R_CORRECT = 10.0
R_INCORRECT = -10.0
C_STEP = 0.02  # cost per step (token/group)
# =================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Utilities ----------
def load_concatenated_json(file_path):
    """Your provided loader — assumes file may contain concatenated json objects."""
    decoder = json.JSONDecoder()
    data_parts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        idx = 0
        while idx < len(content):
            while idx < len(content) and content[idx].isspace():
                idx += 1
            if idx == len(content):
                break
            try:
                obj, end = decoder.raw_decode(content, idx)
                data_parts.append(obj)
                idx = end
            except json.JSONDecodeError:
                break
    if not data_parts:
        return None
    final_data = {k: v for k, v in data_parts[0].items() if k != 'traces'}
    all_traces = [trace for part in data_parts for trace in part.get('traces', [])]
    final_data['traces'] = all_traces
    final_data['num_traces'] = len(all_traces)
    return final_data

def compute_question_baseline_s(traces, num_calib=NUM_CALIBRATION_TRACES, use_low=USE_LOW_THRESHOLD):
    """Compute baseline percentile s for a question using random calibration subset."""
    if len(traces) < num_calib:
        return None
    calib = random.sample(traces, num_calib)
    mins = [min(t['group_confidence']) for t in calib if t.get('group_confidence')]
    if not mins:
        return None
    s_high = np.percentile(mins, 10)
    s_low = np.percentile(mins, 90)
    return s_high if use_low else s_low

# ---------- Environment (per-question) ----------
class DeepConfQuestionEnv:
    """
    Environment that simulates processing all traces for one question.
    An episode corresponds to one question: agent sequentially processes each trace.
    For each trace, agent is repeatedly given state (current conf, min_so_far, progress, warmup_s)
    and chooses action 0=CONTINUE, 1=STOP. If STOP chosen, trace is early-stopped.
    After all traces processed (or budget exhausted), we compute voting outcome and reward.
    """
    def __init__(self, question_data, warmup_s=None, per_step_cost=C_STEP, max_steps_per_trace=MAX_TRACE_STEPS):
        self.question = question_data
        self.traces = question_data['traces']  # list of traces dicts
        self.num_traces = len(self.traces)
        self.warmup_s = warmup_s
        self.per_step_cost = per_step_cost
        self.max_steps_per_trace = max_steps_per_trace
        self.reset()

    def reset(self):
        # shuffle order in which traces are processed (to avoid ordering bias)
        self.order = list(range(self.num_traces))
        random.shuffle(self.order)
        self.cur_trace_idx = 0  # index into order
        # per-trace runtime states
        self.trace_positions = [0] * self.num_traces
        self.trace_min_so_far = [float('inf')] * self.num_traces
        self.trace_kept = [False] * self.num_traces  # whether finished without early stopping
        self.total_steps = 0
        return self._get_state()

    def _get_state(self):
        # if finished all traces, return terminal-like state
        if self.cur_trace_idx >= self.num_traces:
            return np.array([0.0, 0.0, 1.0, float(self.warmup_s or 0.0)], dtype=np.float32)
        tid = self.order[self.cur_trace_idx]
        pos = self.trace_positions[tid]
        conf_curve = self.traces[tid]['group_confidence']
        if pos >= len(conf_curve):
            current_conf = conf_curve[-1] if len(conf_curve) > 0 else 0.0
        else:
            current_conf = conf_curve[pos]
        prev_conf = conf_curve[pos-1] if pos > 0 else current_conf
        delta = current_conf - prev_conf
        min_so_far = min(self.trace_min_so_far[tid], current_conf)
        progress = pos / max(1, len(conf_curve))
        warmup_s = float(self.warmup_s or 0.0)
        # state vector: [current_conf, min_so_far, delta_recent, progress, warmup_s]
        return np.array([current_conf, min_so_far, delta, progress, warmup_s], dtype=np.float32)

    def step(self, action):
        """
        action: 0 continue, 1 stop
        Return: next_state, reward (immediate), done, info
        We give small negative reward per continue; main reward at episode end.
        Here we shape small negative immediately for CONTINUE to reflect cost.
        """
        if self.cur_trace_idx >= self.num_traces:
            # already done
            return self._get_state(), 0.0, True, {}

        tid = self.order[self.cur_trace_idx]
        pos = self.trace_positions[tid]
        conf_curve = self.traces[tid]['group_confidence']
        # immediate cost for taking a step (we count step if continue)
        if action == 0:  # CONTINUE
            # move ahead one step (simulate generating next group)
            # if already at end, mark kept and move to next trace
            if pos >= len(conf_curve)-1:
                self.trace_positions[tid] = len(conf_curve)
                self.trace_kept[tid] = True
                # advance to next trace
                self.cur_trace_idx += 1
            else:
                self.trace_positions[tid] += 1
            self.total_steps += 1
            # update min
            newpos = self.trace_positions[tid]
            curconf = conf_curve[newpos] if newpos < len(conf_curve) else conf_curve[-1]
            self.trace_min_so_far[tid] = min(self.trace_min_so_far[tid], curconf)
            return self._get_state(), -self.per_step_cost, False, {}
        else:  # STOP early
            # mark early-stopped -> not kept (we will not include its answer)
            self.trace_kept[tid] = False
            # advance to next trace
            self.cur_trace_idx += 1
            # small cost for stop action is zero (we already penalize continues)
            return self._get_state(), 0.0, False, {}

    def is_done(self):
        return self.cur_trace_idx >= self.num_traces

    def compute_episode_reward_and_metrics(self):
        """
        After processing all traces (or forced end), compute final vote and reward.
        Voting: majority vote among answers from traces with trace_kept==True.
        If no kept traces, we consider final answer incorrect (penalize).
        Reward: R_CORRECT or R_INCORRECT minus step costs already applied during steps.
        Here return: final_reward, metrics_dict
        """
        kept_answers = []
        for i, t in enumerate(self.traces):
            if self.trace_kept[i]:
                ans = t.get('answer')
                if ans is not None:
                    kept_answers.append(ans)
        if kept_answers:
            # majority vote (simple)
            cnt = Counter(kept_answers)
            final_answer = cnt.most_common(1)[0][0]
        else:
            final_answer = None

        # ground truth: whether any kept trace has is_correct True and answer matches true? We have trace-level is_correct but we need question-level ground truth
        # We assume traces have same ground truth label in data (data['ground_truth'] maybe). Fall back to majority correctness by comparing to trace.is_correct.
        # For safety, infer question-level ground truth by checking traces: if many traces marked is_correct True, take their answer as ground truth.
        gt_answers = [t['answer'] for t in self.traces if t.get('is_correct')]
        if gt_answers:
            ground_truth = Counter(gt_answers).most_common(1)[0][0]
        else:
            ground_truth = None  # unknown

        if final_answer is not None and ground_truth is not None and final_answer == ground_truth:
            reward = R_CORRECT - 0.0  # step costs already subtracted during steps
            correct_flag = True
        else:
            reward = R_INCORRECT
            correct_flag = False

        metrics = {
            'final_answer': final_answer,
            'ground_truth': ground_truth,
            'correct': correct_flag,
            'total_steps': self.total_steps,
            'num_kept_traces': sum(self.trace_kept),
        }
        return reward, metrics

# ---------- Policy / Value networks ----------
class ActorCritic(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value

# ---------- Helper: baseline policy action (for BC pretrain and baseline comparison) ----------
def baseline_action_for_state(state, s_threshold):
    """
    Baseline follows article: stop if min_so_far <= s_threshold.
    State vector: [current_conf, min_so_far, delta, progress, warmup_s]
    """
    min_so_far = state[1]
    if s_threshold is None:
        return 0  # continue if no threshold
    return 1 if min_so_far < s_threshold else 0

# ---------- Build dataset: load all question files ----------
def load_all_questions_from_dir(dirpath):
    qfiles = [os.path.join(dirpath, f) for f in os.listdir(dirpath) if f.endswith(".jsonl") or f.endswith(".json")]
    questions = []
    for f in qfiles:
        q = load_concatenated_json(f)
        if q:
            questions.append(q)
    return questions

# --------- Prepare questions list ----------
print("[*] Loading questions from", QUESTION_FILES_DIR)
questions = load_all_questions_from_dir(QUESTION_FILES_DIR)
print(f"[*] Loaded {len(questions)} questions.")

# If only one question file was provided (per earlier interaction), you can create questions = [data1, data2, ...]
if len(questions) < 5:
    print("[WARN] Less than 5 questions loaded — consider using cross-validation or providing more files.")

# split by question-level
train_qs, test_qs = train_test_split(questions, test_size=0.33, random_state=SEED)
train_qs, val_qs = train_test_split(train_qs, test_size=0.2, random_state=SEED)
print(f"Train questions: {len(train_qs)}, Val: {len(val_qs)}, Test: {len(test_qs)}")

# ---------- Precompute warmup threshold s per question ----------
question_warmup_s = {}
for q in train_qs + val_qs + test_qs:
    s_val = compute_question_baseline_s(q['traces'], NUM_CALIBRATION_TRACES, USE_LOW_THRESHOLD)
    question_warmup_s[q.get('question_id', q.get('question', 'q'))] = s_val

# ---------- Initialize policy ----------
ac = ActorCritic(input_dim=5).to(device)
optimizer = optim.Adam(ac.parameters(), lr=LR)

# ---------- Behavioral cloning pretrain from baseline (optional) ----------
def behavioral_cloning_pretrain(ac_net, questions_pool, steps=BC_PRETRAIN_STEPS):
    print("[*] Starting behavioral cloning pretrain from baseline...")
    bc_opt = optim.Adam(ac_net.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for step in range(steps):
        # sample one question and one state-action from baseline rollout
        q = random.choice(questions_pool)
        env = DeepConfQuestionEnv(q, warmup_s=question_warmup_s.get(q.get('question_id')))
        env.reset()
        # simulate to collect states & baseline actions
        states = []
        actions = []
        while not env.is_done():
            st = env._get_state()
            a = baseline_action_for_state(st, env.warmup_s)
            states.append(st)
            actions.append(a)
            # advance baseline step: baseline chooses continue until stop condition triggers,
            # emulate taking that action by calling env.step(a)
            env.step(a)
        if not states:
            continue
        idx = random.randrange(len(states))
        s = torch.tensor(states[idx], dtype=torch.float32, device=device).unsqueeze(0)
        a = torch.tensor([actions[idx]], dtype=torch.long, device=device)
        logits, _ = ac_net(s)
        loss = loss_fn(logits, a)
        bc_opt.zero_grad()
        loss.backward()
        bc_opt.step()
        if step % 500 == 0 and step > 0:
            print(f"  BC step {step}/{steps}")
    print("[*] BC pretrain finished.")

behavioral_cloning_pretrain(ac, train_qs, steps=1000)

# ---------- PPO-style training (simplified) ----------
def collect_rollout(ac_net, questions_pool, n_steps):
    # collect transitions until n_steps accumulated
    rollouts = []
    steps_collected = 0
    while steps_collected < n_steps:
        q = random.choice(questions_pool)
        env = DeepConfQuestionEnv(q, warmup_s=question_warmup_s.get(q.get('question_id')))
        env.reset()
        episode = []  # list of (state, action, logp, value, reward) per step (per decision)
        while not env.is_done() and steps_collected < n_steps:
            st = env._get_state()
            st_t = torch.tensor(st, dtype=torch.float32, device=device).unsqueeze(0)
            logits, val = ac_net(st_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
            logp = dist.log_prob(torch.tensor(action, device=device)).item()
            value = val.item()
            next_st, r, done, _ = env.step(action)
            episode.append((st, action, logp, value, r))
            steps_collected += 1
            if done:
                break
        # after finishing processing traces for this question, compute final episode reward and metrics
        final_reward, metrics = env.compute_episode_reward_and_metrics()
        # append final_reward to all time steps as terminal shaping; here we put it as last-step reward addition
        # we will add final_reward to the last time step's reward
        if episode:
            # add final_reward to the last stored reward value
            last = episode[-1]
            episode[-1] = (last[0], last[1], last[2], last[3], last[4] + final_reward)
        rollouts.append((episode, metrics))
    return rollouts

def flatten_rollouts(rollouts):
    states = []
    actions = []
    logps = []
    values = []
    rewards = []
    episode_bounds = []
    for ep, metrics in rollouts:
        for (s,a,logp,v,r) in ep:
            states.append(s)
            actions.append(a)
            logps.append(logp)
            values.append(v)
            rewards.append(r)
        episode_bounds.append(len(ep))
    return states, actions, logps, values, rewards, episode_bounds

def compute_returns_and_advantages(rewards, values, gamma=GAMMA):
    # simple returns and advantages (no GAE) for brevity
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    values = torch.tensor(values, dtype=torch.float32, device=device)
    advantages = returns - values
    return returns, advantages

print("[*] Starting PPO-style training...")
for iteration in range(1, NUM_TRAIN_EPISODES+1):
    rollouts = collect_rollout(ac, train_qs, PPO_BATCH_STEPS)
    states, actions, old_logps, values, rewards, episode_bounds = flatten_rollouts(rollouts)
    if len(states) == 0:
        continue
    # convert to tensors
    s_tensor = torch.tensor(states, dtype=torch.float32, device=device)
    a_tensor = torch.tensor(actions, dtype=torch.long, device=device)
    old_logps_tensor = torch.tensor(old_logps, dtype=torch.float32, device=device)
    returns_tensor, adv_tensor = compute_returns_and_advantages(rewards, values, GAMMA)
    adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

    # PPO update epochs
    for _ in range(PPO_EPOCHS):
        # get logits and values from current policy
        logits, vals = ac(s_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        logps = dist.log_prob(a_tensor)
        ratio = torch.exp(logps - old_logps_tensor)
        surr1 = ratio * adv_tensor
        surr2 = torch.clamp(ratio, 1.0-CLIP_EPS, 1.0+CLIP_EPS) * adv_tensor
        policy_loss = -torch.min(surr1, surr2).mean()
        # value loss
        value_loss = ((returns_tensor - vals) ** 2).mean()
        # entropy bonus
        entropy = dist.entropy().mean()
        loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # evaluate on validation set periodically
    if iteration % 50 == 0:
        # validation rollouts (deterministic policy: choose argmax)
        val_metrics = []
        for q in val_qs:
            env = DeepConfQuestionEnv(q, warmup_s=question_warmup_s.get(q.get('question_id')))
            env.reset()
            while not env.is_done():
                st = env._get_state()
                st_t = torch.tensor(st, dtype=torch.float32, device=device).unsqueeze(0)
                logits, _ = ac(st_t)
                action = torch.argmax(logits, dim=-1).item()
                env.step(action)
            _, metrics = env.compute_episode_reward_and_metrics()
            val_metrics.append(metrics)
        # aggregate metrics
        acc = np.mean([1.0 if m['correct'] else 0.0 for m in val_metrics])
        avg_steps = np.mean([m['total_steps'] for m in val_metrics])
        kept = np.mean([m['num_kept_traces'] for m in val_metrics])
        print(f"[Iter {iteration}] Val Acc={acc:.3f}, AvgSteps={avg_steps:.1f}, KeptTraces={kept:.1f}")

print("[*] Training finished. Saving policy...")
torch.save(ac.state_dict(), os.path.join(OUPUT_DATA_DIR, "deepconf_rl_policy.pth"))

# ---------- Final evaluate on test set ----------
print("[*] Final evaluation on test set...")
test_metrics = []
for q in test_qs:
    env = DeepConfQuestionEnv(q, warmup_s=question_warmup_s.get(q.get('question_id')))
    env.reset()
    while not env.is_done():
        st = env._get_state()
        st_t = torch.tensor(st, dtype=torch.float32, device=device).unsqueeze(0)
        logits, _ = ac(st_t)
        action = torch.argmax(logits, dim=-1).item()
        env.step(action)
    r, metrics = env.compute_episode_reward_and_metrics()
    test_metrics.append(metrics)

test_acc = np.mean([1.0 if m['correct'] else 0.0 for m in test_metrics])
test_avg_steps = np.mean([m['total_steps'] for m in test_metrics])
print(f"Test Acc={test_acc:.3f}, AvgSteps={test_avg_steps:.1f}")

# save test details
with open(os.path.join(OUPUT_DATA_DIR, "rl_test_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(test_metrics, f, ensure_ascii=False, indent=2)

print("[*] Done.")

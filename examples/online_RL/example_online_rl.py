"""
Creaated by Yingchuan Zhang, UGA Stat
"""
import json
import pickle
import argparse
from datetime import datetime
from deepconf import DeepThinkLLM
from vllm import SamplingParams
import time
from dynasor.core.evaluator import math_equal
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# ============= 强化学习（RL）核心组件 =============

class RLController(nn.Module):
    """
    A simple Feed-Forward Neural Network to act as the RL policy.
    It takes the state from the environment and outputs an action.
    """
    def __init__(self, state_dim, action_dim):
        super(RLController, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.optimizer = None

    def forward(self, state):
        return self.network(state)

    def select_action(self, state):
        """Selects an action based on the current policy (epsilon-greedy for simplicity)."""
        if random.random() < 0.1:  # Epsilon-greedy exploration
            return random.randint(0, self.network[-1].out_features - 1)
        with torch.no_grad():
            q_values = self.forward(torch.FloatTensor(state))
            return torch.argmax(q_values).item()

    def update_policy(self, replay_buffer, batch_size=32):
        """Updates the policy network using a batch from the replay buffer."""
        if len(replay_buffer) < batch_size:
            return
        
        if self.optimizer is None:
            # Lazy optimizer initialization
            self.optimizer = optim.Adam(self.parameters(), lr=0.001)

        batch = replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        # Placeholder for actual Q-learning or Policy Gradient update logic
        # For this example, we'll use a simplified loss
        
        q_values = self.forward(states)
        action_q_values = q_values.gather(1, actions.unsqueeze(1))
        
        loss = nn.MSELoss()(action_q_values.squeeze(), rewards) # Simplified loss for demonstration
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class RLEnvironment:
    """
    Wraps the LLM reasoning process to create an environment for the RL agent.
    """
    def __init__(self, llm, tokenizer, ground_truth, prompt, sampling_params, reward_config):
        self.llm = llm
        self.tokenizer = tokenizer
        self.ground_truth = ground_truth
        self.prompt = prompt
        self.sampling_params = sampling_params
        self.reward_config = reward_config
        
        self.current_trace = ""
        self.token_count = 0
        self.confidence_signals = []
        self.consensus_stats = {} # Placeholder for consensus stats

    def reset(self):
        """Resets the environment for a new episode (a new reasoning trace)."""
        self.current_trace = ""
        self.token_count = 0
        self.confidence_signals = []
        return self._get_state()

    def _get_state(self):
        """Computes the current state from confidence signals and other metrics."""
        # This should be a fixed-size vector.
        # Example state: [avg_confidence, min_confidence, step_index, consensus_metric]
        if not self.confidence_signals:
            return [0.0, 0.0, 0, 0.0]
        
        avg_conf = np.mean(self.confidence_signals)
        min_conf = np.min(self.confidence_signals)
        step_index = len(self.current_trace.split()) # Simple step index
        consensus_metric = 0.5 # Placeholder
        return [avg_conf, min_conf, step_index, consensus_metric]

    def step(self, action: int):
        """
        Executes an action and returns the new state, reward, and done flag.
        Action mapping: 0: Continue, 1: Stop, 2: Accept, 3: Terminate
        """
        done = False
        reward = 0.0
        
        if action == 0: # Continue
            # Generate more tokens
            # In a real implementation, this would generate a small chunk of text
            # For simplicity, we simulate this step.
            new_output = self.llm.generate(
                self.prompt + self.current_trace, 
                self.sampling_params, 
                use_tqdm=False
            )[0].outputs[0]
            
            new_text_chunk = new_output.text[len(self.current_trace):]
            self.current_trace += new_text_chunk
            self.token_count += len(new_output.token_ids)
            
            # Update confidence signals
            if new_output.logprobs:
                new_conf = np.mean([-lp.logprob for lp in new_output.logprobs[0].values()])
                self.confidence_signals.append(new_conf)

            # Intermediate reward is typically 0, reward is given at the end
            reward = -self.reward_config['token_cost_weight'] * len(new_output.token_ids)

        elif action in [1, 2, 3]: # Stop, Accept, Terminate
            done = True
            final_answer = quick_parse(extract_answer(self.current_trace))
            is_correct = equal_func(final_answer, self.ground_truth)
            
            # Calculate final reward
            correctness_reward = self.reward_config['correctness_weight'] if is_correct else 0
            confidence_score = np.mean(self.confidence_signals) if self.confidence_signals else 0
            token_cost = self.reward_config['token_cost_weight'] * self.token_count
            
            reward = correctness_reward + self.reward_config['confidence_weight'] * confidence_score - token_cost

        next_state = self._get_state()
        return next_state, reward, done, {}


class ReplayBuffer:
    """A simple replay buffer to store experiences."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# ============= HELPER FUNCTIONS (UNCHANGED) =============

def prepare_prompt(question: str, tokenizer, model_type: str = "deepseek") -> str:
    if model_type == "deepseek":
        messages = [
            {"role": "system", "content": "该助手为DeepSeek-R1，由深度求索公司创造。\n今天是2025年5月28日，星期一。\n"},
            {"role": "user", "content": question}
        ]
    else:
        messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def quick_parse(text: str) -> str:
    if text is None: return ""
    while '\\text{' in text:
        start = text.find('\\text{')
        end = text.find('}', start)
        if end == -1: break
        text = text[:start] + text[start + 6:end] + text[end + 1:]
    return text

def extract_answer(text: str) -> str:
    if "boxed" in text:
        ans = text.split("boxed")[-1]
        if not ans: return ""
        if ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{": stack += 1
                elif c == "}": stack -= 1
                if stack == 0: break
                a += c
            return a.strip()
        else:
            return ans.split("$")[0].strip()
    return ""

def equal_func(answer: str, ground_truth: str) -> bool:
    answer = quick_parse(answer)
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    return math_equal(answer, ground_truth)


# ============= MAIN SCRIPT =============

def main():
    parser = argparse.ArgumentParser(description='DeepThinkLLM Online Mode with RL Controller')
    # Model and Data Arguments
    parser.add_argument('--model', type=str, default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", help='Model path or name')
    parser.add_argument('--dataset', type=str, default="aime_2025.jsonl", help='Dataset file path')
    parser.add_argument('--qid', type=int, required=True, help='Question ID to process (0-based index)')
    parser.add_argument('--rid', type=str, default="rl_online_run", help='Run ID for identification')
    parser.add_argument('--output_dir', type=str, default="outputs_rl", help='Output directory for results')

    # Generation Arguments
    parser.add_argument('--total_budget', type=int, default=256, help='Total trace budget')
    parser.add_argument('--max_tokens', type=int, default=8192, help='Maximum tokens per generation')
    parser.add_argument('--temperature', type=float, default=0.6, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p sampling parameter')
    parser.add_argument('--top_k', type=int, default=0, help='Top-k sampling parameter')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Tensor parallel size for model')
    parser.add_argument('--model_type', type=str, default="deepseek", choices=["deepseek", "gpt"], help='Model type for prompt formatting')

    # RL Controller Arguments
    parser.add_argument('--rl_episodes', type=int, default=100, help='Number of episodes to train the RL agent')
    parser.add_argument('--reward_correctness_weight', type=float, default=1.0, help='Weight for correctness in reward function')
    parser.add_argument('--reward_confidence_weight', type=float, default=0.1, help='Weight for confidence in reward function')
    parser.add_argument('--reward_token_cost_weight', type=float, default=0.001, help='Weight for token cost in reward function')

    args = parser.parse_args()

    # --- 1. Initialization ---
    print("--- Initializing ---")
    # Load dataset
    with open(args.dataset, 'r', encoding='utf-8') as file:
        data = [json.loads(line.strip()) for line in file]
    question_data = data[args.qid]
    question = question_data['question']
    ground_truth = str(question_data.get('answer', '')).strip()

    # Initialize LLM
    deep_llm = DeepThinkLLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size, enable_prefix_caching=True)
    prompt = prepare_prompt(question, deep_llm.tokenizer, args.model_type)

    # Initialize RL Components
    state_dim = 4 # [avg_conf, min_conf, step_idx, consensus_metric]
    action_dim = 4 # [Continue, Stop, Accept, Terminate]
    rl_controller = RLController(state_dim, action_dim)
    replay_buffer = ReplayBuffer(10000)
    
    reward_config = {
        'correctness_weight': args.reward_correctness_weight,
        'confidence_weight': args.reward_confidence_weight,
        'token_cost_weight': args.reward_token_cost_weight,
    }
    sampling_params_step = SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, max_tokens=50, logprobs=20) # for single step
    rl_environment = RLEnvironment(deep_llm, deep_llm.tokenizer, ground_truth, prompt, sampling_params_step, reward_config)
    
    print(f"Processing question {args.qid} with RL Controller...")

    # --- 2. RL Training and Inference Loop ---
    print("--- Starting RL Loop ---")
    start_time = time.time()
    all_traces = []
    total_tokens = 0

    for episode in range(args.rl_episodes):
        state = rl_environment.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = rl_controller.select_action(state)
            next_state, reward, done, _ = rl_environment.step(action)
            
            replay_buffer.push(state, action, reward, next_state, done)
            rl_controller.update_policy(replay_buffer)
            
            state = next_state
            episode_reward += reward

        # Store trace and stats
        all_traces.append({
            'text': rl_environment.current_trace,
            'final_reward': episode_reward,
            'extracted_answer': quick_parse(extract_answer(rl_environment.current_trace))
        })
        total_tokens += rl_environment.token_count

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{args.rl_episodes}, Total Reward: {episode_reward:.3f}")

    end_time = time.time()
    print(f"RL loop completed in {end_time - start_time:.2f}s")
    
    # --- 3. Evaluation and Saving ---
    print("--- Evaluating Final Results ---")
    # In a full implementation, you would now use the trained policy for pure inference
    # and perform voting on the 'accepted' traces. For this example, we'll just show the generated traces.
    
    correct_count = sum(1 for trace in all_traces if equal_func(trace['extracted_answer'], ground_truth))
    print(f"Final Accuracy over {len(all_traces)} generated traces: {correct_count / len(all_traces):.2%}")
    
    # Save results
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    result_data = {
        'question': question,
        'ground_truth': ground_truth,
        'qid': args.qid,
        'run_id': args.rid,
        'all_traces': all_traces,
        'total_tokens': total_tokens,
        'total_time': end_time - start_time,
        'config': vars(args)
    }
    
    result_filename = f"{args.output_dir}/deepthink_online_rl_qid{args.qid}_rid{args.rid}_{timestamp}.pkl"
    with open(result_filename, 'wb') as f:
        pickle.dump(result_data, f)
    
    print(f"\nResults saved to {result_filename}")

if __name__ == "__main__":
    main()
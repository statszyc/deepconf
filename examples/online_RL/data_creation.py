"""
Generate trace-confidence dataset for RL training (per-question mode).

This script runs DeepThinkLLM in an offline batch mode to generate multiple
reasoning traces for a single question. It manually calculates the
group-level (sliding window) confidence scores for each trace, evaluates
the correctness of the extracted answer, and saves the structured data
to a JSONL file.

The script is designed to be compatible with SLURM array jobs by accepting a
question ID (--qid) as a command-line argument.

Usage example (per question on SLURM):
  srun python generate_trace_dataset.py --dataset aime25.jsonl --qid $SLURM_ARRAY_TASK_ID --budget 256 --output_dir ./trace_data
"""

import os
import json
import pickle
import argparse
from datetime import datetime
from vllm import SamplingParams

# -------------------------------
# Import necessary components from the deepconf project
# -------------------------------
from deepconf import DeepThinkLLM
from deepconf.utils import compute_least_grouped # 关键：导入group confidence计算函数
from dynasor.core.evaluator import math_equal

# -------------------------------
#  Helper functions
# -------------------------------

def quick_parse(text: str) -> str:
    """Simplify LaTeX-style answers by removing \\text{...} wrappers."""
    if not isinstance(text, str):
        return ""
    if '\\text{' in text and '}' in text:
        while '\\text{' in text:
            start = text.find('\\text{')
            end = text.find('}', start)
            if start == -1 or end == -1:
                break
            text = text[:start] + text[start + 6:end] + text[end + 1:]
    return text

def equal_func(answer: str, ground_truth: str) -> bool:
    """Check if the model's answer is equivalent to the ground truth."""
    if not answer or not ground_truth:
        return False
        
    answer = quick_parse(answer)
    
    if (
        len(answer) == 1 and answer.isalpha()
        and len(ground_truth) == 1 and ground_truth.isalpha()
    ):
        return answer.lower() == ground_truth.lower()
    else:
        try:
            # Use math_equal for robust mathematical expression comparison
            return math_equal(answer, ground_truth)
        except Exception:
            # Fallback to string comparison for other cases
            return str(answer).strip() == str(ground_truth).strip()

def prepare_prompt(question: str, tokenizer, model_type: str = "deepseek", reasoning_effort: str = "high"):
    """Prepare the prompt using the appropriate chat template for the model."""
    if model_type == "deepseek":
        messages = [
            {"role": "system", "content": "该助手为DeepSeek-R1，由深度求索公司创造。\n今天是2025年5月28日，星期一。\n"},
            {"role": "user", "content": question}
        ]
        # Deepseek's template does not use 'reasoning_effort'
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else: # Default to GPT-like models
        messages = [{"role": "user", "content": question}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=reasoning_effort
        )

# -------------------------------
#  Main logic
# -------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate a trace dataset for RL training")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", help="Model name or path.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for vLLM.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the input JSONL dataset.")
    parser.add_argument("--qid", type=int, required=True, help="Question ID (0-based index) to process.")
    parser.add_argument("--budget", type=int, default=256, help="Number of traces to generate per question.")
    parser.add_argument("--window_size", type=int, default=2048, help="Sliding window size for group confidence calculation.")
    parser.add_argument("--max_tokens", type=int, default=64000, help="Maximum new tokens per trace.")
    parser.add_argument("--model_type", type=str, default="deepseek", choices=["gpt", "deepseek"], help="Type of model for prompt formatting.")
    parser.add_argument("--reasoning_effort", type=str, default="high", help="Reasoning effort parameter for GPT models.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter.")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling parameter (0 means disabled).")
    parser.add_argument("--output_dir", type=str, default="trace_data", help="Directory to save the output files.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- Load Data ---
    print(f"[INFO] Loading dataset from {args.dataset}")
    try:
        with open(args.dataset, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[ERROR] Could not read dataset file: {e}")
        return

    if args.qid < 0 or args.qid >= len(data):
        print(f"[ERROR] Invalid qid={args.qid}. Dataset has {len(data)} entries (0 to {len(data)-1}).")
        return

    question_entry = data[args.qid]
    question = question_entry["question"]
    ground_truth = str(question_entry.get("answer", "")).strip()
    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    question_id_str = f"{dataset_name}_{args.qid:03d}"

    print(f"[INFO] Processing QID={args.qid} ({question_id_str}): {question[:80]}...")
    
    # --- Initialize Model ---
    deep_llm = DeepThinkLLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=True
    )

    # --- Prepare Prompt and Sampling (修正之处) ---
    prompt = prepare_prompt(question, deep_llm.tokenizer, args.model_type, args.reasoning_effort)
    
    # 在 SamplingParams 中移除 n 参数
    # n 的值将由 deepthink 的 budget 参数唯一控制
    sampling_params = SamplingParams(
        # n=args.budget,  <-- 移除或注释掉这一行
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        logprobs=20, 
    )

    # --- Generate Traces in Offline Mode ---
    print(f"[INFO] Generating {args.budget} traces...")
    
    # budget 参数是控制生成总数的唯一来源
    result = deep_llm.deepthink(
        prompt=prompt,
        mode="offline",
        budget=args.budget, 
        window_size=args.window_size,
        sampling_params=sampling_params,
        compute_multiple_voting=False
    )

    # --- Process and Extract Trace Data ---
    print("[INFO] Processing generated traces...")
    processed_traces = []
    for i, trace in enumerate(result.all_traces):
        # 1. Get the raw token-level confidences
        token_confidences = trace.get("confs", [])
        
        # 2. **Manually compute group confidences using the imported utility function**
        group_confidences = compute_least_grouped(token_confidences, group_size=args.window_size)
        
        # 3. Get tokens by converting token IDs
        token_ids = trace.get("token_ids", [])
        tokens = deep_llm.tokenizer.convert_ids_to_tokens(token_ids)
        
        # 4. Get the answer and check correctness
        answer_text = trace.get("extracted_answer", None)
        is_correct = equal_func(answer_text, ground_truth)

        processed_traces.append({
            "trace_id": i,
            "tokens": tokens,
            "group_confidence": group_confidences,
            "answer": answer_text,
            "is_correct": is_correct
        })

    # --- Prepare Final Output and Save ---
    output_data = {
        "question_id": question_id_str,
        "question": question,
        "ground_truth": ground_truth,
        "num_traces": len(processed_traces),
        "traces": processed_traces,
    }

    # Save as a single JSONL file for the question, which is easier to handle
    output_path = os.path.join(args.output_dir, f"{question_id_str}.jsonl")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(output_data) + "\n")
        print(f"[SUCCESS] Saved {len(processed_traces)} traces to {output_path}")
    except IOError as e:
        print(f"[ERROR] Failed to write to output file: {e}")

if __name__ == "__main__":
    main()
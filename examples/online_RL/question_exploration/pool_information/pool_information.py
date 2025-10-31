import argparse
from deepconf import DeepThinkLLM
from vllm import SamplingParams, LLM

# 假设 deep_llm 实例和第一轮的 traces 数据已经准备好
# deep_llm = DeepThinkLLM(model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
# traces = load_concatenated_json('trace_data/aime_2025_0_full.jsonl')['traces']
import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
random.seed(13)

######### LOAD DATA #########
# --- 1. 配置 ---
# --- 2. 将工作目录设置为项目根目录 (如果需要) ---
os.chdir(os.path.expanduser('~/Projects/Method/deepconf'))

def load_concatenated_json(file_path):
    """
    Reads a file containing one or more concatenated JSON objects
    and merges them into a single, valid data structure.
    """
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

    # Merge the parts
    final_data = {k: v for k, v in data_parts[0].items() if k != 'traces'}
    all_traces = [trace for part in data_parts for trace in part.get('traces', [])]
    
    final_data['traces'] = all_traces
    final_data['num_traces'] = len(all_traces)
    
    return final_data

def build_follow_up_question(current_answer, current_count, other_answers, answer_counts):
    """
    current_answer: str
    current_count: int
    other_answers: dict[str, str]  -> 每个答案对应的 reasoning 简要
    answer_counts: dict[str, int]  -> 每个答案对应的 trace 数
    """
    # 1️⃣ 整理其他答案内容
    other_lines = []
    for i, (ans, text) in enumerate(other_answers.items(), start=1):
        # count_info = f"(supported by {answer_counts.get(ans, 1)} traces)"
        other_lines.append(f"{i}. Candidate answer {ans}:\n{text.strip()}\n")
    other_answers_text = "\n".join(other_lines)

    # 2️⃣ 主体 prompt
    follow_up_question = f"""
Previously, you concluded the final answer was: {current_answer}.

Below are other candidate answers produced by other independent reasoning traces, each with a short summary of its reasoning and the number of traces that reached it:
{other_answers_text}

Note:
- The number of traces supporting an answer indicates how many independent reasoning paths arrived at that conclusion.
- However, *frequency does not guarantee correctness*. Use this information only as auxiliary evidence.

Carefully examine the reasoning processes of the other answers, especially where they differ from yours.
Then thoroughly re-check your own logic for possible calculation mistakes or conceptual oversights.

After reconsideration:
- If you keep your previous answer, reply exactly:  
  `I still choose: {current_answer}`
- If you change your mind, reply exactly:  
  `I now choose: <new_answer>`

Finally, output your decision in the exact format:  
**FINAL ANSWER: \\boxed{{X}}**
"""
    return follow_up_question.strip()

def parse_args():
    parser = argparse.ArgumentParser(description="Run follow-up self-check experiment")

    parser.add_argument("--model", type=str, default="deepseek-r1-qwen-8b",
                        help="LLM model name to use.")
    parser.add_argument("--question_id", type=int, required=True,
                        help="AIME problem ID (for logging).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save output traces.")

    return parser.parse_args()
# --- 4. Load the Target File ---
def main():
    args = parse_args()
    file_path = f'trace_data/aime_2025_{args.question_id}_full.jsonl'
    data = load_concatenated_json(file_path)

    if data:
        traces = data.get('traces', [])
        question = data.get('question', [])
        print(f"\nSuccessfully loaded and merged data for QID {args.question_id}.")
        print(f"Total traces found: {len(traces)}")
    else:
        print(f"\nERROR: Could not load data from {file_path}")
        traces = []

    ######### SCREENING #########

    # --- Early Stopping Simulation Configuration ---
    NUM_CALIBRATION_TRACES = 16
    USE_LOW_THRESHOLD = False  # True: 10% percentile (lenient), False: 90% percentile (strict)
    NUM_TRACES_TO_PLOT = 250
    random.seed(13)

    # --- Calculate Threshold ---
    s = None
    if len(traces) >= NUM_CALIBRATION_TRACES:
        calibration_traces = random.sample(traces, NUM_CALIBRATION_TRACES)
        lowest_confs = [min(t['group_confidence']) for t in calibration_traces if t['group_confidence']]
        if lowest_confs:
            s_high = np.percentile(lowest_confs, 10)
            s_low = np.percentile(lowest_confs, 90)
            s = s_high if USE_LOW_THRESHOLD else s_low
            print(f"--- Early Stopping Simulation ---")
            print(f"Threshold computed from {len(lowest_confs)} calibration samples.")
            print(f"  - High Threshold (10th percentile): {s_high:.4f}")
            print(f"  - Low Threshold (90th percentile): {s_low:.4f}")
            print(f"--- Active Threshold s = {s:.4f} ---")

    # --- Plotting and Confusion Matrix Calculation ---
    if s is not None:
        TP, FP, TN, FN = 0, 0, 0, 0

        predicted_good = []  # ✅ 收集未被截断的 trace
        predicted_bad = []

        traces_to_plot = random.sample(traces, min(len(traces), NUM_TRACES_TO_PLOT))

        for trace in traces:
            actual_is_correct = trace['is_correct']
            conf_curve = trace['group_confidence']

            stop_indices = np.where(np.array(conf_curve) < s)[0] if conf_curve else []
            predicted_as_bad = len(stop_indices) > 0

            # ✅ 保存分类结果
            if predicted_as_bad:
                predicted_bad.append(trace)
            else:
                predicted_good.append(trace)
            # Update confusion matrix
            if not actual_is_correct and predicted_as_bad: TP += 1
            elif actual_is_correct and predicted_as_bad: FP += 1
            elif actual_is_correct and not predicted_as_bad: TN += 1
            elif not actual_is_correct and not predicted_as_bad: FN += 1
        print("Positive Class: 'Bad Trace' (Incorrect Answer)")
        print("Negative Class: 'Good Trace' (Correct Answer)\n")
        print(f"{'':<15}{'Predicted: Bad':<20}{'Predicted: Good'}")
        print(f"{'Actual: Bad':<15}{TP:<20}(TP){FN:<20}(FN)")
        print(f"{'Actual: Good':<15}{FP:<20}(FP){TN:<20}(TN)")
        print("-" * 60)

        # Calculate and print metrics
        accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        print(f"Accuracy: {accuracy:.2%}")
        print(f"Precision (for bad traces): {precision:.2%}")
        print(f"Recall (for bad traces): {recall:.2%}")

        # --- ✅ Show Predicted Good Answers ---
        print("\n--- Answers from Predicted Good Traces ---")
        good_answers = [t["answer"] for t in predicted_good if t.get("answer") is not None]
        if good_answers:
            from collections import Counter
            counts = Counter(good_answers)
            for ans, cnt in counts.most_common(15):  # 只打印前 15 个最常见的
                print(f"{ans!r:40s}  →  {cnt} traces")
            # ✅ 仅保留前 5 个最常见答案及其所有 trace
            # 查询traces数量中最大的五个数
            answer_number_sorted = sorted(set(counts.values()), reverse=True)
            print(answer_number_sorted[:min(5, len(answer_number_sorted))])
            top5_answers = [ans for ans, cnt in counts.items() if cnt in answer_number_sorted[:min(5, len(answer_number_sorted))]]
            filtered_traces = [t for t in predicted_good if t.get("answer") in top5_answers]

            print(f"\n✅ Kept {len(filtered_traces)} traces belonging to top 5 answers, answers are: {top5_answers}")
        else:
            print("No predicted good traces found.")
        predicted_good = filtered_traces
    ########### INITIALIZE DEEP THINK LLM ###########
    # TODO: 不能同时启用两个，不然显存不够
    deep_llm = DeepThinkLLM(model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    summarizer = LLM(model="Qwen/Qwen2.5-Math-7B-Instruct")  # reasoning summarizer
    # --- 使用已有的 JSONL 数据进行追问 ---
    # 1. 提取第一轮的上下文
    # 假设我们选择第一条 trace (trace_id: 0) 作为上下文

    all_traces_2=[]
    if good_answers:
        # 所有候选答案
        # all_candidate_answers = top5_answers

        print("\n=== Generating Self-Check Prompts for Each Candidate Answer ===")

        for current_answer in top5_answers:
            # 取该答案对应的good traces
            current_answer_number = counts[current_answer]
            same_answer_traces = [t for t in predicted_good if t.get("answer") == current_answer]
            if not same_answer_traces:
                continue

            # 随机抽一个trace作为base
            base_trace = random.choice(same_answer_traces)
            trace_1_tokens = base_trace.get("tokens", "(Reasoning trace missing...)")
            # 其他每个答案都抽一个trace，并且从他们的回答中用split_thinking_and_answer提取回答的部分，返回的结果是{}
            other_answers = [ans for ans in top5_answers if ans != current_answer]
            other_answers_text = {}
        #     for ans in other_answers:
        #         ans_trace = random.choice([t for t in predicted_good if t.get("answer") == ans])
        #         ans_text = deep_llm.tokenizer.convert_tokens_to_string(ans_trace.get("tokens", "(Reasoning trace missing...)"))
        #         _, ans_only = split_thinking_and_answer(ans_text)
        #         other_answers_text[ans] = ans_only

        # for current_answer in top5_answers:
        #     same_answer_traces = [t for t in filtered_traces if t["answer"] == current_answer]
        #     if not same_answer_traces:
        #         continue
        #     base_trace = random.choice(same_answer_traces)
        #     trace_1_tokens = base_trace.get("tokens", [])
        #     trace_1_string = deep_llm.tokenizer.convert_tokens_to_string(trace_1_tokens)

        #     # 🔹 对其他答案做 summary
        #     other_answers = [a for a in top5_answers if a != current_answer]
        #     other_answers_text = {}
            for ans in other_answers:
                ans_trace = random.choice([t for t in filtered_traces if t["answer"] == ans])
                ans_tokens = ans_trace.get("tokens", [])
                ans_text = deep_llm.tokenizer.convert_tokens_to_string(ans_tokens)
                # reasoning, _ = split_thinking_and_answer(ans_text)

                summarizer_prompt = f"""
    You are given a reasoning trace produced by a math model.
    Please summarize the *key reasoning steps* clearly and concisely, keeping the essential logic
    but omitting redundant details or exploratory thoughts.

    Requirements:
    - Write in complete sentences.
    - Keep the total summary under **500 tokens**.
    - End with the model’s **final numerical or symbolic answer**, enclosed in LaTeX format:
    **Final Answer: \\boxed{{X}}**
    - If the reasoning trace does not contain a valid answer, output: **Final Answer: \\boxed{{None}}**
    - Do not include any unrelated text or commentary.

    Here is the reasoning trace:
    ---
    {ans_text}
    ---
    Now produce your concise summary:
    """

                summary_output = summarizer.generate(summarizer_prompt, SamplingParams(max_tokens=512, temperature=0.4))
                summary_text = summary_output[0].outputs[0].text.strip()
                other_answers_text[ans] = summary_text

            # 2. **关键步骤**: 将 tokens 列表转换回字符串
            #    我们使用 tokenizer 的 convert_tokens_to_string 方法
            trace_1_string = deep_llm.tokenizer.convert_tokens_to_string(trace_1_tokens)

            # 3. 准备我们的追问
            follow_up_question = build_follow_up_question(current_answer, current_answer_number, other_answers_text, dict(counts.most_common(5)))
            # 4. 构建包含完整历史的消息列表
            #    [user_q1, assistant_a1, user_q2]
            messages_turn_2 = [
                {"role": "system", "content": "该助手为DeepSeek-R1，由深度求索公司创造。\n今天是2025年5月28日，星期一。\n"},
                {"role": "user", "content": question},
                {"role": "assistant", "content": trace_1_string}, # 使用我们刚刚转换好的字符串
                {"role": "user", "content": follow_up_question}
            ]

            # 5. 应用聊天模板，生成包含上下文的第二轮 prompt
            prompt_2 = deep_llm.tokenizer.apply_chat_template(
                messages_turn_2,
                tokenize=False,
                add_generation_prompt=True
            )

            # 6. 调用 vLLM 获取追问的回答
            sampling_params = SamplingParams(temperature=0.6, max_tokens=64000)
            outputs_2 = deep_llm.generate(prompt_2, sampling_params)
            trace_2 = outputs_2[0].outputs[0].text
            all_traces_2.append({
                "base_trace_id": base_trace["trace_id"],
                "base_summary": trace_1_string, 
                "base_answer": current_answer,
                "other_answers": other_answers,
                "trace_2": trace_2
            })
    # write to file
    os.makedirs(args.output_dir, exist_ok=True)
    with open(f'{args.output_dir}/aime_2025_{args.question_id}_deepconflow_self_check.jsonl', 'w', encoding='utf-8') as f:
        for item in all_traces_2:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
if __name__ == '__main__':
    main()
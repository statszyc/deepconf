import json
from datasets import load_dataset

# Load dataset
dataset = load_dataset("MathArena/aime_2025", split="train")

# Convert to JSONL
with open("aime_2025.jsonl", "w", encoding="utf-8") as f:
    for example in dataset:
        entry = {
            "question": example["problem"],
            "answer": str(example["answer"])
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Converted {len(dataset)} examples to aime_2025.jsonl")

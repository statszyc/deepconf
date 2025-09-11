# DeepThinkLLM

A powerful framework for enhanced LLM reasoning with confidence-based methods and multiple voting strategies. Supports both online (confidence-based early stopping) and offline (batch generation) modes.

## Features

- **Dual Processing Modes**: Online mode with confidence-based early stopping, and offline mode with batch generation
- **Multiple Voting Algorithms**: 7 different voting strategies including confidence-weighted voting
- **Memory Optimized**: Stores only confidence values, not full logprobs
- **Separated Concerns**: Prompt processing and evaluation handled separately from inference
- **Comprehensive Analysis**: Built-in performance analysis and comparison tools
- **Flexible Configuration**: Support for different model types and sampling parameters

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from deepconf import DeepThinkLLM
from deepconf import prepare_prompt, equal_func

# Initialize model
deep_llm = DeepThinkLLM(model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")

# Prepare prompt
question = "What is the square root of 144?"
ground_truth = "12"
prompt = prepare_prompt(question, deep_llm.tokenizer, "deepseek")

# Run offline mode with multiple voting
result = deep_llm.deepthink(
    prompt=prompt,
    mode="offline",
    budget=64,
    compute_multiple_voting=True
)

# Evaluate results
for method, method_result in result.voting_results.items():
    if method_result and method_result.get('answer'):
        is_correct = equal_func(method_result['answer'], ground_truth)
        print(f"{method}: {method_result['answer']} ({'✓' if is_correct else '✗'})")
```

## Architecture

### Core Design Philosophy
The `deepthink` method works like `generate()` - it takes a prepared prompt and returns answers using different voting methods, without knowing about ground truth or evaluation. This separates concerns clearly:

- **Inference**: `DeepThinkLLM.deepthink()` handles generation and voting
- **Preprocessing**: External prompt preparation using utility functions
- **Evaluation**: External comparison of results against ground truth

### Processing Modes

#### Online Mode (Confidence-Based)
Uses warmup traces to establish confidence thresholds, then applies early stopping:

```python
# Prepare prompt
prompt = prepare_prompt(question, tokenizer, "deepseek")

# Run online mode
result = deep_llm.deepthink(
    prompt=prompt,
    mode="online",
    warmup_traces=16,
    total_budget=256
)
```

#### Offline Mode (Batch Generation)
Generates all traces at once, then analyzes with multiple voting methods:

```python
# Prepare prompt
prompt = prepare_prompt_gpt(question, tokenizer, "high")

# Run offline mode
result = deep_llm.deepthink(
    prompt=prompt,
    mode="offline",
    budget=512,
    compute_multiple_voting=True
)
```

## Voting Methods

The framework supports 7 different voting strategies:

1. **Simple Majority Vote** - Basic majority voting
2. **Mean Confidence Weighted** - Weighted by average token confidence
3. **Tail Confidence Weighted** - Weighted by confidence of last 2048 tokens
4. **Bottom Window Weighted** - Weighted by bottom 10% of sliding window means
5. **Min Window Weighted** - Weighted by minimum sliding window confidence
6. **Top 10% Tail Filtered** - Only top 10% by tail confidence, then weighted
7. **Top 10% Bottom Window Filtered** - Only top 10% by bottom window confidence, then weighted

## Example Scripts

### Single Question Processing

#### Offline Mode
```bash
python examples/example_offline.py --qid 0 --dataset dataset.jsonl --budget 512
```

#### Online Mode  
```bash
python examples/example_online.py --qid 0 --dataset brumo_2025.jsonl --total_budget 256
```


## Prompt Preparation

The framework provides utility functions for different model types:

```python
from utils import prepare_prompt, prepare_prompt_gpt

# For DeepSeek models
prompt = prepare_prompt(question, tokenizer, "deepseek")

# For GPT models with reasoning effort
prompt = prepare_prompt_gpt(question, tokenizer, "high")
```

## Evaluation and Analysis

Both example scripts provide comprehensive evaluation against ground truth:

### Offline Mode Evaluation
- Individual trace accuracy
- Voting method comparison  
- Best performing method identification
- Performance statistics

### Online Mode Evaluation  
- Confidence threshold analysis
- Warmup vs final trace performance
- Early stopping effectiveness
- Voting method comparison with confidence metrics

### Analysis Tools

Analyze voting performance across experiments:

```bash
python analysis_voting.py --outputs_dir outputs --generate_report
```

## Configuration

### Model Types
- `deepseek`: DeepSeek models with Chinese system prompt
- `gpt`: GPT-style models with reasoning effort support

### Sampling Parameters
```python
from vllm import SamplingParams

custom_params = SamplingParams(
    temperature=1.0,
    top_p=0.95,
    top_k=40,
    max_tokens=64000
)

result = deep_llm.deepthink(
    prompt=prompt,
    sampling_params=custom_params
)
```

## Output Format

Results include comprehensive voting analysis:

```python
result = {
    'final_answer': '12',
    'voting_results': {
        'majority': {'answer': '12', 'confidence': None, 'num_votes': 64},
        'tail_confidence_weighted': {'answer': '12', 'confidence': 15.234, 'num_votes': 64},
        # ... other voting methods
    },
    'all_traces': [...],  # Individual traces with confidence values
    'timing_stats': {...},
    'token_stats': {...},
    'config': {...}
}
```

## Performance Optimization

### Memory Usage
- Only confidence values stored, not full logprobs
- Large data structures cleaned up immediately after processing
- Configurable batch sizes for large-scale experiments

### Throughput
- vLLM backend for efficient parallel generation
- Prefix caching enabled by default
- Optimized confidence computation using sliding windows

## Examples

### Single Question Processing with Evaluation
```python
from wrapper import DeepThinkLLM
from utils import prepare_prompt, equal_func

# Load question data
question = "What is 2^10?"
ground_truth = "1024"

# Initialize and prepare
deep_llm = DeepThinkLLM(model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
prompt = prepare_prompt(question, deep_llm.tokenizer, "deepseek")

# Run inference
result = deep_llm.deepthink(prompt=prompt, mode="offline", budget=32)

# Evaluate results
for method, method_result in result.voting_results.items():
    if method_result:
        is_correct = equal_func(method_result['answer'], ground_truth)
        print(f"{method}: {is_correct}")
```

### Confidence Analysis
```python
# Online mode with confidence analysis
result = deep_llm.deepthink(
    prompt=prompt,
    mode="online", 
    warmup_traces=16,
    total_budget=64
)

print(f"Confidence threshold: {result.conf_bar:.3f}")
print(f"Warmup traces: {len(result.warmup_traces)}")
print(f"Final traces: {len(result.final_traces)}")

# Analyze traces by confidence
above_threshold = [t for t in result.warmup_traces if t['min_conf'] >= result.conf_bar]
print(f"Traces above threshold: {len(above_threshold)}")
```

## Requirements

- Python 3.8+
- vLLM >= 0.2.0
- transformers >= 4.30.0
- numpy
- pandas
- tqdm
- dynasor (for math evaluation)

## License

MIT License - see LICENSE file for details.
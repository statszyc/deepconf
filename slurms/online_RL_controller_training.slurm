#!/bin/bash
#SBATCH --job-name=online_rl_findpara_1
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%A.log

# ---- 1. 加载必要模块 ----
module load CUDA/12.8.0

# ---- 2. 激活虚拟环境 ----
source ~/PY_envs/deepconf_py312/bin/activate

# ---- 3. 进入项目目录 ----
cd ~/Projects/Method/deepconf

nvidia-smi

# ---- 4. 运行程序，使用数组索引当作 qid ----
date
python examples/online_RL/train_rl_controller.py \
    --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --dataset aime_2025.jsonl \
    --epochs 3 \
    --lr 1e-4 \
    --max_tokens 32000 \
    --decision_stride 128 \
    --K 5 \
    --temperature 0.6 \
    --top_p 0.95 \
    --window_size 2048 \
    --alpha_token 4.0 \
    --confidence_scale 4.0 \
    --confidence_mid 0.45 \
    --confidence_k 8.0 \
    --confidence_s 4.0 \
    --trend_scale 4.0 \
    --token_cost_weight 0.002 \
    --correctness_weight 30.0 \
    --wrong_penalty 6.0 \
    --early_stop_penalty 10.0 \
    --confidence_weight 1.0 \
    --save_every 100 \
    --print_every 5
date
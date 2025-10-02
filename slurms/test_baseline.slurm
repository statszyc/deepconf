#!/bin/bash
#SBATCH --job-name=online_rl_more_conservative
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

python examples/online_RL/test_baseline_without_RL.py \
    --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --dataset aime_2025.jsonl \


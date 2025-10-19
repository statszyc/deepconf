#!/bin/bash
#SBATCH --job-name=warmup_creation
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=15:00:00
#SBATCH --output=logs/%A.log
#SBATCH --array=10

# ---- 1. 加载必要模块 ----
module load CUDA/12.8.0

# ---- 2. 激活虚拟环境 ----
source ~/PY_envs/deepconf_py312/bin/activate

# ---- 3. 进入项目目录 ----
cd ~/Projects/Method/deepconf

nvidia-smi

# ---- 4. 运行程序，使用数组索引当作 qid ----
date
python examples/online_RL/data_creation.py \
  --dataset aime_2025.jsonl \
  --qid ${SLURM_ARRAY_TASK_ID} \
  --budget 16 \
  --output_file trace_data_3/warmup_data/aime_2025_${SLURM_ARRAY_TASK_ID}_warmup.jsonl
date
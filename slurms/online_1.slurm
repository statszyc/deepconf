#!/bin/bash
#SBATCH --job-name=online_1
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/qid_%A_%a.log

# ---- 1. 加载必要模块 ----
module load CUDA/12.8.0

# ---- 2. 激活虚拟环境 ----
source ~/PY_envs/deepconf_py312/bin/activate

# ---- 3. 进入项目目录 ----
cd ~/deepconf

nvidia-smi

# ---- 4. 运行程序，使用数组索引当作 qid ----
python examples/example_online.py \
  --qid ${SLURM_ARRAY_TASK_ID} \
  --rid 2 \
  --dataset aime_2025.jsonl \
  --total_budget 256 \
  --output_dir online-dpsk


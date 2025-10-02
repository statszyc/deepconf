#!/bin/bash
#SBATCH --job-name=online_1
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/%A_%a.log

# ---- 1. 加载必要模块 ----
module load CUDA/12.8.0

# ---- 2. 激活虚拟环境 ----
source ~/PY_envs/deepconf_py312/bin/activate

# ---- 3. 进入项目目录 ----
cd ~/Projects/Method/deepconf

nvidia-smi

# ---- 4. 运行程序，使用数组索引当作 qid ----
python examples/online_RL/example_online_rl.py \
  --qid ${SLURM_ARRAY_TASK_ID} \
  --rid 2 \
  --dataset aime_2025.jsonl \
  --total_budget 256 \
  --output_dir online_rl-dpsk \
  --tensor_parallel_size 1 \
  --rl_episodes 100 \
  --reward_correctness_weight 1.0 \
  --reward_confidence_weight 0.1 \
  --reward_token_cost_weight 0.001

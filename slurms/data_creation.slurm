#!/bin/bash
#SBATCH --job-name=data_creation_array  # 修改作业名称，更具描述性
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=9:00:00
#SBATCH --output=logs/%A_%a.log  # 修改日志文件名，为每个子作业单独记录

# ---- 新增：定义作业数组 ----
# 格式为 --array=<起始索引>-<结束索引>%<最大并行数>
# 这里表示创建从 0 到 14 的作业，但最多只允许 3 个同时运行
#SBATCH --array=0-14%3

# ---- 1. 加载必要模块 ----
module load CUDA/12.8.0

# ---- 2. 激活虚拟环境 ----
source ~/PY_envs/deepconf_py312/bin/activate

# ---- 3. 进入项目目录 ----
cd ~/Projects/Method/deepconf

nvidia-smi
echo "Starting SLURM array job ${SLURM_ARRAY_JOB_ID} with task ID ${SLURM_ARRAY_TASK_ID}"

# ---- 4. 运行程序，使用 SLURM 提供的环境变量作为 qid ----
date
python examples/online_RL/data_creation.py \
  --dataset aime_2025.jsonl \
  --qid ${SLURM_ARRAY_TASK_ID}  # 关键修改：使用环境变量替换硬编码的 0
date

echo "Finished SLURM task ID ${SLURM_ARRAY_TASK_ID}"
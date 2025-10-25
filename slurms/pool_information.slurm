#!/bin/bash
#SBATCH --job-name=pool_v22
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=logs/%A_%a.log
#SBATCH --array=20-29

# ---- 1. 加载必要模块 ----
module load CUDA/12.8.0

# ---- 2. 激活虚拟环境 ----
source ~/PY_envs/deepconf_py312/bin/activate

# ---- 3. 进入项目目录 ----
cd ~/Projects/Method/deepconf

nvidia-smi

# ---- 4. 运行程序，使用数组索引当作 qid ----
QID=${SLURM_ARRAY_TASK_ID}
OUT_DIR="trace_data/pool_information_v3"
OUT_PATH="${OUT_DIR}/aime_2025_${QID}_deepconflow_self_check.jsonl"

if [ -f "$OUT_PATH" ]; then
    echo "✅ Skipping question ${QID}: result already exists at ${OUT_PATH}"
    exit 0
fi

# 运行主脚本
echo "🚀 Running follow-up self-check for question ${QID}"
python examples/online_RL/question_exploration/pool_information/pool_information.py \
    --question_id "$QID" \
    --output_dir "$OUT_DIR" 


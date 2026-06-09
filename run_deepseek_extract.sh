#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --export=NONE
#SBATCH --time=4:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=deepseek_r1_extract
#SBATCH --output=logs/deepseek_extract-%j.out
#SBATCH --error=logs/deepseek_extract-%j.err

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Run DeepSeek-R1-Distill-Qwen-7B on all four ExtractBench domains.
#
# What this does:
#   - Loads DeepSeek-R1-7B in bfloat16 on an A100 GPU
#   - Runs structured JSON extraction on all 4 domains (~25 documents)
#   - Saves per-field activations at 14 layers to .npz files
#   - Saves token log-probs and extraction JSON to artifacts/
#
# Prerequisites:
#   1. python scripts/00_download_model.py  (run ONCE on login node first)
#   2. git pull (ensure latest code is on the cluster)
#
# Submit from the LOGIN NODE:
#   mkdir -p logs
#   sbatch run_deepseek_extract.sh
#
# After this job completes, run:
#   sbatch run_deepseek_analysis.sh
# ─────────────────────────────────────────────────────────────────────────────

# A100 nodes use AMD EPYC CPUs — must load AMD module stack explicitly.
#source /etc/profile
#module unuse /software/easybuild-INTEL_A40/modules/all 2>/dev/null || true
#module use   /software/easybuild-AMD_A100/modules/all

module load Python/3.12.3
module load CUDA/12.4.0

source ~/nlp_lab/bin/activate
export PYTHONPATH=$HOME/NLP_Lab/src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail
cd ~/NLP_Lab

echo "=== ENVIRONMENT ==="
grep "model name" /proc/cpuinfo | head -1 || true
echo "Python: $(python --version 2>&1)"
echo "Which python: $(which python)"
nvidia-smi || true
echo "=== ENVIRONMENT OK ==="

echo ""
echo "=== STAGE 01: DeepSeek-R1-Distill-Qwen-7B extraction (all 4 domains) ==="
echo "Config: configs/exp_deepseek_r1_7b_pooled.yaml"
echo "All domains extracted in one pass → artifacts/deepseek_r1_7b_pooled/"
echo ""

python scripts/01_extract.py \
    --config configs/exp_deepseek_r1_7b_pooled.yaml

echo ""
echo "=== STAGE 01 COMPLETE ==="
echo "Activations saved to: artifacts/deepseek_r1_7b_pooled/activations/"
echo "Extractions saved to: artifacts/deepseek_r1_7b_pooled/extractions/"
echo ""
echo "Next step: sbatch run_deepseek_analysis.sh"
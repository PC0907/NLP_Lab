#!/bin/bash
#SBATCH --partition=A40medium
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=deepseek_r1_extract
#SBATCH --output=logs/deepseek_extract-%j.out
#SBATCH --error=logs/deepseek_extract-%j.err

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Run DeepSeek-R1-Distill-Qwen-7B on all four ExtractBench domains.
# A40 VERSION — runs on the Intel A40 nodes with the standard ~/nlp_lab venv.
#
# Why A40 (not A100):
#   DeepSeek-R1-Distill-Qwen-7B in bf16 is ~14-16 GB and fits the 48 GB A40
#   trivially. The A100 nodes are AMD EPYC and need a SEPARATELY-COMPILED venv
#   (~/nlp_lab_a100 via install_a100_env.sh); the ~/nlp_lab venv is Intel-built
#   and Illegal-instruction-crashes on A100. Staying on A40 avoids all of that.
#
# Known caveat (non-fatal):
#   The 3 largest credit-agreement PDFs (~460-516k chars) may CUDA-OOM on the
#   A40. 01_extract.py catches per-document errors and continues, so the run
#   still completes with the remaining documents (records the OOM as an error).
#
# Prerequisites:
#   1. python scripts/00_download_model.py   (run ONCE on the login node)
#   2. git pull                              (latest code on the cluster)
#   3. mkdir -p logs
#
# Submit from the LOGIN NODE:
#   sbatch run_deepseek_extract.sh
#
# After this completes:
#   sbatch run_deepseek_analysis.sh
#   sbatch run_clap.sh
# ─────────────────────────────────────────────────────────────────────────────

module load Python/3.12.3
module load CUDA/12.4.0
source ~/nlp_lab/bin/activate
export PYTHONPATH=$HOME/NLP_Lab/src:$PYTHONPATH

# Reduce CUDA fragmentation during long reasoning-model generations.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail
cd ~/NLP_Lab

echo "=== ENVIRONMENT ==="
grep "model name" /proc/cpuinfo | head -1 || true
echo "Python:        $(python --version 2>&1)"
echo "Which python:  $(which python)"
python -c "import torch; print('torch', torch.__version__, '| cuda avail', torch.cuda.is_available())"
nvidia-smi || true
echo "=== ENVIRONMENT OK ==="

echo ""
echo "=== STAGE 01: DeepSeek-R1-Distill-Qwen-7B extraction (all 4 domains) ==="
echo "Config: configs/exp_deepseek_r1_7b_pooled.yaml"
echo "All domains extracted in one pass -> artifacts/deepseek_r1_7b_pooled/"
echo ""

python scripts/01_extract.py \
    --config configs/exp_deepseek_r1_7b_pooled.yaml

echo ""
echo "=== STAGE 01 COMPLETE ==="
echo "Activations -> artifacts/deepseek_r1_7b_pooled/activations/"
echo "Extractions -> artifacts/deepseek_r1_7b_pooled/extractions/"
echo ""
echo "Extraction summary:"
python -m json.tool artifacts/deepseek_r1_7b_pooled/extractions/_summary.json 2>/dev/null | head -40 || true
echo ""
echo "Next step: sbatch run_deepseek_analysis.sh"
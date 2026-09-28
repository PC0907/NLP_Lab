#!/bin/bash
#SBATCH --partition=A100medium
#SBATCH --export=NONE
#SBATCH --time=8:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=sob_deepseek_extract_a100
#SBATCH --output=logs/sob_extract_a100-%j.out
#SBATCH --error=logs/sob_extract_a100-%j.err

# Stage 1: DeepSeek-R1 extraction on SOB (text/multi-hop), on an A100 node.
# A100 nodes are AMD-CPU: --export=NONE + the AMD-built venv (~/nlp_lab_a100).
# LOGIN-NODE prereqs (once): build ~/nlp_lab_a100, download model + SOB.

module load Python/3.12.3
module load CUDA/12.4.0
source ~/nlp_lab_a100/bin/activate
export PYTHONPATH=$HOME/NLP_Lab/src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_DATASETS_OFFLINE=1

set -euo pipefail
cd ~/NLP_Lab

echo "=== ENVIRONMENT ==="
hostname; grep "model name" /proc/cpuinfo | head -1 || true
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi || true

# Real GPU self-test — abort before any silent CPU fallback.
python - <<'PY' || { echo "FATAL: GPU not usable; resubmit with --exclude=$(hostname -s)."; exit 1; }
import torch, sys
try:
    assert torch.cuda.is_available() and torch.cuda.device_count() > 0
    x = torch.zeros(2048, 2048, device="cuda:0"); _ = (x + 1.0).sum().item()
    free, total = torch.cuda.mem_get_info(0)
    print(f"GPU OK | {torch.cuda.get_device_name(0)} | free {free/1e9:.1f}/{total/1e9:.1f} GB")
except Exception as e:
    print("GPU SELF-TEST FAILED:", repr(e)); sys.exit(1)
PY
echo "=== GPU GUARD PASSED ==="

echo "=== STAGE 01: DeepSeek-R1 extraction on SOB (A100) ==="
python scripts/01_extract.py --config configs/exp_deepseek_r1_7b_sob.yaml

echo "=== STAGE 01 COMPLETE ==="
python -m json.tool artifacts/deepseek_r1_7b_sob/extractions/_summary.json 2>/dev/null | head -30 || true
echo "Next: sbatch run_sob_analysis_a100.sh"

#!/bin/bash
#SBATCH --partition=A40medium
#SBATCH --time=8:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=sob_deepseek_extract
#SBATCH --output=logs/sob_extract-%j.out
#SBATCH --error=logs/sob_extract-%j.err

# Stage 1: DeepSeek-R1-Distill-Qwen-7B extraction on the SOB text/multi-hop set.
# Captures the reasoning trace automatically. SOB contexts are short (~919
# tokens) so this is much faster than ExtractBench; 300 records ~2-3h.
#
# LOGIN-NODE PREREQS (run once, they need internet):
#   python scripts/00_download_model.py            # DeepSeek weights
#   python scripts/00_download_sob.py --out data/sob   # SOB dataset -> data/sob

module load Python/3.12.3
module load CUDA/12.4.0
source ~/nlp_lab/bin/activate
export PYTHONPATH=$HOME/NLP_Lab/src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_DATASETS_OFFLINE=1   # read the cached SOB from disk, no internet

set -euo pipefail
cd ~/NLP_Lab

echo "=== ENVIRONMENT ==="
hostname; grep "model name" /proc/cpuinfo | head -1 || true
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi || true

# ── Real GPU self-test (allocate + compute) — aborts before any CPU fallback ──
python - <<'PY' || { echo "FATAL: GPU not usable on this node. Resubmit with --exclude=$(hostname -s)."; exit 1; }
import torch, sys
try:
    assert torch.cuda.is_available() and torch.cuda.device_count() > 0
    x = torch.zeros(2048, 2048, device="cuda:0")
    _ = (x + 1.0).sum().item()
    free, total = torch.cuda.mem_get_info(0)
    print(f"GPU OK | {torch.cuda.get_device_name(0)} | free {free/1e9:.1f}/{total/1e9:.1f} GB")
except Exception as e:
    print("GPU SELF-TEST FAILED:", repr(e)); sys.exit(1)
PY
echo "=== GPU GUARD PASSED ==="

echo "=== STAGE 01: DeepSeek-R1 extraction on SOB ==="
python scripts/01_extract.py --config configs/exp_deepseek_r1_7b_sob.yaml

echo "=== STAGE 01 COMPLETE ==="
python -m json.tool artifacts/deepseek_r1_7b_sob/extractions/_summary.json 2>/dev/null | head -30 || true
echo "Next: sbatch run_sob_analysis.sh"

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

# Stage 1: DeepSeek-R1-Distill-Qwen-7B extraction on all 4 ExtractBench domains.
# A40 version with a REAL GPU self-test guard: torch.cuda.is_available() returns
# True even on an ECC-faulted card (e.g. node-02), so we actually allocate +
# compute on the GPU. If that fails, abort instead of silently using the CPU.

module load Python/3.12.3
module load CUDA/12.4.0
source ~/nlp_lab/bin/activate
export PYTHONPATH=$HOME/NLP_Lab/src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail
cd ~/NLP_Lab

echo "=== ENVIRONMENT ==="
hostname
grep "model name" /proc/cpuinfo | head -1 || true
echo "Which python: $(which python)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi || true
echo "=== ENVIRONMENT OK ==="

# ── Real GPU self-test (allocate + compute + mem_get_info) ───────────────────
python - <<'PY' || { echo "FATAL: GPU not usable on this node (likely ECC-faulted). Aborting before CPU fallback. Resubmit, e.g. sbatch --exclude=$(hostname -s) ..."; exit 1; }
import torch, sys
try:
    assert torch.cuda.is_available() and torch.cuda.device_count() > 0, "no CUDA device"
    x = torch.zeros(2048, 2048, device="cuda:0")        # allocation fails on faulted GPU
    val = (x + 1.0).sum().item()                        # real compute on the GPU
    free, total = torch.cuda.mem_get_info(0)            # exactly what accelerate probes
    print(f"GPU SELF-TEST OK | {torch.cuda.get_device_name(0)} | "
          f"free {free/1e9:.1f} / {total/1e9:.1f} GB | compute={val:.0f}")
    ecc = torch.cuda.cuda_version  # touch driver; informational only
    sys.exit(0)
except Exception as e:
    print("GPU SELF-TEST FAILED:", repr(e))
    sys.exit(1)
PY
echo "=== GPU GUARD PASSED ==="

echo ""
echo "=== STAGE 01: DeepSeek-R1-Distill-Qwen-7B extraction (all 4 domains) ==="
python scripts/01_extract.py --config configs/exp_deepseek_r1_7b_pooled.yaml

echo ""
echo "=== STAGE 01 COMPLETE ==="
echo "Extraction summary:"
python -m json.tool artifacts/deepseek_r1_7b_pooled/extractions/_summary.json 2>/dev/null | head -40 || true
echo ""
echo "Next: sbatch run_deepseek_analysis.sh"
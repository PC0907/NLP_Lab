#!/bin/bash
#SBATCH --partition=A100medium
#SBATCH --export=NONE
#SBATCH --time=8:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=sob_attr_extract
#SBATCH --output=logs/sob_attr_extract-%j.out
#SBATCH --error=logs/sob_attr_extract-%j.err

# Re-extraction of SOB with DeepSeek-R1 that ALSO saves per-token reasoning
# states (4 layers) + token-string sidecars, for field-localized attribution.
# Separate experiment name (deepseek_r1_7b_sob_attr) preserves the first run.
# Greedy decoding => parsed JSON + labels reproduce the first run exactly.

module load Python/3.12.3
module load CUDA/12.4.0
source ~/nlp_lab_a100/bin/activate
export PYTHONPATH=$HOME/NLP_Lab/src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_DATASETS_OFFLINE=1

# Turn on per-token reasoning capture for the 4 analysis layers.
export REASONING_TOKEN_LAYERS="16,19,23,26"
export REASONING_TOKEN_CAP="2048"

set -euo pipefail
cd ~/NLP_Lab

echo "=== ENVIRONMENT ==="
hostname; grep "model name" /proc/cpuinfo | head -1 || true
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

echo "=== STAGE 01: DeepSeek-R1 extraction on SOB WITH reasoning-token capture ==="
python scripts/01_extract.py --config configs/exp_deepseek_r1_7b_sob_attr.yaml

echo "=== STAGE 01 COMPLETE ==="
ls -lh artifacts/deepseek_r1_7b_sob_attr/activations/*.rtokens.json 2>/dev/null | head -3 || true
echo "Next: sbatch run_sob_attr_analysis_a100.sh"

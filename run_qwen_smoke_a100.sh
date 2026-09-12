#!/bin/bash
#SBATCH --partition=A100medium
#SBATCH --export=NONE
#SBATCH --time=1:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=qwen_smoke
#SBATCH --output=logs/qwen_smoke-%j.out
#SBATCH --error=logs/qwen_smoke-%j.err

# GATE before any large Qwen extraction.
#
# The pipeline detects a reasoning trace by looking for the literal "</think>"
# in the generated token stream. Hybrid-thinking models (the Qwen3.x family)
# emit that block only when the chat template is asked for it, and the pipeline
# previously hardcoded enable_thinking=False. Getting this wrong does not crash
# anything -- extraction succeeds, and Stage 7 later finds no trace to attribute.
# Which is exactly why it is worth 15 minutes to check first.
#
# Also reports the model's layer count and hidden size, because activations.layers
# is architecture-specific and must not be copied across model families.
#
# Usage (arguments, NOT environment variables -- this job runs with
# --export=NONE, so an exported MODEL= would be stripped before the script
# starts and the config's placeholder would be used instead):
#
#   sbatch run_qwen_smoke_a100.sh Qwen/Qwen3.5-9B
#   sbatch run_qwen_smoke_a100.sh Qwen/Qwen3.5-9B off     # force thinking off
#
# sbatch passes trailing arguments straight through to the script.

module load Python/3.12.3
module load CUDA/12.4.0
source ~/nlp_lab_a100/bin/activate
export PYTHONPATH=$HOME/NLP_Lab/src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_DATASETS_OFFLINE=1

set -euo pipefail
cd ~/NLP_Lab

CFG="configs/exp_qwen35_reasoning_sob.yaml"
MODEL_ARG="${1:-}"
THINKING_ARG="${2:-}"

if [ -z "$MODEL_ARG" ]; then
  echo "FATAL: no model given."
  echo "  usage: sbatch run_qwen_smoke_a100.sh <hf-model-id> [on|off]"
  echo "  e.g.   sbatch run_qwen_smoke_a100.sh Qwen/Qwen3.5-9B"
  echo ""
  echo "  (An exported MODEL= will NOT work: this job uses --export=NONE, so"
  echo "   the environment is not inherited.)"
  exit 2
fi

ARGS="--model-override ${MODEL_ARG}"
if [ -n "$THINKING_ARG" ]; then ARGS="$ARGS --thinking-override ${THINKING_ARG}"; fi
echo "model: ${MODEL_ARG}${THINKING_ARG:+  (thinking forced ${THINKING_ARG})}"

echo "=== ENVIRONMENT ==="
hostname; nvidia-smi || true

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

echo ""
echo "=== SMOKE TEST ==="
set +e
python scripts/00_smoke_reasoning_model.py --config "$CFG" --n 3 ${ARGS}
STATUS=$?
set -e

echo ""
if [ $STATUS -eq 0 ]; then
  echo "USABLE. Before the full run, copy the suggested activations.layers into"
  echo "  ${CFG} and set REASONING_TOKEN_LAYERS in the extract script to match."
else
  echo "NOT USABLE as configured -- read the FAIL lines above. Do not start the"
  echo "full extraction until this exits 0."
fi
exit $STATUS

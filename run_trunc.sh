#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --time=8:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --export=NONE
#SBATCH --output=logs/trunc_%x_%j.out
#SBATCH --error=logs/trunc_%x_%j.err
#
# Extraction + labeling at max_input_chars=300000, pymupdf, all four domains.
#   sbatch -J trunc_qwen  run_trunc.sh qwen35_4b
#   sbatch -J trunc_gemma run_trunc.sh gemma3_4b
# Text comes from the parsed cache; truncation is applied at prompt build.

unset SLURM_EXPORT_ENV
source ~/NLP_Lab/setup_env_a100.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
set -euo pipefail
cd ~/NLP_Lab

MODEL="$1"
CFG="configs/exp_trunc_${MODEL}_pymupdf.yaml"
echo "=== $MODEL @ 300k chars ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/01_extract.py --config "$CFG"
python scripts/02_label.py   --config "$CFG"

python - "$MODEL" << 'PYEOF'
import json, glob, sys
from collections import defaultdict
m = sys.argv[1]
d = defaultdict(lambda: [0, 0, 0])
for f in glob.glob(f"artifacts/trunc_{m}_pymupdf/labels/*.json"):
    if f.endswith("_summary.json"): continue
    x = json.load(open(f)); L = x.get("labels", [])
    k = x.get("domain", "?")
    d[k][0] += 1; d[k][1] += len(L); d[k][2] += sum(int(i.get("is_error", 0)) for i in L)
print("=== per-domain ===")
for k, (c, n, e) in sorted(d.items()):
    print(f"  {k:32} {c:2} docs {n:5} fields {e/max(n,1):6.1%}")
PYEOF

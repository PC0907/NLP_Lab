#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --time=8:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --export=NONE
#SBATCH --output=logs/grid_%x_%j.out
#SBATCH --error=logs/grid_%x_%j.err
#
# One cell of the (model x parser) grid: extract + label on the given domains.
#
#   sbatch -J <name> run_grid_cell.sh <model> <parser> <domain> [<domain> ...]
#   e.g. sbatch -J q_pym  run_grid_cell.sh qwen35_4b pymupdf academic/research sport/swimming
#
# Text comes from data/parsed/<parser>/ (the verified-deterministic cache), so
# both models read byte-identical input for every document. Stops after
# labeling; probes and nested LODO are run separately on the intersection.

unset SLURM_EXPORT_ENV
source ~/NLP_Lab/setup_env_a100.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
set -euo pipefail
cd ~/NLP_Lab

MODEL="$1"; PARSER="$2"; shift 2
DOMAINS=("$@")
CFG="configs/exp_grid_${MODEL}_${PARSER}.yaml"
[ -f "$CFG" ] || { echo "no config $CFG"; exit 1; }

# Write a per-job config with the requested domains substituted in, so the
# artifacts land in one directory per (model, parser) regardless of which
# domain subset a given job covers.
TMPCFG="configs/.grid_${MODEL}_${PARSER}_$$.yaml"
python - "$CFG" "$TMPCFG" "${DOMAINS[@]}" << 'PYEOF'
import sys, re
src, dst, *domains = sys.argv[1:]
txt = open(src).read()
block = "  domains:\n" + "".join(f'    - "{d}"\n' for d in domains)
txt = re.sub(r"  domains:\n(?:    - .*\n)+", block, txt)
open(dst, "w").write(txt)
PYEOF

echo "=== model=$MODEL parser=$PARSER domains=${DOMAINS[*]} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/01_extract.py --config "$TMPCFG"
python scripts/02_label.py   --config "$TMPCFG"
rm -f "$TMPCFG"

echo "=== per-domain ==="
python - "$MODEL" "$PARSER" << 'PYEOF'
import json, glob, sys
from collections import defaultdict
m, p = sys.argv[1:3]
d = defaultdict(lambda: [0, 0, 0])
for f in glob.glob(f"artifacts/grid_{m}_{p}/labels/*.json"):
    if f.endswith("_summary.json"): continue
    x = json.load(open(f)); L = x.get("labels", [])
    k = x.get("domain", "?")
    d[k][0] += 1; d[k][1] += len(L); d[k][2] += sum(int(i.get("is_error", 0)) for i in L)
for k, (c, n, e) in sorted(d.items()):
    print(f"  {k:32} {c:2} docs {n:5} fields {e/max(n,1):6.1%}")
PYEOF
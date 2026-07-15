#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --time=1:30:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --export=NONE
#SBATCH --job-name=gemma_test5
#SBATCH --output=logs/gemma_test5_%j.out
#SBATCH --error=logs/gemma_test5_%j.err
# Gemma 5-doc confirmation: does max_new_tokens=8192 fix the OOM/runtime, and does
# Gemma produce usable extractions with a sane error rate? Confirm BEFORE the 8h run.
unset SLURM_EXPORT_ENV
source ~/NLP_Lab/setup_env_a100.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
set -euo pipefail
cd ~/NLP_Lab
echo "=== gpu ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== extract 5 docs ==="
python scripts/01_extract.py --config configs/exp_gemma3_12b_test5.yaml

echo "=== label ==="
python scripts/02_label.py --config configs/exp_gemma3_12b_test5.yaml

echo "=== quick check: per-doc success, finish_reason, error rate ==="
python -c "
import json, glob
exts = sorted(glob.glob('artifacts/gemma3_12b_test5/extractions/*.json'))
print(f'extracted docs: {len(exts)}')
for f in exts:
    d = json.load(open(f))
    fr = d.get('finish_reason','?')
    parsed = d.get('parsed_json') is not None
    print(f\"  {f.split('/')[-1][:40]:42} finish={fr:8} parsed={parsed}\")
labs = [f for f in glob.glob('artifacts/gemma3_12b_test5/labels/*.json') if not f.endswith('_summary.json')]
tot=err=0
for f in labs:
    L=json.load(open(f)).get('labels',[])
    tot+=len(L); err+=sum(int(x.get('is_error',0)) for x in L)
print(f'fields={tot} errors={err} rate={err/max(tot,1):.1%}')
"
echo "=== done ==="
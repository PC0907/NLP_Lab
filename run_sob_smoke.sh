#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=2:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=sob_smoke
#SBATCH --output=logs/sob_smoke_%j.out
#SBATCH --error=logs/sob_smoke_%j.err
# SOB 50-doc smoke run: extract -> label -> error-rate check. GPU needed.
# The config caps at max_documents:50. Goal: does Qwen produce parseable JSON on
# QA-schema tasks, do keys align, and what is the error rate (trainable target?).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

source ~/NLP_Lab/setup_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail
cd ~/NLP_Lab
echo "=== node/gpu ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== [1/3] extract (50 docs) ==="
python scripts/01_extract.py --config configs/exp_qwen35_4b_sob.yaml

echo "=== [2/3] label ==="
python scripts/02_label.py --config configs/exp_qwen35_4b_sob.yaml

echo "=== [3/3] error-rate + parse check ==="
python -c "
import json, glob
fs = [f for f in glob.glob('artifacts/qwen35_4b_sob_text/labels/*.json') if not f.endswith('_summary.json')]
tot = err = 0
for f in fs:
    d = json.load(open(f)); L = d.get('labels', [])
    tot += len(L); err += sum(int(x.get('is_error',0)) for x in L)
print(f'docs={len(fs)} fields={tot} errors={err} rate={err/max(tot,1):.1%}')
print('-> <8%: error-starved (weak probe). 15-40%: good trainable signal. Card says ~17% expected.')
"
echo "=== done ==="
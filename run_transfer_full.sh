#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=transfer_full
#SBATCH --output=logs/transfer_full_%j.out
#SBATCH --error=logs/transfer_full_%j.err
# CPU-only: transfer BOTH directions across SEVERAL layers. No GPU.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
source ~/NLP_Lab/setup_env.sh
set -euo pipefail
cd ~/NLP_Lab

EB=configs/exp_qwen35_4b_pooled_alltokens.yaml
INS=configs/exp_qwen35_4b_insurance.yaml

# layers present in BOTH configs' captured sets (EB: 1,4,8,12,14,16,18,20,22,24,26,28,30,32;
# INS: same). Sweep a mid-network band around the selected ~16.
for L in 8 12 14 16 18 20 24; do
  echo ""
  echo "########## LAYER $L ##########"
  echo "----- FORWARD: ExtractBench -> insurance (layer $L) -----"
  python transfer_probe.py \
      --train-config $EB --test-config $INS \
      --layer $L --train-exclude-domains finance/10kq \
      --out transfer_eb2ins_L${L}.json

  echo "----- REVERSE: insurance -> ExtractBench (layer $L) -----"
  python transfer_probe.py \
      --train-config $INS --test-config $EB \
      --layer $L --test-exclude-domains finance/10kq \
      --out transfer_ins2eb_L${L}.json
done

echo ""
echo "########## SUMMARY ##########"
python -c "
import json, glob, os
def grab(pat):
    rows=[]
    for f in sorted(glob.glob(pat)):
        d=json.load(open(f))
        rows.append((d['layer'], d['transfer_raw_auroc'], d['transfer_l2_auroc']))
    return sorted(rows)
print('FORWARD (EB->insurance):   layer  raw_AUROC  L2_AUROC')
for L,r,l2 in grab('artifacts/qwen35_4b_insurance/results/transfer_eb2ins_L*.json'):
    print(f'                            {L:>3}    {r:.4f}    {l2:.4f}')
print('REVERSE (insurance->EB):   layer  raw_AUROC  L2_AUROC')
for L,r,l2 in grab('artifacts/qwen35_4b_pooled_alltokens/results/transfer_ins2eb_L*.json'):
    print(f'                            {L:>3}    {r:.4f}    {l2:.4f}')
"
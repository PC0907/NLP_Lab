#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --time=6:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --export=NONE
#SBATCH --job-name=parse_all
#SBATCH --output=logs/parse_all_%j.out
#SBATCH --error=logs/parse_all_%j.err
#
# Parse ALL FOUR DOMAINS with three extractors and cache the text to
# data/parsed/<parser>/<doc_id>.txt -- the same location the patched
# pdf_utils.py reads from, so extraction runs afterwards are parse-free.
#
# --repeats 1: determinism was already established on the swimming subset
# (3 repeats, identical hashes, for both pymupdf and docling). Repeating here
# would triple a job whose cost is dominated by docling on the long financial
# documents (30-56s each).
#
# MinerU is omitted: 3.4.5 requires a newer transformers than the pipeline
# uses (PPDocLayoutV2Config lacks reading_order_config), and upgrading
# transformers would put every existing result at risk.
#
# A GPU is requested for docling's layout and table-structure models. pymupdf
# and camelot are CPU-only.
#
# EXPECTED FAILURES, not bugs: camelot extracts TABLES ONLY, so it will return
# little or nothing on academic papers and credit agreements, which are prose.
# Those documents will be reported as errors and skipped for that parser.

unset SLURM_EXPORT_ENV
source ~/NLP_Lab/setup_env_a100.sh

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

set -euo pipefail
cd ~/NLP_Lab

python parse_corpus.py \
    --domains academic/research finance/credit_agreement finance/10kq sport/swimming \
    --parsers pymupdf docling camelot \
    --repeats 1

echo ""
echo "=== CACHE CONTENTS ==="
for p in pymupdf docling camelot; do
  n=$(ls data/parsed/$p/*.txt 2>/dev/null | wc -l)
  echo "  $p: $n documents cached"
done
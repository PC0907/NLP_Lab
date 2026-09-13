#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --time=2:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --export=NONE
#SBATCH --job-name=parse_corpus
#SBATCH --output=logs/parse_corpus_%j.out
#SBATCH --error=logs/parse_corpus_%j.err
#
# Parse the swimming documents with four extractors and write the text out.
# A GPU is requested because docling and MinerU run neural models; pymupdf and
# camelot are CPU-only and would not need it.
#
# Swimming first: the reading-order failure lives there, and five short
# documents answer the question cheaply.

unset SLURM_EXPORT_ENV
source ~/NLP_Lab/setup_env_a100.sh

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

set -euo pipefail
cd ~/NLP_Lab

python parse_corpus.py \
    --domains sport/swimming \
    --parsers pymupdf docling camelot mineru \
    --repeats 3
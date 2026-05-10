#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=4:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=qwen35_pymupdf
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# --- Setup ---
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

# Load required modules
module load Python/3.12
module load CUDA/12.1.1   # or whatever CUDA version is available

# Activate Python virtual environment
source ~/venvs/nlp_lab/bin/activate

# --- Run pipeline ---
cd ~/NLP_Lab

# Set HF token if needed (for gated models)
# HF_TOKEN is read from ~/.bashrc on the cluster (set there manually).
# Don't hardcode it here — GitHub blocks pushes containing tokens.

python scripts/01_extract.py --config configs/exp_qwen35_4b_pymupdf.yaml
python scripts/02_label.py --config configs/exp_qwen35_4b_pymupdf.yaml
python scripts/03_train_probe.py --config configs/exp_qwen35_4b_pymupdf.yaml
python scripts/04_evaluate.py --config configs/exp_qwen35_4b_pymupdf.yaml

echo "Job completed at $(date)"
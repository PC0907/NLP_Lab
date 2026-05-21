#!/bin/bash
# Setup for A100 (AMD EPYC) nodes.
# Use ONLY in jobs submitted with --export=NONE.

source /etc/profile

# Force the AMD stack explicitly — the wiki says auto-switching is usually
# fine but occasionally fails, and explicit is safer.
module unuse /software/easybuild-INTEL_A40/modules/all 2>/dev/null || true
module use   /software/easybuild-AMD_A100/modules/all

module load Python/3.12.3
module load CUDA/12.4.0

source ~/nlp_lab_a100/bin/activate
export PYTHONPATH=$HOME/NLP_Lab/src:$PYTHONPATH
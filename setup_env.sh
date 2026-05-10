#!/bin/bash
# Sources the cluster environment for NLP_Lab jobs.
# Don't run directly — source it from a job script:
#     source ~/NLP_Lab/setup_env.sh

module load Python/3.12.3
module load CUDA/12.4.0
source ~/nlp_lab/bin/activate
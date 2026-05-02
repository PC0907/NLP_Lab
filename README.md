# Probe-Based Trust Signals for Structured Information Extraction

Investigating whether linear probes on LLM internal activations can identify
risky extracted fields well enough to improve the cost-quality tradeoff of
selective regeneration.

## Research Question

Extracting structured information from documents is a common LLM application,
but errors or hallucinated fields can cause silent failures even when the
output looks plausible. This project asks:

> Can probe-based trust signals identify risky extracted fields well enough
> to improve the cost-quality tradeoff of selective regeneration, compared
> to black-box baselines (token log-probabilities, self-consistency, etc.)?

The probe itself is a means, not the end. The headline result is a Pareto
curve comparing probe-guided selective regeneration against baselines on
extraction accuracy vs. compute cost.

## Approach

1. Run an open-weights LLM on extraction benchmark documents.
2. Compare extracted fields against gold annotations to produce per-field
   correctness labels.
3. Capture LLM hidden states at the moment each field is generated.
4. Train a linear probe (logistic regression) to predict per-field errors
   from these hidden states.
5. Compare the probe against black-box baselines for error detection (AUROC).
6. Use the probe's risk scores to drive selective regeneration; measure the
   resulting cost-quality tradeoff against alternatives.

## Setup

### Requirements

- Python 3.12
- CUDA-capable GPU (tested on Kaggle P100, 16GB)
- HuggingFace account (for model download)

### Installation

```bash
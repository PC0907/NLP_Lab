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

## Start here

This branch holds the **reasoning-trace track**: DeepSeek-R1 on SOB, asking
whether a *reasoning model's* chain of thought carries error information beyond
the answer token — and what acting on that information is actually worth.

- **[`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md)** — read this first:
  dataset, method, full results, limitations, reproduction steps.
- **[`docs/DEEP_DIVE.md`](docs/DEEP_DIVE.md)** — every module, every design
  decision, and the reasoning behind them.
- [`docs/REASONING_TRACK.md`](docs/REASONING_TRACK.md) — condensed pipeline and
  result-file map.
- [`archive/README.md`](archive/README.md) — superseded work, and the old→new
  stage mapping for documents written before the pipeline was renumbered.

The project also has a partner track (Qwen3.5 on ExtractBench) that shares this
library and lives on its own branch. Its experiment scripts and configs are not
on this branch; the benchmark loaders it uses remain in `src/probe_extraction/`
because `scripts/01_extract.py` dispatches to them.

## Pipeline

Stages run in number order, `scripts/01_extract.py` through
`scripts/10_make_figures.py`, with nothing skipped.

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
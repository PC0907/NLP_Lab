# Project Progress — Days 1-2

**Project:** Probe-based trust signals for structured information extraction

**Research question:** Can probe-based trust signals identify risky extracted fields well enough to improve the cost-quality tradeoff of selective regeneration, compared to black-box baselines?

---

## TL;DR

In two working days, built the full pipeline for the first four stages of the project (extraction → labeling → probe training → baseline comparison). All stages tested and runnable. Produced a first sandbox-scale comparison: linear probes on Qwen2.5-7B residual stream activations achieve **CV-AUROC 0.75-0.79** for per-field error detection, vs. **0.69-0.71** for token log-probability baselines. Result is suggestive of probe utility but not yet statistically conclusive; confidence intervals overlap due to small sandbox size (n=103, 16 errors).

**Next:** Stage 5 — selective regeneration (the actual research question).

---

## What we built

### Project infrastructure

- Repository: `NLP_Lab` on GitHub. Modern Python project layout.
- Python package: `probe_extraction` under `src/`. Modular, testable.
- Configuration: YAML-driven via `configs/default.yaml`, validated by Pydantic.
- Stage-based scripts: each stage saves artifacts to disk, downstream stages read from disk. Means stages can be re-run independently.
- Editable install via `pyproject.toml` + `requirements.txt`. Runs cleanly on Kaggle.
- Centralized logging utility used by all scripts.

### Stage 1 — Extraction (Days 1-2)

**Built:**

- Abstract `Benchmark` interface (`src/probe_extraction/data/base.py`). Any new benchmark = one new loader file, no downstream changes.
- ExtractBench loader with PDF/gold pairing, schema-format detection (handles both wrapped and unwrapped schema variants).
- PyMuPDF-based PDF text extraction with image-only PDF detection.
- HuggingFace LLM wrapper supporting:
  - 4-bit / 8-bit / fp16 / bf16 quantization
  - Per-layer hidden-state capture during generation
  - Per-token log-probability extraction
  - Chat-template formatting
- Extraction prompt templates with schema-as-JSON-in-context.
- JSON output parser handling fenced/unfenced model outputs.
- Field localization in token space — maps each extracted leaf JSON value to its token positions.
- Activation slicing per field at configured position (last token / mean).
- Entry point script (`scripts/01_extract.py`) with progress tracking and per-doc artifact saving.

**Tested:** 16 smoke tests on real data. Passing.

### Stage 2 — Labeling (Day 2)

**Built:**

- Primitive value comparators (`value_compare.py`) for: exact, case-insensitive, fuzzy (Jaccard), number-with-tolerance, URL, email, date.
- Schema-driven matcher (`matcher.py`) that walks gold + extracted JSONs in parallel, picks comparison strategies from the schema's `evaluation_config`, produces per-field error labels.
- Five error categories tracked: match, value_mismatch, hallucination, omission, type_mismatch.
- Object array alignment: positional. Primitive array comparison: set-based.
- `anyOf` schema variant resolution for nullable fields.

**Tested:** 95 unit tests. All passing.

### Stage 3 — Probe training (Day 2)

**Built:**

- `LinearProbe` class wrapping sklearn LogisticRegression with metadata.
- `train_probe` function with stratified K-fold CV + held-out test split + final refit on full data.
- `ProbeMetrics` reporting AUROC, AUPRC, Brier score, threshold-for-recall.
- Script that loads activations + labels, trains one probe per layer, saves probes as pickles.
- Filtering: skips fields where the model emitted null/empty (synthetic activations not informative for probing).

### Stage 4 — Baseline comparison (Day 2)

**Built:**

- Token log-probability baseline (mean and min logprob per field).
- Evaluation script that runs probes and baselines on the same fields, reports head-to-head AUROC/AUPRC.

---

## Real issues we found and resolved

This is worth documenting because it shows what the test-first approach caught.

### 1. ExtractBench has two schema formats

Resume schemas wrap the JSON Schema in `{"name": ..., "schema_definition": {...}}`. Research schemas don't — JSON Schema is at the top level. Discovered by smoke test before any model code ran. Loader updated to detect and unwrap both formats.

### 2. Image-only PDFs in hiring/resume

5 of 7 resume PDFs are Google Docs exports without text layers — PyMuPDF returns nothing. Loader correctly flags these via `extraction_error` field. Sandbox domain switched from hiring/resume to academic/research, where all 6 PDFs have proper text. Resumes deferred until OCR is added or full-benchmark run.

### 3. Off-by-one in activation alignment

HuggingFace `generate()` returns hidden states for `n-1` steps when the final token is EOS. Initial extractor strictly required matching lengths and rejected every field. Fixed by clamping field token-spans to available activation length; the dropped position is the EOS or final `}` — not meaningful for probing.

### 4. P100 GPU memory limits

Free Kaggle GPU is P100 (Volta architecture, no FlashAttention support). Attention memory is quadratic in sequence length; long survey papers (20-30k tokens) trigger 14+ GB single-tensor allocations that don't fit in 16 GB. Resolved with input truncation (~4k tokens / 15k chars). Lossless for academic metadata extraction since title/authors/abstract live in the first 1-2 pages. One paper still fails due to cumulative memory creep across docs — known limitation, accepted at sandbox scale.

### 5. Cross-call GPU memory leak

Initial pipeline OOMed on doc 2 because PyTorch retained ~3 GB of cached activations from doc 1. Fixed by explicit cleanup (`del outputs`, `gc.collect()`, `torch.cuda.empty_cache()`, `torch.cuda.ipc_collect()`) after each generation. Improved but didn't fully eliminate creep on long survey papers.

---

## Sandbox results (academic/research domain, 5 of 6 documents)

### Stage 1: extraction

| Metric | Value |
|---|---|
| Documents successful | 5 of 6 |
| Documents failed (GPU OOM) | 1 |
| Total fields extracted | 199 |
| Empty fields (null/[]) | 78 |
| Useful fields (non-empty) | 121 |
| Total generated tokens | 3,525 |
| Total runtime | 11.5 minutes |
| Avg generation speed | ~3.5-4 tok/sec on P100 4-bit |

Per-document: 25-78 fields each, with 1229 generated tokens for the largest paper. Variance reflects different metadata richness across papers.

### Stage 2: labeling

| Metric | Value |
|---|---|
| Total fields labeled | 192 |
| Errors | 47 |
| Error rate | 24.5% |
| Hallucinations | 5 |
| Omissions | 28 |
| Value mismatches | 14 |
| Type mismatches | 0 |

Error rate is in the productive zone for probing research — neither so high that the model is broken, nor so low that there's nothing to detect. Errors dominated by omissions (60%) — the model frequently emits null where gold has values.

### Stage 3: probe training

After filtering synthetic activations (where model emitted null), training set had:

- 103 fields total
- 16 errors (15.5%)
- Per layer: 4 trained probes (layers 12, 16, 20, 24)

| Layer | CV AUROC (5-fold, mean ± std) |
|---|---|
| 12 | 0.785 ± 0.164 |
| 16 | 0.759 ± 0.186 |
| 20 | 0.755 ± 0.155 |
| 24 | 0.754 ± 0.167 |

All four probes outperform chance. Across-layer differences are smaller than cross-fold variance — cannot conclude which layer is best at this scale.

### Stage 4: head-to-head comparison

| Method | AUROC | AUPRC |
|---|---|---|
| Baseline: mean token logprob | 0.690 | 0.371 |
| Baseline: min token logprob | 0.706 | 0.349 |
| Probe (layer 12, CV) | 0.785 ± 0.164 | — |
| Probe (layer 16, CV) | 0.759 ± 0.186 | — |
| Probe (layer 20, CV) | 0.755 ± 0.155 | — |
| Probe (layer 24, CV) | 0.754 ± 0.167 | — |

**Best probe (layer 12) beats best baseline (min logprob) by 0.08 AUROC.**

---

## Honest interpretation

### Can claim

- Pipeline runs end-to-end on real data.
- Linear probes on Qwen2.5-7B residual stream activations carry signal about per-field extraction correctness.
- Probes outperform token log-probability baselines on the sandbox dataset.
- The advantage is consistent across all four captured layers.
- Token log-probability baselines themselves are non-trivial (~0.70 AUROC) — the bar for "probes are useful" is meaningfully higher than chance.

### Cannot claim (yet)

- Probes are statistically significantly better than baselines (confidence intervals overlap).
- Any layer is preferable to any other (within-noise differences).
- The result will hold at scale or across domains.
- Probe-guided selective regeneration improves cost-quality tradeoffs (Stage 5 not yet built — and this is the actual research question).

### What's needed for stronger claims

- **Stage 5: selective regeneration.** Use probe scores to drive thresholded regeneration. Compare Pareto curves (accuracy vs. compute) for probe-guided, baseline-guided, and uniform regeneration. This is where the project's contribution actually lives. Without it, we have "probes have higher AUROC than logprobs" — interesting but not the project's stated goal.
- **Scale.** 103 fields with 16 errors is sandbox-sized. Confidence intervals tighten roughly as 1/√N. Need either full ExtractBench (35 docs) or additional benchmarks (CORD, SciREX) for statistically meaningful conclusions.
- **Better compute.** P100 limits truncation length severely. A100 with FlashAttention would let us use full document context.

---

## Current state

- **Lines of code:** ~3,000 across ~35 source files
- **Tests:** 111 unit/integration tests passing
- **Pipeline:** Runnable end-to-end with `python scripts/0X_*.py --config configs/default.yaml`

### Repository structure

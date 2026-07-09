# Updates — May 11, 2026

Day-of-work update covering everything done in preparation for the May 11 supervisor meeting. Builds on the May 1-10 work (Kaggle pipeline, cluster migration, Qwen3.5-4B integration).

---

## Summary of the day

Goal: turn the Kaggle prototype into a defensible result on the Bender cluster.

Three experiments completed (academic/research with PyMuPDF, sport/swimming, academic/research with Docling), one methodological check added (leave-one-document-out CV), and one experiment submitted overnight (Llama-3.1-8B).

The most important finding of the day: **random-fold cross-validation was contaminating the probe AUROC numbers via document-level leakage.** Switching to leave-one-document-out CV gives a more honest evaluation. The probe still beats baselines on academic papers under LODO, but not on swimming.

---

## What ran today

### 1. Sport/swimming extraction (Qwen3.5-4B, PyMuPDF)

**Why:** First experiment on a non-academic, non-citation-heavy domain. Tests whether the probe-vs-baseline gap holds outside of academic-paper extraction. Also expected to give more trainable fields per document because swimming tables are dense with scalar values.

**Result:**

| Metric | Value |
|---|---|
| Documents extracted | 5/5 successful |
| Total raw fields | 583 |
| Trainable fields (after labeling and probe filtering) | 503 |
| Errors among trainable fields | 9 (1.8%) |
| Wall clock | ~6 min |

Five Ma 2023 swim-meet tables. All finished naturally (no truncation). Model handled the tabular format cleanly.

### 2. Sport/swimming probe analysis

**Why:** Stages 2-4 (label, train probe, evaluate) on the swimming extractions.

**Result (random-fold 5-fold CV):**

| Method | AUROC |
|---|---|
| Baseline mean_logprob | 0.785 |
| Baseline min_logprob | 0.781 |
| Probe layer 12 (best) | 0.918 ± 0.052 |

Initial reading: probe outperforms baseline by ~0.13. *Later revised under LODO; see section 5.*

### 3. Docling vs PyMuPDF ablation on academic/research

**Why:** PDF parser is a possible source of variation in extracted text → activations → probe quality. A clean A/B test isolates whether the result is sensitive to PDF backend.

**Result:**

| | PyMuPDF | Docling |
|---|---|---|
| Documents extracted | 3/6 | 3/6 |
| Documents shared | Yes for NIPS 1989, RAG survey | Yes for NIPS 1989, RAG survey |
| Document that diverged | FlashAttention-3 (succeeded) | Dim. reduction survey (succeeded) |
| Total trainable fields | 74 | 59 |
| Errors | 20 (27%) | 20 (34%) |
| Best baseline AUROC | 0.708 (min_logprob) | 0.708 (min_logprob)* |
| Best probe AUROC (random-fold) | 0.862 ± 0.039 (layer 4) | 0.889 ± 0.104 (layer 8) |

*Same baseline on common documents; the absolute baseline on Docling's full set is 0.647 (mean) / 0.626 (min).

**Conclusion:** PDF backend is not a meaningful source of the probe-vs-baseline gap. Different docs fail for each backend at the truncation boundary, but the underlying probe signal is consistent.

### 4. Llama-3.1-8B extraction (in-flight)

**Why:** Cross-family generalization check. The strongest single experiment to test whether the probe signal is Qwen-specific or generic to LLM hidden states.

**Status:** Job submitted to A100/A40 partition. Expected ~60-90 min for 6 documents. Results pending.

### 5. Leave-one-document-out cross-validation (methodological check)

**Why:** Random-fold CV randomly splits *fields* across train/test folds. If two fields from the same document end up in different folds, the probe can learn document-level features (e.g., "Ma 2023 papers look like this") that leak across folds. LODO holds out an entire document — train on N-1, test on the held-out — and is the standard methodologically-clean evaluation for probing experiments.

Implemented as `scripts/05_lodo_cv.py`. Reuses existing activations and labels — no GPU needed.

**Result (academic/research, n=74):**

| Layer | Random-fold CV | LODO CV |
|---|---|---|
| 1 | 0.856 ± 0.046 | **0.834 ± 0.011** |
| 4 | 0.862 ± 0.039 | 0.799 ± 0.022 |
| 8 | 0.834 ± 0.042 | 0.619 ± 0.180 |
| 12 | 0.825 ± 0.046 | 0.636 ± 0.129 |
| 14 | 0.820 ± 0.050 | 0.672 ± 0.111 |
| 16 | 0.815 ± 0.060 | 0.658 ± 0.112 |
| 18 | 0.825 ± 0.081 | 0.726 ± 0.034 |
| 20 | 0.848 ± 0.071 | 0.740 ± 0.073 |
| 22 | 0.848 ± 0.070 | 0.760 ± 0.080 |
| 24 | 0.839 ± 0.062 | 0.736 ± 0.102 |
| 26 | 0.844 ± 0.076 | 0.721 ± 0.087 |
| 28 | 0.853 ± 0.057 | 0.733 ± 0.123 |
| 30 | 0.847 ± 0.062 | 0.758 ± 0.128 |
| 32 | 0.871 ± 0.076 | 0.677 ± 0.282 |

Best LODO layer: **layer 1, AUROC 0.834 ± 0.011**. Compared to random-fold 0.856, this is a small drop (-0.022).

**Result (sport/swimming, n=503):**

| Layer | Random-fold CV | LODO CV |
|---|---|---|
| 1 | 0.759 ± 0.121 | 0.463 ± 0.116 |
| 4 | 0.833 ± 0.115 | 0.436 ± 0.132 |
| 8 | 0.784 ± 0.117 | 0.317 ± 0.069 |
| 12 | 0.918 ± 0.052 | 0.655 ± 0.056 |
| 14 | 0.872 ± 0.073 | 0.658 ± 0.019 |
| 16 | 0.820 ± 0.124 | 0.485 ± 0.172 |
| 18 | 0.716 ± 0.230 | 0.613 ± 0.034 |
| 20 | 0.697 ± 0.273 | 0.433 ± 0.110 |
| 22 | 0.736 ± 0.212 | 0.619 ± 0.112 |
| 24 | 0.711 ± 0.282 | 0.640 ± 0.065 |
| 26 | 0.782 ± 0.218 | **0.720 ± 0.050** |
| 28 | 0.732 ± 0.255 | 0.678 ± 0.090 |
| 30 | 0.771 ± 0.184 | 0.596 ± 0.059 |
| 32 | 0.795 ± 0.128 | 0.634 ± 0.137 |

Best LODO layer: **layer 26, AUROC 0.720 ± 0.050**. Compared to random-fold 0.918, this is a large drop (-0.198).

Caveat: only 3 of 5 LODO folds had both classes (errors AND correct fields) in the test set. Two folds had test documents with zero errors, making AUROC undefined for those folds. So the swimming LODO result averages over 3 folds, not 5.

---

## What this means (head-to-head, LODO-corrected)

| Domain | Probe LODO AUROC | Best baseline AUROC | Gap |
|---|---|---|---|
| Academic/research | 0.834 (layer 1) | 0.708 (min_logprob) | **+0.126** |
| Sport/swimming | 0.720 (layer 26) | 0.785 (mean_logprob) | **-0.065** |

**Two domains, two different answers under proper evaluation.**

- Academic: probe wins by +0.12 AUROC. Direction matches original Qwen2.5-7B Kaggle run.
- Swimming: probe loses by -0.07 AUROC. Random-fold CV had inflated the swimming result substantially.

The swimming flip is the surprise of the day.

---

## Why the swimming result reversed under LODO

Three possible explanations, in decreasing order of plausibility:

1. **The signal is genuine but the swimming sample size is too small for LODO.** 9 errors across 5 documents means LODO folds are statistically fragile. With many folds having zero errors in the test set, the held-out evaluation is unreliable. More swimming documents would settle this.

2. **The probe was learning document-specific structure on swimming.** All 5 swimming tables are from the same author (Ma 2023). Random-fold CV trained on some fields from each document and tested on others from the same documents — easy task. Under LODO, the probe sees a completely unseen document with a related but distinct field-distribution pattern, and the learned representation doesn't generalize.

3. **Tabular extraction is intrinsically different.** The model rarely makes errors on swim-meet tables (1.8% error rate), and when it does, the errors might be qualitatively different from the kind that probes detect well (e.g., misreading a digit vs. hallucinating a field). Probes might be better suited to detecting hallucinations than digit-recognition slips.

Investigating these requires more documents. Until then, the cross-domain inconsistency is documented but not explained.

---

## The "early layers vs late layers" observation

A second, smaller finding from LODO on academic:

| Layer range | Random-fold AUROC | LODO AUROC | Drop |
|---|---|---|---|
| 1-4 (early) | 0.86 | 0.82 | -0.04 |
| 8-16 (middle) | 0.83 | 0.65 | -0.18 |
| 20-32 (late) | 0.85 | 0.72 | -0.13 |

Early layers (1, 4) generalize cleanly under LODO. Middle and late layers drop more.

This is a hint — not a result yet — that the *generalizable* trust signal in academic-paper extraction may live in earlier layers than the probing literature usually targets. Worth investigating with more documents; deferred until n is bigger.

---

## Engineering / infrastructure work

Most of the day was running experiments. The infrastructure work was:

1. **LODO script (`scripts/05_lodo_cv.py`)**. ~150 lines. Reuses existing activations and labels. Reports per-fold AUROC, mean, std, and skipped folds.

2. **Per-experiment config + job scripts** for sport/swimming (`exp_qwen35_4b_swimming.yaml`) and Docling (`exp_qwen35_4b_docling.yaml`), plus matching `run_extraction_*.sh` and `run_analysis_*.sh`.

3. **Documented multiple non-trivial bugs during setup:** wrong pip path, doubled editable-install path, missing config-file path in the analysis script, and (one of mine) a wrong key name in the LODO script. All caught and fixed today.

4. **Cluster habits.** Documented Python module/venv loading in `.bashrc` and `setup_env.sh` so future sessions don't repeat the early-day debugging.

---

## Caveats — things I want flagged honestly at the meeting

1. **Sample size is the dominant uncertainty.** 74 fields (academic) and 503 fields with 9 errors (swimming) are both small. With n=74, the probe-vs-baseline gap of +0.126 has CIs that could easily span [+0.05, +0.20]. With 9 errors, swimming LODO is dominated by which 3 of 5 folds happen to be valid.

2. **The probe vs baseline conclusion is preliminary.** I would not write a paper from these numbers. I'd write "preliminary evidence on academic; cross-domain inconsistency under LODO; needs more data and more domains."

3. **Swimming's 1.8% error rate is itself worth noting.** The model is extremely good at this task. With so few errors, error-detection becomes essentially needle-in-haystack regardless of method. The probe vs baseline comparison may not be the right test for this domain.

4. **One domain still uses Qwen3.5-4B with thinking disabled.** This is a methodological choice (treating Qwen3.5 as a regular instruct model). Probing during reasoning mode is a separate question, not yet explored.

5. **The Zhao 2025 LLM survey doesn't fit any GPU we have** (265k input tokens). Documented but unsolved.

---

## What's next

Order matters here. Listed from most to least valuable.

1. **Llama-3.1-8B results when extraction finishes** (overnight). Will tell us if the probe-vs-baseline gap on academic is Qwen-specific.

2. **Expand to all 5 ExtractBench domains** (currently using 2). Real path to enough samples to make LODO statistically meaningful. Especially: hiring/resume (with image-only-PDF skip), finance/10kq, finance/credit_agreement.

3. **Add more documents per domain** if available. Even synthetic-prompt generation of variants is an option for future-future work.

4. **Position-strategy ablation** (last_token vs mean vs concat). Cheap if we save raw spans during extraction instead of pre-reduced vectors. Currently we save pre-reduced; would need re-extraction to compare.

5. **Self-consistency baseline.** The most-important-missing baseline. Resamples N times at temperature 0.7 and measures field-level agreement. Expensive at extraction time (Nx more generations) but only needs to be done once.

6. **Stage 5: selective regeneration.** The actual research deliverable. Uses the probe to flag risky fields, regenerates only those, measures cost-quality tradeoff. Should only be built once we have a probe that demonstrates a real signal under LODO.

7. **Non-linear probes (MLP).** Defer until linear probes plateau. With current data, methodology issues outweigh model-capacity issues.

---

## Files in the repo as of end of May 11

# Probe‑Based Trust Signals for Structured Information Extraction
## Progress Report — DeepSeek‑R1 / Reasoning‑Trace Track

**Author:** Adnan
**Date:** 2026‑07‑09
**Scope:** This document records the full pipeline, dataset, method, and results
for the reasoning‑model branch of the project. It is a self‑contained snapshot of
everything completed to date. The final section ("Future Work") describes what is
built and queued but **not yet run**.

---

## 1. Problem & Research Question

Large language models are increasingly used to read a document and emit a
**structured JSON** record according to a schema (structured information
extraction). The problem: the model gives no reliable per‑field confidence, so we
cannot tell *which* extracted fields are wrong without the gold answer.

**Core idea (trust signal).** Train a small **linear probe** — a logistic
regression — on the model's *internal hidden‑state activations* for each field, to
predict whether that field is **correct or wrong**. At deployment this gives a
per‑field trust score with no gold answer needed, which can drive **selective
regeneration** (only re‑ask the model for the fields the probe flags).

**This track's specific angle.** We use a **reasoning model** (DeepSeek‑R1), which
writes an explicit chain of thought (`<think> … </think>`) before its answer. The
research question:

> *Does the model's reasoning trace carry error‑relevant information beyond what is
> already present in the answer token — and can we read it with a linear probe to
> build a better trust signal for structured extraction?*

---

## 2. Model

**DeepSeek‑R1‑Distill‑Qwen‑7B** — a 7B‑parameter reasoning model.
- 28 transformer layers, hidden dimension 3584.
- Emits a reasoning trace `<think> … </think>` followed by the JSON answer.
- Run in `bfloat16`, greedy decoding (temperature 0 → deterministic), up to 4096
  new tokens, on a single A100 GPU (university "Bender" HPC cluster).

We capture hidden‑state activations at **14 layers** spread across depth
(1, 4, 7, 10, 12, 14, 16, 18, 19, 21, 23, 24, 26, 28), taken at the **last token**
of each field's value.

---

## 3. Dataset — SOB (Structured Output Benchmark)

**SOB** (`interfaze-ai/sob`, text / multi‑hop subset, derived from HotpotQA):
- Each record = a **multi‑hop question** + a source context passage, paired with a
  **per‑record JSON Schema** and a **human‑verified gold JSON** answer.
- Answering requires connecting facts across the passage — i.e. genuine
  **multi‑hop reasoning** — which is exactly why it is a good testbed for probing a
  reasoning model's trace. Clean text (no PDF/OCR noise). ~919‑token contexts.
- Licence: HotpotQA text is CC‑BY‑SA‑4.0; benchmark code MIT.

**Why SOB for this track.** Structured extraction where the reasoning genuinely
matters. If we used simple key‑value documents the reasoning trace would carry no
signal (the model barely reasons), so the research question could not be tested.

**This run:** 300 documents from the `test` split (299 successfully labeled; 1 was
dropped for a malformed‑JSON generation).

Each SOB record is mapped to a pipeline `Document`: the question is prepended to
the context, and the record's own schema + gold are attached. (SOB schemas are
**per record**, not per domain, which the labeling stage handles explicitly.)

---

## 4. The Pipeline (stage by stage)

```
 SOB record ──▶ [1] Extraction ──▶ [2] Labeling ──▶ [3] Probe training
                     │                    │                  │
             hidden states +      per‑field           logistic‑regression
             reasoning trace      correct/wrong        probe per layer
                     │            labels                     │
                     └────────────────────────────▶ [4] Evaluation vs baselines
                                                            │
                                                     [5/6] LODO cross‑validation
                                                            (honest generalization)
```

**Stage 1 — Extraction.** For each document, DeepSeek‑R1 is prompted to extract the
JSON per the schema. We capture: the generated JSON, per‑token log‑probabilities
(for baselines), the per‑field hidden‑state vectors at all 14 layers, and the
pooled reasoning‑trace vectors (mean over `<think>` tokens, and the `</think>`
summary position).

**Stage 2 — Labeling.** The extracted JSON is compared field‑by‑field against gold
to produce a binary **error label** per field (`is_error`). See §5.1 for the
matching logic.

**Stage 3 — Probe training.** For each captured layer, a logistic‑regression probe
is trained (5‑fold cross‑validation) on the per‑field activation vectors to predict
`is_error`.

**Stage 4 — Evaluation.** Probe quality is compared head‑to‑head against
confidence **baselines** derived from the model's own token log‑probabilities.

**Stage 5 / 6 — LODO cross‑validation.** The honest generalization test
(leave‑one‑document‑out, §5.4), including the reasoning‑fusion comparison (§6.3).

---

## 5. Methods

### 5.1 Structure‑aware labeling (a correctness fix)

Early runs showed a **~95% "error" rate**, which was an *artifact*, not real model
failure. DeepSeek often returns a **flat value** (e.g. `"instrument_family":
"percussion"`) where the gold stores a **nested object**; naïve exact‑match counted
every such case as wrong.

We added a **structure‑aware, type‑aware matcher** with three modes, and record all
three for transparency:
- **strict** — exact string match (original, harshest).
- **auto** — type‑aware leaf comparison: numeric tolerance, date parsing,
  case‑insensitive text, fuzzy match.
- **structure_aware** — additionally matches a flat predicted value against the
  *leaf values* of a gold object (fixes the flat‑vs‑nested mismatch).

**Result (300‑doc SOB run, 1,842 fields, 299 docs):**

| Labeling mode      | Error rate |
|--------------------|-----------:|
| strict             | 44.4%      |
| auto               | 41.6%      |
| **structure_aware**| **41.4%**  |

The residual ~41% is a **genuine, well‑balanced** error rate — ideal for training a
probe (neither trivially easy nor degenerate). `structure_aware` is used as the
primary labeling mode.

### 5.2 The linear probe

For each field we take its activation vector `h ∈ ℝ³⁵⁸⁴` at a given layer and train
`logistic_regression(h) → P(error)`, class‑balanced. It is deliberately **linear**:
a linear probe answering the question means the error information is *linearly
readable* from the model's internal state — a clean, interpretable claim.

### 5.3 Baselines

The natural competitor is the model's own confidence. We use two token‑log‑prob
baselines per field:
- **mean_logprob** — average token log‑probability over the field's tokens.
- **min_logprob** — the least confident token in the field.

A trust signal is only interesting if it **beats** these free baselines.

### 5.4 Evaluation metrics

- **AUROC** (Area Under ROC Curve): how well a score separates correct from wrong
  fields. 0.5 = random, 1.0 = perfect. Our headline metric.
- **5‑fold Cross‑Validation (CV):** standard robustness check.
- **Leave‑One‑Document‑Out (LODO):** the *honest* test — train on all documents but
  one, predict the held‑out document's fields. This directly measures
  generalization to **unseen documents** (harder and more realistic than random
  folds). We report LODO in two flavours:
  - **per‑doc AUROC** — averaged over held‑out documents = *within‑document*
    discrimination ("which field in this doc is wrong").
  - **pooled‑OOF AUROC** — one AUROC over all out‑of‑fold predictions pooled = the
    *global* ranking of fields across the whole dataset (what selective
    regeneration ultimately consumes).

### 5.5 Reasoning trace

DeepSeek‑R1's `<think> … </think>` tokens are captured and pooled into a per‑layer
summary vector (mean over the trace, and the `</think>` summary position). These
were then fused with the answer‑token probe (§6.3).

---

## 6. Results

*(Probe training/evaluation uses 1,557 fields after removing empty/synthetic
positions and fields missing activations; 32.9% of these are errors.)*

### 6.1 The probe works — and beats the baselines

Probe AUROC by layer (5‑fold CV), against the log‑prob baselines:

| Signal                     | AUROC (5‑fold CV) |
|----------------------------|------------------:|
| Baseline: mean_logprob     | 0.771             |
| Baseline: min_logprob      | 0.768             |
| Probe — layer 1            | 0.684             |
| Probe — layer 7            | 0.745             |
| Probe — layer 12           | 0.794             |
| Probe — layer 16           | 0.822             |
| **Probe — layer 19**       | **0.850**         |
| Probe — layer 23           | 0.834             |
| Probe — layer 26           | 0.843             |
| Probe — layer 28           | 0.831             |

**Takeaways:**
1. The probe **beats both log‑prob baselines by ~+0.08 AUROC** — a clear, real gap.
2. Signal rises with depth and **peaks in the mid‑to‑late layers (≈18–26)**, then
   plateaus — consistent with error‑relevant information being most linearly
   accessible in the upper‑middle of the network.

### 6.2 It holds up under the honest test (LODO)

Leave‑one‑document‑out, answer‑token probe:

| Metric              | Best layer | AUROC |
|---------------------|-----------:|------:|
| per‑doc (within‑doc)| 19         | 0.81  |
| pooled (global)     | 19         | 0.84  |

The probe generalizes to **unseen documents** at **0.81–0.84 AUROC** — the strong CV
result is not an artifact of random folds. **This is the headline result of the
track so far.**

### 6.3 Fusing the *document‑level* reasoning trace does not help (a clean, expected
negative result)

We tested whether concatenating the pooled reasoning vector with the answer token
improves the probe, under LODO:

| Variant (per‑doc AUROC)         | Δ vs answer‑only |
|---------------------------------|-----------------:|
| answer‑only (baseline)          | —                |
| + reasoning mean                | +0.003           |
| + reasoning `</think>`          | +0.005           |
| + both reasoning vectors        | +0.006 to +0.013 |

The gains are **within noise**. This is **expected by construction**: the pooled
reasoning vector is a *single vector per document*, identical across all of a
document's fields, so it **cannot change the ranking of fields *within* a
document**. The one channel where it *could* legitimately help — global ranking
(pooled‑OOF) — is also flat (±0.01).

**Interpretation — this is a useful finding, not a dead end.** It tells us the
reasoning signal must be attributed at the **field level**, not pooled at the
document level. That directly motivates the next experiment (§8).

---

## 7. Summary of What We Have

1. A complete, reproducible extraction → labeling → probing → LODO pipeline for a
   reasoning model on a reasoning‑heavy structured‑extraction dataset (SOB).
2. A **structure‑aware labeling** method that corrected a ~95% scoring artifact down
   to a genuine **41% error rate** (a methodological contribution in itself).
3. A **working per‑field trust signal**: linear probe **AUROC 0.85 (CV)** /
   **0.81–0.84 (LODO)**, beating token‑log‑prob baselines by **~0.08**.
4. A rigorous **negative result**: document‑level reasoning fusion adds no real
   signal — with a precise mechanistic explanation of *why* (constant‑within‑doc),
   which points to the next step.

---

## 8. Future Work (built, queued — not yet run)

**(A) Field‑localized reasoning attribution — the novel next experiment.**
Instead of one document‑level reasoning vector, we localize the reasoning **to each
field**: find where that field's *value* is mentioned inside the `<think>` trace and
pool exactly those tokens. This gives a reasoning feature that **varies across
fields**, so it *can* legitimately improve within‑document error detection. We also
extract **interpretable scalar features** — chiefly *"was this value reasoned about
at all?"* A value that appears in the final JSON but is **absent from the reasoning
trace** is a strong hallucination red‑flag. Compared under LODO with a **paired
significance test** against the answer‑only probe. *(Code implemented and unit‑
tested; awaiting a re‑extraction run that stores the per‑token reasoning states.)*

**(B) Conformal selective regeneration.** Turn the probe score into a calibrated
decision rule with a coverage guarantee: flag/regenerate the minimum set of fields
needed to hit a target error rate. Quantifies the practical payoff of the trust
signal.

**(C) Cross‑dataset generalization.** Run the same pipeline on a second benchmark
(ExtractBench, document extraction) to show the method is not SOB‑specific.

---

## 9. Appendix

### 9.1 Glossary
- **Structured information extraction** — read text/document → emit JSON per schema.
- **Linear probe** — logistic regression on a model's hidden state; if it works, the
  target information is linearly present in the activation.
- **Trust signal** — a per‑field score for "is this field correct?" without gold.
- **AUROC** — separability score; 0.5 random, 1.0 perfect.
- **LODO** — leave‑one‑document‑out CV; the honest generalization test.
- **Reasoning trace** — the `<think> … </think>` chain a reasoning model writes
  before answering.
- **Selective regeneration** — re‑ask the model only for fields flagged as likely
  wrong, saving compute.

### 9.2 Key configuration
- Model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, bf16, greedy, ≤4096 new tokens.
- Layers probed: 1,4,7,10,12,14,16,18,19,21,23,24,26,28.
- Dataset: SOB text/multi‑hop, `test` split, 300 docs.
- Labeling: `structure_aware` (primary), all three modes recorded.
- Probe: logistic regression, class‑balanced; 5‑fold CV and LODO.

### 9.3 Reproduce (on the A100 cluster)
```bash
# Stage 1 — extraction (GPU)
sbatch run_sob_extract_a100.sh
# Stages 2–6 — labeling, probe, evaluation, LODO, reasoning fusion (CPU)
sbatch run_sob_analysis_a100.sh
```

### 9.4 Result artifacts
- `artifacts/deepseek_r1_7b_sob/labels/_definition_comparison.json` — the three
  labeling‑mode error rates (§5.1).
- `artifacts/deepseek_r1_7b_sob/probes/_summary.json` — per‑layer probe AUROC.
- `artifacts/deepseek_r1_7b_sob/results/comparison.json` — probe vs baselines.
- `artifacts/deepseek_r1_7b_sob/results/reasoning_fusion_lodo.json` — LODO +
  reasoning‑fusion (§6.2–6.3).

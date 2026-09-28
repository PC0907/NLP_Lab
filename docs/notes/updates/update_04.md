# Project Updates — Probe-Based Trust Signals for Selective Regeneration

_Team #9 · ExtractBench · Bender HPC_
_Last updated: 2 June 2026_

---

## 1. Headline status

A linear probe on mid-network hidden states detects field-level extraction
errors substantially better than token-logprob baselines, under
leave-one-document-out (LODO) cross-validation — the methodologically honest
regime. The result holds on a cleaned, 25-document, four-domain pooled set.
Cross-model replication on Llama-3.1-8B is now underway (extraction complete).

### Primary result (Qwen3.5-4B, pooled, v2 / cleaned labels)

| Method | LODO AUROC |
|---|---|
| Baseline: mean log-probability | 0.764 |
| Baseline: min log-probability | 0.771 |
| **Linear probe (layer 18)** | **0.849 ± 0.14** |

- 25 documents, 2,115 trainable fields, 253 errors (12.0%).
- Probe beats the strongest baseline by **+0.078 AUROC** under LODO.
- Layer profile: layer 1 = 0.645 (weakest), rising to a peak at layer 18,
  declining to 0.759 at layer 32 — the signal lives in mid-network contextual
  representations, not surface lexical features.

> The headline moved from 0.865 (pre-matcher) to 0.849 after cleaning
> false-positive labels. The drop is within per-fold noise (±0.14) and is
> healthy: removing mislabelled "errors" the model actually got right left a
> cleaner, harder, more honest evaluation set. The probe-vs-baseline gap
> actually widened.

---

## 2. Work completed this period

### 2.1 Dataset completed (A100 recovery)
The finance/credit_agreement domain was missing 3 of 10 documents that
exceeded A40 GPU memory. These were re-extracted on the larger-VRAM A100
partition, bringing the pooled set to 25 documents and tightening LODO
variance.

### 2.2 Matching algorithm improved (AUTO comparison)
An audit found that ~91% of fields were being compared with strict,
case-sensitive string matching (the benchmark schemas rarely specify a
comparison method), producing false-positive "errors" where the model was
correct. A type-aware default ("AUTO") was added:
- numeric fields: tolerance-based, after stripping currency symbols/commas
  (so `$1,234` = `1234`, `49.7` = `49.70`; but `11.1` ≠ `11.3`);
- date fields: parsed across formats (`April 27, 2011` = `2011-04-27`),
  with a guard so fiscal-period strings (`FY2025 Q2` vs `Q1`) stay flagged;
- string fields: case-insensitive (not fuzzy), so case differences match but
  token-overlapping real errors stay flagged.

This cleared ~20 false-positive errors (concentrated in the legal-text credit
domain) and made the labels cleaner and more defensible.

### 2.3 Field exclusions (systematic gold/model mismatches)
Fields whose gold-vs-model disagreement is systematic and not a genuine trust
signal are excluded from the pooled probe set:
- 10kq `segment_name`, `data_period` — systematic metadata bias;
- academic `citations` — gold stores full citations, model emits short
  author-year keys; one set-valued field collapsing correct and incorrect
  entries into a single uninformative label.

### 2.4 Cross-model replication started (Llama-3.1-8B)
The full pipeline is being replicated on Llama-3.1-8B to test whether the
result generalises beyond a single model architecture. **Extraction is
complete across all four domains** (run on A100 for the memory-heavy domains).
Per-domain labelling and pooling are pending a directory-naming cleanup
(an experiment-naming inconsistency caused three domains to share one output
directory); this is a mechanical fix, not a modelling issue. Probe/LODO
results for Llama will follow.

---

## 3. New baselines (in progress)

Per request, additional baselines are being added beyond token log-probability,
spanning trivial controls to strong black-box methods. Two are implemented and
run on existing data (CPU); two require additional generation passes (GPU) and
are queued behind the Llama run.

| Baseline | Type | Status | Purpose |
|---|---|---|---|
| Token log-probability (mean, min) | scalar, black-box | done | basic uncertainty signal |
| Hand-crafted features | trained, white-box-free | implemented | deflationary control |
| Combined probe + log-probability | trained | implemented | tests signal complementarity |
| P(True) | scalar, black-box | queued (GPU) | model self-judgement |
| Self-consistency | scalar, black-box | queued (GPU) | strongest cheap black-box method |

(See accompanying baseline explainer for definitions and references.)

---

## 4. Data-quality findings (manual audit)

A document-by-document audit of flagged errors surfaced distinct failure modes.
Systematic mislabelling corrupts the probe (it learns a wrong rule); sporadic
mislabelling is tolerable noise that affects probe and baselines equally and
leaves the relative comparison valid.

1. **Matcher false positives** (case/number/date formatting) — fixed by AUTO.
2. **Gold under-annotation — swimming `records`**: when an athlete sets a
   record, the document shows a "WR"/"CR" marker that the model correctly
   extracts, but the gold leaves the field empty → false "hallucination".
   Confirmed against source documents.
3. **Gold incorrectness — swimming `event_details.length`**: a 200m event
   labelled "50m" in gold; the model correctly extracted "200m" and was
   penalised. An annotation error, not a model error.
4. **10kq omissions**: 10-Q gold encodes two periods per line item; the model
   emits only the current period, producing many "omissions" (correctly
   filtered from probe training). Under separate investigation.

**Implication:** a fraction of apparent model errors are gold-standard errors.
This places a noise floor on absolute AUROC, but because all methods are scored
against identical labels, the probe-vs-baseline comparison remains valid and
conservative. The probe's own false positives were the diagnostic that
surfaced these benchmark-quality issues.

---

## 5. Engineering notes

- **A100 environment:** A100 nodes use AMD CPUs (vs Intel A40), requiring
  `--export=NONE`, an AMD-stack module load, and a separate AMD-built venv.
  Now scripted and working; recovered all memory-heavy documents.
- **Llama is memory-heavier than Qwen** (8B vs 4B); long-input documents
  (credit agreements, large surveys) require the A100. The long-context
  *prefill*, not the output length, drives the memory usage.
- **Workflow:** edit locally → git push → git pull on cluster → run.

---

## 6. Next steps

1. Finish Llama-3.1-8B: directory cleanup → per-domain labelling → pool →
   probe/LODO. Cross-model comparison vs Qwen.
2. Run the two implemented baselines (hand-crafted, combined) on the pooled
   Qwen set; add P(True) and self-consistency once GPU frees from Llama.
3. Selective regeneration (Stage 5): threshold sweep → cost-quality Pareto
   curve, probe-guided vs baseline-guided. The project's intended deliverable.
4. Quantify the gold-error rate from a hand-audited sample (for the report's
   limitations section).

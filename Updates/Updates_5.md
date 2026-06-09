# Project Updates — Probe-Based Trust Signals for Selective Regeneration

_Team #9 · ExtractBench · Bender HPC_
_Week of 9 June 2026_

---

## 1. Headline status

- **Primary result (Qwen3.5-4B) holds and is clean:** linear probe on
  mid-network hidden states detects field-level extraction errors well above
  token-logprob baselines, under leave-one-document-out (LODO) CV.
- **Cross-model replication (Llama-3.1-8B) is underway:** the probe replicates
  on the swimming domain; the finance and academic domains hit a systematic
  label-contamination problem (Llama's output structurally diverges from the
  gold schema). This is documented separately — see
  `llama_label_contamination.md`.
- **Selective regeneration (the deliverable) is in progress:** the atomic
  single-field re-extraction works end-to-end; a value-in-text fixability
  filter is built; the cost-quality Pareto sweep is the next build.
- **New baselines:** two implemented (hand-crafted features, combined
  probe+logprob); two queued (P(True), self-consistency).

---

## 2. Primary result (Qwen3.5-4B, pooled)

| Method | LODO AUROC |
|---|---|
| Baseline: mean log-probability | 0.764 |
| Baseline: min log-probability | 0.771 |
| **Linear probe (layer 18)** | **0.849 ± 0.14** |

25 documents, 4 domains, ~2,115 trainable fields, 12% error rate. Probe beats
the strongest baseline by **+0.078 AUROC** under LODO. Signal peaks at a
mid-network layer; weak at layer 1, declining at the final layer — consistent
with the signal living in contextual representations, not surface features.

---

## 3. Cross-model replication (Llama-3.1-8B)

**Extraction complete on all four domains** (run on A100 for the memory-heavy
finance documents). After labelling, three of four domains showed implausibly
high error rates:

| Domain | Qwen error rate | Llama error rate |
|---|---|---|
| sport/swimming | ~10% | 10.2% |
| academic/research | low | 44% |
| finance/10kq | ~12% | 85% |
| finance/credit_agreement | moderate | 86% |

**Cause (documented in `Problem_with_llama.md`):** Llama does not emit
the schema's literal key structure. Where gold/Qwen use
`parties.administrative_agent` (snake_case, flat), Llama emits
`parties.Administrative Agent.address` (Title-Case keys with spaces, extra
nesting). The path-based matcher counts these as wholesale mismatches even
though the extracted *values* are often correct. The effect scales with schema
nesting, so swimming (flat) is clean and the deeply-nested finance schemas
explode. This is a model-specific extraction-formatting difference, not a
probe-quality issue.

**Decision:** pooling all four Llama domains would train the probe on ~85%
mislabelled finance data — corrupting it. We therefore report the Llama
cross-model result on **swimming only**, where labels are trustworthy.

**Llama swimming probe (LODO, 5 docs):** best layers reach **~0.94–0.95**
(layer 30 = 0.951 ± 0.031). Clearly above baseline; the probe replicates on a
second architecture.

**Important caveats on the cross-model comparison:**
- Both single-domain swimming probes are trained on only 5 documents, so LODO
  estimates are **noisy** (few folds). The Qwen swimming-only LODO is in fact
  unstable (≈0.72, and only 3 of 5 folds valid because some test documents had
  no errors) — a data-starvation artefact, NOT evidence that Qwen's probe is
  weak. Qwen's reliable result is the **pooled 25-doc** 0.849.
- The honest reading is **qualitative**: in both models the probe beats
  baseline and the signal sits in mid-to-late layers. Precise numbers across
  differently-sized samples (Qwen pooled 25-doc vs Llama swimming 5-doc) are
  not directly comparable and should not be headlined as "Llama > Qwen".
- The exact peak layer differs by model (Qwen pooled ~18, Llama swimming ~30)
  — expected for different architectures; the *shape* (weak early, strong
  mid/late) replicates, the precise depth does not.

---

## 4. Selective regeneration (deliverable, in progress)

**Goal:** use the probe to flag likely-wrong fields, re-extract only those
("selective" regeneration), and trace a cost-quality Pareto curve — probe-
guided regeneration should dominate logprob-guided, random, and approach an
oracle.

**Progress:**
- **Atomic operation works end-to-end:** pick a probe-flagged field, build a
  targeted single-field re-extraction prompt (schema description resolved via
  a `$ref`-aware walker), re-extract at temperature > 0, compare to gold via
  the AUTO matcher. Confirmed running on GPU.
- **Key design finding:** naive single-field re-extraction of a deeply-nested
  field underperforms — stripped of sibling context, the model returns null.
  The mechanism should be **correction-with-context** (show the field's
  sibling values, ask the model to re-derive just the target). To be finalised.
- **Value-in-text fixability filter built** (`08_fixability_filter.py`): for
  each error, checks whether the gold value is even present in the extracted
  document text. Errors whose answer is absent (PDF-parsing loss, or metadata
  like page count) are **unfixable by regeneration** and must not count against
  the probe in the cost-quality analysis. This quantifies how much of the error
  set is genuine model error vs upstream extraction loss — and defines the fair
  denominator for regeneration.

**Next:** finalise the correction-with-context prompt; run the fixability
filter across domains to set the fixable denominator; build the threshold sweep
and Pareto curve (probe vs logprob vs random vs oracle).

---

## 5. New baselines (per supervisor request)

| Baseline | Type | Status |
|---|---|---|
| Token log-probability (mean, min) | scalar | done |
| Hand-crafted surface features | trained, LODO | implemented |
| Combined probe + log-probability | trained, LODO | implemented |
| P(True) (Kadavath et al. 2022) | scalar, GPU | queued |
| Self-consistency (Wang 2022 / Manakul 2023) | scalar, GPU | queued |

Semantic entropy (Farquhar 2024) is deliberately excluded: it clusters
free-form generations by meaning via NLI, which does not fit short structured
field values; self-consistency is its structured-output-appropriate substitute.

---

## 6. A discovered data-quality issue worth tracking

The fixability work surfaced that some "errors" are **upstream PDF-extraction
losses**, not model errors — e.g. `number_of_pages` (gold = 18, correct) is
metadata not present in the flattened PyMuPDF text, so neither model can
recover it from the text it was given. This connects to the supervisor's
suggestions: (a) a value-in-text check as an objective parsing-quality metric,
and (b) retrying Docling for more faithful extraction. The fixability filter is
the first step toward quantifying this.

---

## 7. In progress / tracked for coming work

- Finalise selective regeneration: correction-with-context prompt → fixability-
  filtered threshold sweep → cost-quality Pareto curve vs baselines. (deliverable)
- Run the two implemented baselines on the pooled Qwen set; add P(True) and
  self-consistency once GPU frees.
- Qwen swimming-only result is noisy by design (5 docs); not headlined.
- Methodological hardening (from supervisor feedback, planned):
  pre-registered audit thresholds for field exclusion / matcher / prompt
  changes; nested-LODO or precommitted layer selection to remove test-set
  layer-selection bias; Docling retry + shipping extracted text for
  reproducibility.
- Llama finance/academic: documented as structural contamination; full
  multi-domain cross-model comparison would require schema-constrained decoding
  (future work).

---

## Accompanying documents
- `llama_label_contamination.md` — full diagnosis of the Llama structural /
  key-naming mismatch, why it contaminates labels, and ranked solutions.

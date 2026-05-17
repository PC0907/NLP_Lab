# Updates — May 17, 2026

Progress update covering the work done since the May 11 supervisor meeting.
Builds on the prior Qwen3.5-4B experiments (academic/research, sport/swimming,
and the LODO methodological check).

---

## Summary of the day

The headline result: a **pooled four-domain probe**, evaluated under
leave-one-document-out (LODO) cross-validation, **outperforms the
token-logprob baseline** — the first multi-domain confirmation that the probe
trust signal is real and transferable.

Two strands of work led here:

1. A dataset evaluation effort to address the sample-size limitation flagged
   at the last meeting. This is summarised briefly below; full detail is in a
   separate document (`dataset_evaluation_realkie.md`).
2. Building and running the pooled four-domain experiment on ExtractBench,
   which produced the result.

---

## 1. Dataset evaluation (brief — see separate document)

The May 11 meeting identified **sample size** as the dominant limitation: the
academic domain had only 74 trainable fields, and the swimming result
collapsed under LODO partly due to too few errors.

To address this, the RealKIE benchmark was evaluated as a larger-scale
candidate. Two of its datasets (NDA, then fcc_invoices) were assessed and both
found unsuitable — RealKIE's annotations are not complete enough per document,
and its line-item fields are tabular, which the pipeline's matching does not
support. **Full reasoning, evidence, and the recommended dataset requirements
are documented separately in `dataset_evaluation_realkie.md`.**

The evaluation was not wasted: it produced reusable pipeline capabilities
(a benchmark-dispatch mechanism, a set-membership matching mode, a
recursion-bug fix) and a **dataset gate report tool** (`check_dataset.py`)
that screens any candidate benchmark before adoption.

The conclusion of that effort was to **return to ExtractBench and use all
available domains** — its gold is complete and trustworthy, which is the
property RealKIE lacked. The rest of this document covers that work.

---

## 2. ExtractBench: full-domain coverage check

A no-GPU load-time check was run across all five ExtractBench domains to
measure how many documents yield usable text:

| Domain | Documents | Usable |
|---|---|---|
| academic/research | 6 | 6 (1 excluded later — see below) |
| finance/10kq | 7 | 7 |
| finance/credit_agreement | 10 | 7 (3 OOM — see below) |
| hiring/resume | 5 | 0 — image-only PDFs, no text layer |
| sport/swimming | 5 | 5 |

`hiring/resume` was dropped: all five PDFs are image-only and yield no text
without OCR, which the pipeline does not perform. The other four domains are
usable.

Two known exclusions:

- **Zhao 2025 LLM survey** (academic) — 860k characters, exceeds available GPU
  memory. Excluded, as previously documented.
- **3 of 10 credit agreements** — the largest documents (460-516k characters)
  hit CUDA out-of-memory on the A40 GPU (44 GB). A rerun on the larger-memory
  A100 partition is queued; in the meantime the experiment uses the 7
  successfully extracted credit agreements. An infrastructure issue with the
  A100 partition (an incompatible-CPU node causing `Illegal instruction`
  errors) is being followed up with HPC support.

---

## 3. The pooled four-domain experiment

The four usable domains were extracted with Qwen3.5-4B (bf16, thinking
disabled) and their labels + activations pooled into a single combined
dataset for probe training and evaluation.

**One labelling decision.** The `finance/10kq` domain produced a misleading
84% raw error rate. Inspection showed ~90% of those "errors" were *omissions*
(the model not extracting a field — filtered out for probing anyway) and that
the genuine value-mismatches were heavily concentrated in two metadata fields,
`segment_name` and `data_period`, which the model gets wrong almost every
time. Such systematic, single-field errors would let a probe "succeed" by
learning field identity rather than a trust signal. These two field types
were therefore **excluded** from the pooled dataset. The other domains'
errors are diverse and were kept in full.

**Pooled dataset:**

| Property | Value |
|---|---|
| Domains | academic, swimming, 10kq, credit_agreement |
| Documents | 20 |
| Usable fields (extracted, non-synthetic) | 2,065 |
| Errors | 247 (12.0%) |

---

## 4. Results

Probes (logistic regression, one per layer) were trained on the pooled
dataset and evaluated three ways: standard random-fold CV, LODO CV, and
against the token-logprob baselines.

**Head-to-head, best layer (layer 18):**

| Method | AUROC |
|---|---|
| Baseline — mean_logprob | 0.764 |
| Baseline — min_logprob | 0.772 |
| Probe — random-fold CV | 0.926 |
| **Probe — LODO CV** | **0.828 ± 0.13** |

**LODO AUROC by layer (the honest, document-level metric):**

| Layer | LODO AUROC |
|---|---|
| 1 | 0.626 |
| 4 | 0.719 |
| 8 | 0.763 |
| 12 | 0.778 |
| 14 | 0.796 |
| 16 | 0.802 |
| **18** | **0.828** |
| 20 | 0.796 |
| 24 | 0.804 |
| 32 | 0.719 |

(18 of 20 LODO folds were valid; 2 folds had a held-out document with only one
class, leaving AUROC undefined.)

**Reading the results:**

- The probe **beats the strongest baseline by +0.056 AUROC under LODO**
  (0.828 vs 0.772). The trust signal is real and outperforms black-box
  token-confidence.
- The signal **survives document-level cross-validation**. The random-fold to
  LODO drop is moderate (0.93 -> 0.83) — not the collapse seen in the
  swimming-only experiment (0.92 -> 0.72, which fell *below* baseline). This is
  the key contrast: pooled four-domain probing generalises to unseen
  documents.
- Both CV methods agree the generalisable signal lives in **mid-network
  layers** (peak at layer 18). This differs from the earlier academic-only
  finding, where early layers generalised best — pooling multiple domains
  shifted the signal deeper.

---

## 5. Caveats — flagged honestly

1. **The gap is modest.** +0.056 AUROC is smaller than the academic-only gap
   (+0.126). A likely reason: pooling brought in finance domains where the
   token-logprob baseline is itself stronger (the model is genuinely
   low-confidence on wrong financial figures), narrowing the gap. The probe
   still wins, but by less — and this is itself informative about where
   probes add the most value.

2. **High per-fold variance.** LODO std is ±0.13 across 18 folds. The +0.056
   gap is real on average but not large relative to noise. This is suggestive,
   not statistically conclusive; significance has not been formally tested.

3. **Sample size is still modest.** 20 documents, 2,065 fields, 247 errors.
   This is a substantial improvement over the academic-only setup (74 fields)
   and is enough for a meaningful LODO, but it is not a large-scale study. The
   result should be presented as preliminary multi-domain evidence.

4. **3 credit agreements and the Zhao survey are excluded** for GPU-memory
   reasons. Recovering the 3 credit agreements (A100 rerun) would bring the
   document count to 23.

5. **Single model.** All results use Qwen3.5-4B. Cross-model confirmation
   (Llama-3.1-8B) is the most important outstanding robustness check.

---

## 6. What's next

In order of priority:

1. **Cross-model replication with Llama-3.1-8B.** Does the LODO probe-vs-
   baseline gap hold on a different model architecture? This is the strongest
   robustness check and would significantly firm up the result.

2. **Recover the 3 OOM credit agreements** via the A100 rerun, bringing the
   pooled set to 23 documents — pending resolution of the A100 partition
   infrastructure issue with HPC support.

3. **Stage 5: selective regeneration** — the project's actual deliverable. The
   pipeline now has a probe that demonstrably beats the baseline under LODO,
   which is the prerequisite for building the cost-quality Pareto curve for
   selective regeneration.

4. **Self-consistency baseline** — the most important missing baseline, for a
   more complete comparison.

---

## 7. Status

The probe trust signal is **confirmed under LODO across four pooled domains**
and **outperforms the token-logprob baseline** (0.828 vs 0.772). The result is
preliminary — modest gap, modest sample size, single model — but it is the
first multi-domain, methodologically-clean positive result.

---

## 8. Questions from Last week

Two questions raised the previous week concerned *which token and which layer*
the probe captures, with a suggested embedding-only control.

**Which token is captured.** The concern was that the probe might read a
structural JSON token (a closing quote or comma) rather than a content token.
This was checked empirically: the generated text of every field was
re-tokenised and the token at the `last_token` activation position classified
as content vs structural. Result, over 600 sampled fields:

| Position | Content | Structural | Whitespace |
|---|---|---|---|
| `span[1]-1` (the captured `last_token`) | **100.0%** | 0% | 0% |
| `span[1]` (the index after) | 1.0% | 94.7% | 4.3% |

The captured token is the **final content token of the field value** in 100%
of sampled fields. The `token_span` end index is exclusive, so the closing
quote sits at `span[1]` while the captured position `span[1]-1` is genuine
content. The structural-token confound is not present. (Note: Qwen3.5-4B is
decoder-only and has no `[CLS]` token; the relevant risk was a JSON delimiter,
and that risk is ruled out.)

**Which layer.** The existing pooled LODO results already address the
deflationary "surface features" concern. Layer 1 — nearest the embedding
layer — is the *weakest* layer (LODO AUROC 0.626), with performance climbing
to a mid-network peak (layer 18, 0.828) before tapering. If the probe were
detecting lexical or surface features, the earliest layers would perform best.
The increase from layer 1 to layer 18 is the signature of contextual
computation, not vocabulary. The embedding-only control (a probe on layer-0
activations) would make this fully rigorous and is planned (Section 9).

**A related finding.** The token inspection surfaced within-document value
duplication — e.g. all seven authors of one paper sharing an identical
affiliation string, producing near-identical activations. These are not
independent training examples: they inflate the effective sample size and
contribute to the high per-fold LODO variance (±0.13). A deduplication or
down-weighting refinement is noted for future work.

---

## 9. Completed: token-position ablation

The extractor supports `last_token`, `mean`, and `all` position-reduction
strategies; all results above use `last_token`. To test whether that choice
matters, two domains (academic + credit_agreement) were re-extracted with
`position: mean` and compared against `last_token` on the **identical 8
documents** that both runs successfully extracted (the others hit GPU OOM or
JSON truncation — the same documents failing in both runs, so the comparison
set is matched).

**LODO AUROC, best layer each:**

| Strategy | Best layer | LODO AUROC |
|---|---|---|
| `last_token` | 20 | 0.861 |
| `mean` | 28 | 0.831 |

`last_token` edges out `mean` by ~0.03 at peak — directionally consistent with
the literature (HaMI, arXiv:2504.07863, finds last-token strongest among fixed
token choices). **However, with only 8 documents / 8 LODO folds and per-fold
std of ±0.12-0.19, a 0.03 gap is within noise.** The honest conclusion is that
the position choice does not make a large difference at this sample size, and
the existing `last_token` default is validated rather than beaten. A side
observation: `mean` shifts the best layer deeper (28 vs 20).

This ablation also surfaced one consistently pathological document
(NIPS-1989), whose per-fold AUROC sits below 0.5 across most layers for both
strategies — a single document depressing the mean and inflating variance.
Worth a closer look; noted for follow-up.

---

## 10. Planned experiments

In rough order of effort:

- **Embedding-only control** (layer 0). Re-extract with layer 0 added to the
  captured layers and probe on it. A layer-0 probe can only learn which tokens
  appeared, with no contextual computation — so layer-0 ≈ mid-layer would be a
  deflationary result, and layer-0 ≪ mid-layer confirms the signal is genuine.
  Directly tests the layer-1 observation above.

- **Cross-model replication (Llama-3.1-8B).** Does the LODO probe-vs-baseline
  gap hold on a different architecture — the most important robustness check.

- **Cross-Layer Attention Probing (CLAP, arXiv:2509.09700).** A probe that
  attends jointly over all captured layers, rather than one independent probe
  per layer. All 14 layers' activations are already saved, so this is
  feasible, but it requires a new probe architecture (not a config change) and
  is scoped as a larger future-work item.

- **Stage 5: selective regeneration** — the project deliverable, now unblocked
  by a probe that beats the baseline under LODO.

---

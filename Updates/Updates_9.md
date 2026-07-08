# Weekly Update — Scaling Comparison, Baseline Subsumption, Cross-Dataset Transfer

_Team #9 · ExtractBench + insurance-claims · Bender HPC · Qwen3.5 family_

## Summary

Four results this week, three of which reinforce the same conclusion — the trust
signal is **robust and fundamental, not fragile or benchmark-specific**:

1. **Model-scaling (2B/4B/9B):** on an identical document set, the probe's
   cross-document trust signal **strengthens then plateaus with scale**
   (pooled-OOF AUROC 0.60 -> 0.92 -> 0.93) while the extractor error rate drops.
2. **Baseline subsumption (Task 1):** stacking probe + logprob + hand-crafted
   adds only +0.003 over the probe alone — the probe subsumes the black-box
   baselines.
3. **Layer-conflict null:** cross-layer disagreement adds nothing over the
   single best layer (-0.002) — a third independent "probe is saturated" result.
4. **Cross-dataset transfer (headline):** a probe trained on ExtractBench detects
   errors on insurance-claims **zero-shot at 0.87-0.88 AUROC** — comparable to
   within-dataset training, **bidirectional and layer-stable**. The trust signal
   is dataset-independent.

---

## 1. Model-scaling comparison: 2B / 4B / 9B (identical document set)

Each model truncates/OOMs on different documents, so a fair comparison requires
the **intersection** of documents all three extracted, labeled, and that are
usable as LODO folds (both classes present): **15 documents** (1 academic, 10
credit-agreement, 4 swimming; 10kq excluded throughout). All three were run
through the same nested-LODO restricted to these 15 docs.

| Model | pooled-OOF AUROC | pooled-OOF AUPRC | per-fold AUROC | error rate | layer |
|---|---|---|---|---|---|
| Qwen3.5-2B | 0.604 | 0.596 | 0.877 +/- 0.112 | ~34% | 14 (11/15) |
| Qwen3.5-4B | 0.915 | 0.700 | 0.870 +/- 0.106 | 11.1% | 12 (13/13) |
| Qwen3.5-9B | 0.928 | 0.775 | 0.833 +/- 0.236 | 11.4% | 14 (12/14) |

**Two findings:**

- **Extractor improves with scale.** Error rate on the shared docs falls ~34% ->
  ~11% (2B -> 4B/9B). Bigger models make fewer extraction errors.
- **Trust signal strengthens then plateaus.** Pooled-OOF AUROC 0.60 -> 0.92 ->
  0.93 (AUPRC 0.60 -> 0.70 -> 0.77): weak at 2B, strong from 4B up, saturating
  (4B ~= 9B). The probe-based trust signal generalises across model scale and its
  reliability increases with capability.

**Why pooled-OOF, not per-fold, is the comparison axis.** The 4B/9B make almost
no swimming errors, so swimming folds are error-starved (1-2 positives ->
near-random per-fold AUROC, e.g. 9B swimming folds at 0.14 and 0.54), inflating
the per-fold std (9B +/-0.236) without reflecting probe quality. Pooled-OOF
computes one AUROC over all held-out fields, immune to tiny-fold saturation, and
is the metric a single global regeneration threshold actually relies on.

**Secondary finding — cross-document calibration improves with scale.** Compare
per-fold (within-doc ranking) vs pooled-OOF (across-doc ranking): the 2B shows a
large gap (0.877 vs 0.604 — ranks errors within a document but its scores do not
transfer across documents), while 4B and 9B show no gap. Larger models produce
trust signals calibrated *across* documents, which is what selective regeneration
(one global threshold) needs.

**Layer stability holds across scale.** Selected layers 14 (2B), 12 (4B,
unanimous), 14 (9B) — all mid-network (~40-58% relative depth) despite different
depths and architectures.

**Caveats.** n=15, predominantly credit (10/15); small and credit-heavy. Per-fold
AUROC is confounded by swimming error-starvation and should not be compared
across models; pooled-OOF is the axis. The 2B's 0.604 was verified against its
results JSON and is corroborated by the monotonic curve.

**Infra enabling this.** Domain filtering (`--exclude-domains`, previously the
scripts globbed all labels and silently reincluded 10kq), document-intersection
restriction (`--include-docs-file`), and pooled-OOF AUROC added to the output.
9B run: config layer band widened to [8,12,14,16,18,20,24,28]; A100 sbatch fixed
for Bender (`--gpus`, `unset SLURM_EXPORT_ENV`); 25/28 docs (3 lost: 2 truncation,
1 A100 OOM on the 860k-char survey).

---

## 2. Baseline subsumption (Task 1: three-signal stacking)

Stacking all three signals (probe + logprob + hand-crafted) into one classifier,
under LODO on the 4B all-domains set:

| Signal | standardized coefficient (median) |
|---|---|
| probe_score | **2.845** |
| logprob | 0.213 |
| hand_crafted | -0.030 |

- Combined AUROC **0.882** vs probe alone **0.879** -> **delta +0.003**.
- The probe coefficient dominates (~13x logprob); hand-crafted is ~0 (slightly
  negative = redundant/collinear once the probe is present).

**Reading:** the probe subsumes both black-box baselines. Logprobs add a sliver,
hand-crafted features add nothing. Adding signals to the probe does not improve
detection — the internal signal already captures what the external signals offer.
This justifies the white-box approach (no need to engineer features or rely on
logprobs, which per CONSTRUCT are often unavailable on APIs anyway).

---

## 3. Layer-conflict signal (REDD-inspired) — null

Test of whether cross-layer *disagreement* adds error-detection signal over the
single best layer (REDD reports inter-layer conflict predicts errors). On the 4B
clean set:

- conflict alone (spread across layers): AUROC 0.688 — weakly informative on its own.
- **best-layer probe: 0.898; best-layer + conflict scalar: 0.896; delta -0.002.**
- layer ensemble (mean of all layers): 0.880 vs best single 0.898 — averaging is
  slightly worse than picking the best layer.

**Reading:** layer-disagreement is *informative but redundant* — it carries error
signal, but nothing the single-layer probe hasn't already captured. A methodology
note: the naive version of this test (stacking two scalars through a second
logistic regression) spuriously showed +0.047 because the meta-step degraded the
single-layer baseline from 0.898 to 0.774; the corrected test (conflict appended
to the full activation vector) gives the true -0.002.

**This is the third independent "probe is saturated" result** (with the
three-signal subsumption and the ensemble result): neither external features,
nor layer ensembling, nor cross-layer disagreement improves on the single
mid-network probe. The trust signal is low-dimensional and lives at one layer.

---

## 4. Cross-dataset transfer (headline result)

**Question:** does a probe trained on ExtractBench detect errors on a completely
independent extraction benchmark (CONSTRUCT/Cleanlab insurance-claims) — with NO
retraining? This is the strong form of generalisation (train-on-A, apply-to-B
zero-shot), distinct from "retrain on insurance and it works" (which we also
confirmed: 0.863 pooled-OOF).

Self-contained experiment: train a probe on all ExtractBench-4B clean-domain
activations at a shared layer, apply it unchanged to all insurance activations at
the same layer. Both raw and L2-per-sample normalisation reported. Swept both
directions and layers 8-24.

**Transfer AUROC (raw activations):**

| layer | ExtractBench -> insurance | insurance -> ExtractBench |
|---|---|---|
| 8  | 0.826 | 0.788 |
| 12 | 0.856 | 0.847 |
| 14 | 0.874 | 0.861 |
| 16 | **0.881** | 0.863 |
| 18 | 0.877 | **0.871** |
| 20 | 0.875 | 0.868 |
| 24 | 0.854 | 0.859 |

**Findings:**

- **Zero-shot transfer is strong and near-parity with retraining.** A probe that
  saw only financial/academic/swimming documents flags errors on insurance at
  0.88 AUROC — essentially equal to retraining on insurance (0.863). The trust
  signal is **dataset-independent**, not benchmark-specific.
- **Transfer is bidirectional.** Forward peaks 0.881, reverse peaks 0.871 —
  symmetric, despite insurance being the smaller training set. The error signal
  is genuinely shared between the two datasets, not one subsuming the other.
- **Transfer is layer-stable.** Both directions plateau at 0.85-0.88 across the
  mid-network band (12-24), weaker only at layer 8 (early). Not a lucky single
  layer — the whole mid-network carries transferable signal, matching the
  within-dataset layer findings.
- **Raw > L2 in both directions.** The error signal lives in raw activation
  directions/magnitudes that transfer directly; normalising slightly hurts. The
  signal is geometrically stable across datasets, not just present.

**Claim:** the probe's trust signal transfers zero-shot between two independent
extraction benchmarks in both directions and across the mid-network layer band,
at AUROC comparable to within-dataset training. A probe trained on one extraction
benchmark detects errors on a different one — different documents, schema, and
error distribution — without retraining.

**Caveats.** One model (4B) and two datasets; a strong two-point result, not a
universal law — frame as "transfers between these two independent benchmarks."
The transfer test set is 499 insurance fields at 26.3% errors (the `extracted_present`
field population; the retrained-LODO run used a slightly different 474/19.2% set).

**Independent-benchmark viability (prerequisite for the above).** Insurance was
validated as a document-extraction fit: a schema-key smoke test confirmed Qwen3.5
reproduces its nested schema 100% literally (no Llama-style key divergence); a
text-native loader was built (mirroring the RealKIE pattern, no PDF round-trip);
extraction gave 30/30 docs, 474 fields, 19.2% errors — a trainable target.

---

## 5. Scope note: SOB dataset evaluated and rejected

The JigsawStack SOB benchmark (17.7k records, clean gold, nested schemas) was
evaluated as a possible additional dataset. On inspection it is **structured QA**
(context + question -> reason to an answer), not document extraction — errors are
multi-hop reasoning errors, not extraction errors. This is out of scope for the
project's extraction focus ("identify risky *extracted* fields"), so it was
rejected rather than run. Noted as possible future cross-task-type work. Insurance
(genuine document extraction) is the correct independent benchmark, and delivered
the transfer result above.

---

## 6. Status of headline numbers

| Quantity | Number | Status |
|---|---|---|
| Scaling pooled-OOF AUROC 2B/4B/9B (15 shared docs) | 0.604 / 0.915 / 0.928 | Signal strengthens then plateaus |
| Scaling error rate 2B/4B/9B | ~34% / 11% / 11% | Extractor improves with scale |
| Three-signal: combined vs probe | 0.882 vs 0.879 (+0.003) | Probe subsumes baselines |
| Three-signal coefficients (probe/logprob/handcrafted) | 2.85 / 0.21 / -0.03 | Probe dominant |
| Layer-conflict: best-layer + conflict | 0.896 vs 0.898 (-0.002) | Null; probe saturated |
| Insurance retrained (nested LODO) | 0.863 pooled-OOF | Method generalises |
| **Transfer EB->insurance (raw, L16)** | **0.881** | **Signal dataset-independent** |
| **Transfer insurance->EB (raw, L18)** | **0.871** | **Bidirectional** |

---

## Next steps

1. **LLM-as-judge black-box baseline** — the project calls for comparison to
   black-box baselines; the strong one (LLM-judge / CONSTRUCT-style verifier) is
   not yet implemented. Reuses existing ExtractBench docs/gold; needs a verifier
   model (GPU job). Highest-value in-scope build remaining.
2. **Extend transfer** — optionally a third extraction dataset, or per-domain
   transfer breakdown (which ExtractBench domains transfer best to insurance).
3. **Consolidate the "probe is saturated" results** (three-signal, conflict,
   ensemble) into a single robustness argument for write-up.

## Principle this week

Three separate attempts to improve on the single mid-layer probe — external
features, layer ensembling, cross-layer conflict — all returned nulls, and the
probe transferred zero-shot across datasets at near-parity with retraining.
Together these say the trust signal is **fundamental and robust**: low-dimensional,
living at one mid-network layer, and dataset-independent. The recurring discipline
was separating real signal from measurement artifact — the swimming error-starvation
that made 9B look degraded (it wasn't), and the meta-step that made conflict look
helpful (it wasn't).

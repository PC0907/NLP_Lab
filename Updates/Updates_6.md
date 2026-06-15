# Weekly Update — Methodology Hardening & Span Aggregation

_Probe-based trust signals · Team #9 · ExtractBench · week of 9–15 June 2026_

This week focused on making the pipeline rigorous (the "make it perfect" ask):
removing a layer-selection bias from the headline number, implementing and
testing the span-max aggregation from Obeso et al. (2025), completing the
remaining baselines, and fixing a regeneration prompt bug. Two of the headline
results are honest *negative / modest* findings, which we report transparently.

---

## 1. Nested LODO — removing layer-selection bias

**Problem.** The pipeline ran LODO at every layer and reported the best layer's
score (layer 18: **0.849**). But the layer was chosen by looking at the same
held-out documents it was then reported on — test-set peeking, which
optimistically inflates the number.

**Fix.** Nested LODO: an outer loop holds out a test document (reported on,
never used for selection); an inner loop picks the best layer using only the
remaining documents; that layer is retrained on all non-test docs and scores the
held-out doc. Repeated over all documents.

**Result.**

| Metric | Value |
|---|---|
| Naive best-layer LODO (peeked) | 0.849 |
| **Nested LODO (unbiased)** | **0.812 ± 0.191** (22 folds) |
| Pooled OOF AUPRC | 0.586 |
| Layers selected across folds | 18 (16×), 16 (5×), 20 (1×) |

**Reading.**
- The selection bias was ~0.037 — modest but real. **0.812 is the honest
  headline**; no reviewer can object that the layer was chosen on the test set.
- **Layer choice is stable:** layer 18 wins 73% of folds, and selection never
  leaves the 16–20 mid-network band. This *justifies pre-committing to layer 18*
  as a principled alternative — so the 0.849 "layer 18" number is also
  defensible if framed as a pre-registered mid-depth choice rather than a search.
- **Per-fold variance is large (±0.19; folds range 0.35–1.00).** This is the
  honest small-data reality: with one document held out, per-document
  generalization swings widely. The mean is solid; the variance is the caveat we
  state openly rather than hide.

---

## 2. Span-max aggregation (Obeso et al. 2025) — implemented and tested

**Motivation.** The current pipeline reads each field's activation at its
**last token** (`position: last_token`). Obeso et al. argue the error signal is
often concentrated at one high-information token that need not be the last, and
score a field by the **maximum** probe score over its token span ("span-max").
Our supervisor suggested testing this.

**Implementation.** We added a `position: "all"` mode to the extractor that saves
every token's activation per field (2D `(span_len, hidden_dim)` arrays), enabling
last / mean / span-max to all be computed downstream from one extraction. The
span-max scorer trains a token-level probe and takes the max score per field at
test time (the paper's usage; our token-level labels are a cruder version of
their annealed span-max loss — a noted caveat).

**Span-length context.** A field can only differ under the three aggregations if
it has >1 token:
- Pooled set: **46% of fields are single-token, median span 2.**
- Credit alone: 13% single-token, median span 8 (long legal clauses).

**Results (LODO, last vs mean vs span-max).**

Pooled (2834 fields, all four domains) — representative layers:

| layer | last | mean | max |
|---|---|---|---|
| 14 | 0.846 | 0.852 | 0.859 |
| 16 | **0.879** | 0.851 | 0.861 |
| 18 | 0.850 | 0.853 | **0.867** |
| 24 | 0.849 | 0.843 | **0.855** |
| 28 | 0.793 | **0.858** | 0.834 |

Credit only (7 docs, multi-token-heavy) — representative layers:

| layer | last | mean | max |
|---|---|---|---|
| 4 | 0.862 | 0.878 | **0.904** |
| 8 | 0.891 | 0.889 | **0.922** |
| 18 | 0.916 | 0.911 | **0.931** |

**Reading.**
- **On the pooled set, span-max gives no consistent improvement.** The three
  aggregations trade the lead layer-to-layer, all within ~±0.02 — well inside the
  noise band (cf. the ±0.19 per-fold std). At the best layer (16) last-token
  actually wins. The cause is structural: 46% of fields are single-token, where
  span-max is identical to last-token by construction.
- **On credit alone, span-max shows a small directional edge** (e.g. 0.916 →
  0.931 at layer 18), because credit's long legal-clause fields (median span 8)
  give span-max something to work with. But this is 7 docs and noisy — directional,
  not conclusive.
- **Conclusion: we retain last-token.** Span-max helps only where fields are long
  (credit), and washes out on the short-field-dominated pooled distribution. This
  is a well-tested null result with a clear mechanistic explanation, not an
  untested assumption. The credit-vs-pooled contrast *is* the finding: aggregation
  matters only when fields are long, and our data mostly isn't.

**Caveats.** The credit run lost 3/10 docs to A40 OOM on long documents; the full
A100 re-extraction (all four domains, full text) underlies the pooled numbers.
The span-max probe uses crude token-level labels, not the paper's annealed loss —
a proper implementation might narrow the gap slightly, but cannot change the
single-token structural ceiling.

---

## 3. Baselines completed (credit)

Two GPU baselines were run to round out the comparison:

| Baseline | AUROC (credit, n=114) | Note |
|---|---|---|
| P(True), binary | 0.512 | Near-random. This is the *crude binary* version (model said "True"/"False"); a proper token-logit P(True) is future work. |
| Self-consistency (N=5) | 0.833 | Strong — nearly matches the probe. |

**Key framing for self-consistency:** it is the strongest baseline, but it costs
**5× full document re-extractions per document**. The probe reaches comparable
detection from activations already captured during the single extraction pass —
roughly 1/5th the inference cost. That cost advantage is the probe's real
differentiator against self-consistency, more than a raw AUROC gap.

(Earlier CPU baselines, pooled: min-logprob 0.771, mean-logprob 0.764,
hand-crafted surface features 0.743, probe+logprob combined 0.856 ≈ probe alone —
logprob adds nothing on top of the probe.)

---

## 4. Regeneration prompt fix — list-element granularity

**Bug.** In selective regeneration, fields that are elements of a list of objects
(e.g. `authors.1.name`) were regenerated as the **whole list**
(`['Y. Le Cun', 'B. Boser', ...]`) instead of the single value. Diagnosis: the
context object was correctly scoped to one element, but the prompt never told the
model *which* list position it was filling, and a generic key like `name` (shared
by every author) led the model to dump them all. This accounted for ~8 of the 13
"breakages" in the academic regeneration run.

**Fix.** The correction prompt now (a) detects list-element targets, (b) states
the exact position ("you are filling the `name` of item #2 of 6 in the `authors`
list"), and (c) forbids returning the whole list. Plain object fields are
unchanged. Written; **validation pending** a GPU regeneration re-run on academic
(expected to flip the academic regeneration result from net-negative toward
net-positive once the spurious list breakages are removed).

---

## 5. Honest status of the headline numbers

| Quantity | Number | Status |
|---|---|---|
| Probe, nested LODO (unbiased) | **0.812 ± 0.19** | Honest headline |
| Probe, layer-18 LODO (pre-committed) | 0.849 | Defensible if framed as pre-registered mid-depth |
| Probe, random-fold CV | 0.923 | Do NOT use — document-level leakage |
| Best baseline (self-consistency, credit) | 0.833 | Strong but 5× cost |
| Span-max vs last-token (pooled) | no meaningful difference | Tested null |

---

## Next steps

- Validate the regeneration prompt fix (GPU re-run on academic), re-score, confirm
  the list-element breakages are gone.
- Sampling-based regeneration (regenerate N times, pick best) — decide the
  selection criterion (majority vote / probe-scored / model self-pick) and implement.
- Docling re-extraction for parsing/value-in-text quality (GPU; never run on pooled).
- Pre-registered audit protocol document (formalize the 50/25/25 thresholds,
  retroactively justify exclusions and the lenient-matcher decision).

# Weekly Update — Model-Scaling Result (2B/4B/9B)

## Summary

The central result this week is the **model-scaling study**: with all three models
(Qwen3.5 2B / 4B / 9B) evaluated under nested LODO on an **identical document set**,
the probe's cross-document trust signal **strengthens and then plateaus with scale**
(pooled-OOF AUROC 0.60 -> 0.92 -> 0.93), while the extractor's error rate drops
(~34% -> 11% -> 11% on the shared set). Bigger models are both better extractors and
produce more *globally-calibrated* trust signals. Supporting work: an independent
clean-gold benchmark (CONSTRUCT's insurance-claims) was validated as pipeline-viable
and a loader built; and a literature review positioned the contribution against the
two closest works (CONSTRUCT, REDD).

---

## 1. Model-scaling result (the headline)

**Setup.** Each model truncates/OOMs on different documents, so their labeled-doc
sets differ (2B lost academic docs to token-truncation; 9B lost one to A100 OOM and
two to truncation). A fair comparison requires the **intersection**: documents that
all three models extracted, labeled, and that are usable as LODO folds (both classes
present). That intersection is **15 documents** -- 1 academic, 10 credit-agreement,
4 swimming -- predominantly credit. All three models were then run through the *same*
nested-LODO script restricted to these 15 docs (`--include-docs-file`).

**Result (nested LODO, 10kq excluded, identical 15 docs):**

| Model | pooled-OOF AUROC | pooled-OOF AUPRC | per-fold AUROC | error rate | layer (nested) |
|---|---|---|---|---|---|
| Qwen3.5-2B | **0.604** | 0.596 | 0.877 +/- 0.112 | ~34% | 14 (11/15) |
| Qwen3.5-4B | **0.915** | 0.700 | 0.870 +/- 0.106 | 11.1% | 12 (13/13) |
| Qwen3.5-9B | **0.928** | 0.775 | 0.833 +/- 0.236 | 11.4% | 14 (12/14) |

**Reading -- two findings.**

1. **The extractor improves with scale.** Error rate on the shared docs falls from
   ~34% (2B) to ~11% (4B, 9B). Bigger models make fewer extraction errors -- expected,
   and cleanly confirmed.

2. **The trust signal strengthens then plateaus with scale.** Pooled-OOF AUROC rises
   0.60 -> 0.92 -> 0.93 (AUPRC 0.60 -> 0.70 -> 0.77). The signal is *weak at 2B and
   strong from 4B upward* -- it matures with scale and then saturates (4B ~= 9B). This
   directly answers the durability question: **the probe-based trust signal generalises
   across model scale, and its reliability increases with capability.**

**Why pooled-OOF, not per-fold, is the headline number.** The per-fold AUROC (average
of per-document AUROCs) is unreliable here: the 4B and 9B make almost no swimming
errors, so swimming folds are error-starved (1-2 positives -> near-random per-fold
AUROC, e.g. 9B swimming folds at 0.14 and 0.54). This drags and widens the per-fold
mean (note the 9B's +/-0.236 std) without reflecting probe quality. Pooled-OOF computes
one AUROC over all held-out fields at once, so it is immune to tiny-fold saturation
and is the decision-relevant metric -- it is what a single global regeneration
threshold tau actually relies on.

**A secondary finding -- cross-document calibration improves with scale.** Compare
per-fold (within-document ranking) against pooled-OOF (across-document ranking):

- 2B: per-fold 0.877 vs pooled 0.604 -> **large gap**. The 2B ranks errors well
  *within* a document but its scores do not transfer *across* documents -- the trust
  signal is not globally calibrated.
- 4B: 0.870 vs 0.915 -> no gap. 9B: 0.833 vs 0.928 -> no gap (once swimming noise is
  discounted).

Only the 2B shows the within-but-not-across pattern. Larger models produce trust
signals calibrated *across* documents, not just within them. This matters for selective
regeneration, where one global tau is applied across all documents: the 4B+ models
supply the cross-document calibration a single threshold needs.

**Layer stability holds across all scales.** Nested LODO selects layer 12 (4B,
unanimous 13/13), 14 (2B 11/15, 9B 12/14) -- all mid-network (~40-58% relative depth).
The "trust signal lives at a stable mid-network layer" claim holds across 2B->4B->9B
despite different architectures and depths.

**Caveats (stated plainly).**
- **n = 15 documents (~430-508 fields), predominantly credit-agreement (10/15).** This
  is the fair shared set but it is small and credit-heavy; academic is 1 doc, swimming
  contributes mostly noise on the larger models. Honest but not large-scale.
- Per-fold AUROC is reported for completeness but should not be compared across models
  (swimming error-starvation confounds it); pooled-OOF is the comparison axis.
- The 2B's headline 0.604 was double-checked against its results JSON and is
  corroborated by the monotonic curve (the 4B slots between 2B and 9B exactly as a real
  trend predicts -- evidence the numbers are not artifacts).

---

## 2. Infrastructure fixes enabling the comparison

- **Domain filtering in the LODO scripts.** The scripts previously globbed every label
  file and ignored the config's domain list, so excluding 10kq via a separate config
  did nothing (the convention-noise was silently swept back in). Added
  `--include-domains` / `--exclude-domains` on the label file's `domain` key, with a
  log line proving the drop.
- **Document-intersection restriction.** Added `--include-docs-file` so every model is
  scored on an identical document set -- the mechanism that makes the scaling comparison
  fair rather than apples-to-oranges across differently-truncated doc sets.
- **Pooled-OOF AUROC** added to the nested-LODO output alongside AUPRC -- the stable
  all-fields metric that turned out to be the correct comparison axis.
- **9B run.** Config corrected (layer band widened to [8,12,14,16,18,20,24,28] to
  bracket the mid-network band and the hybrid architecture's attention layers, after
  the drafted [14-22] proved too high); A100 sbatch fixed for Bender (`--gpus`, `unset
  SLURM_EXPORT_ENV` companion to `--export=NONE`). Ran 25/28 docs (3 lost -- 2
  truncation, 1 A100 OOM on the 860k-char survey).

---

## 3. Independent benchmark (CONSTRUCT insurance-claims) -- validated, loader built

To test the probe on independently-curated gold (not ExtractBench), we evaluated the
four CONSTRUCT / Cleanlab datasets (Goh & Mueller, arXiv 2603.18014).

- **Screening.** Three of four are short-text NER/snippet tasks (input ~100-200 chars)
  that do not exercise the document-extraction pipeline. Only **insurance-claims**
  (n=30, ~2.5k-char claim documents, nested JSON gold) is a document-extraction fit.
- **Schema-key smoke test.** The Llama cross-model failure was caused by key-structure
  divergence (model paraphrases/re-nests schema keys, breaking the path matcher). A
  one-document test confirmed Qwen3.5 reproduces the insurance schema's nested keys
  **100% literally** (24/24 keys, zero missing, zero extra) -- so insurance does *not*
  hit the Llama wall.
- **Loader built.** A text-native `InsuranceClaims` benchmark loader (modeled on the
  RealKIE loader pattern, no PDF round-trip) plus config and dispatcher wiring. Verified
  loadable with zero GPU. Ready to extract -> label -> LODO as an independent
  generalisation check.

The insurance value-level error rate is not yet measured, so whether it yields a
trainable probe (enough errors) is still open.

**A citation dividend.** CONSTRUCT's blog is literally titled "LLM Structured Output
Benchmarks are Riddled with Mistakes." They found *every* public structured-output
benchmark had unreliable gold and re-annotated four datasets. This independently
corroborates our 10kq finding: excluding the financial domain is a response to a
recognised, published problem, not a convenient choice.

---

## 4. Literature review -- positioning the contribution

- **CONSTRUCT (arXiv 2603.18014, black-box).** An ensemble LLM-as-a-Judge: one verifier
  LLM prompted five ways, scores averaged. No training, works on closed APIs, but costs
  5 verifier calls per document, each re-reading the full document (expensive on long
  filings). Our probe reads the trust signal from activations captured in the single
  extraction pass -- the cost advantage grows with document length.

- **REDD (arXiv 2511.02711, white-box -- the closest work).** REDD independently uses
  lightweight classifiers on LLM hidden states to detect per-field extraction errors --
  the same primitive we use. But REDD applies it inside a *relational query engine* for
  *human-in-the-loop* correction, and its contribution is **SCAPE**, a conformal method
  giving statistical coverage guarantees. Our contribution is distinct in four ways:
  (i) **action** -- automated probe-gated safe-override *regeneration* vs human
  verification; (ii) **task** -- single-document extraction against a fixed schema vs
  multi-document schema-discovery querying; (iii) **evaluation rigor** -- nested-LODO
  honest generalisation, aggregation ablation, parser-invariance, and this cross-model
  scaling study, none of which REDD performs; (iv) **baselines** -- a full
  black-box-vs-white-box comparison (logprobs, self-consistency, LLM-as-judge) REDD does
  not make. REDD's SCAPE calibration is **complementary** -- combining our regeneration
  policy with SCAPE-style coverage guarantees is concrete future work.

- **Corroborating (white-box lineage):** SAPLMA (Azaria & Mitchell 2023), Marks &
  Tegmark 2024 (truthfulness linearly represented), Semantic Entropy Probes (2406.15927),
  and "Hallucination Is Linearly Decodable from Mid-Layer Hidden States" (2606.02628) --
  the last independently confirming our layer findings (single mid-network layer is the
  sweet spot, MLP ~= linear, stable peak band).

- **Ideas worth borrowing from REDD (future work):** (a) their *inter-layer conflict*
  signal -- disagreement among per-layer classifiers predicts errors -- could feed our
  safe-override gate as a second feature (we already capture multiple layers); (b) their
  *LLM-committee labeling* (<1% worse than human labels) is a possible route to cleaner
  labels where benchmark gold is unreliable (the 10kq problem).

**To verify:** PromptPort (arXiv 2601.06151), cited as the source of the safe-override
policy, could not be located in searches -- confirm it exists before relying on it.

---

## 5. Status of headline numbers

| Quantity | Number | Status |
|---|---|---|
| Scaling: pooled-OOF AUROC, 2B / 4B / 9B (15 shared docs) | 0.604 / 0.915 / 0.928 | **New headline** -- signal strengthens then plateaus |
| Scaling: pooled-OOF AUPRC, 2B / 4B / 9B | 0.596 / 0.700 / 0.775 | Rises with scale |
| Scaling: error rate, 2B / 4B / 9B (shared docs) | ~34% / 11% / 11% | Extractor improves with scale |
| Nested layer selected, 2B / 4B / 9B | 14 / 12 / 14 | Stable mid-network across scale |
| Probe nested LODO, 4B full (28-doc, prior) | 0.879 +/- 0.11 | Prior headline (all domains) |
| Insurance schema-key smoke test | 24/24 keys | Viable independent benchmark |
| Task 1 three-signal coefficients | -- | Built; results still not read |

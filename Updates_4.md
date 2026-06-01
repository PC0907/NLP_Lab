# Project Updates — Probe-Based Trust Signals for Selective Regeneration

_Team #9 · Qwen3.5-4B · ExtractBench · Bender HPC_
_Last updated: 1 June 2026_

---

## 1. Headline status

The pooled four-domain probe result is confirmed and has improved with a larger,
cleaner dataset. A linear probe on mid-network hidden states continues to
outperform token-logprob baselines under leave-one-document-out (LODO)
cross-validation — the methodologically honest regime.

| Metric | Value |
|---|---|
| Domains pooled | 4 (academic, finance/10kq, finance/credit, sport/swimming) |
| Documents | 25 |
| Trainable fields | 2,115 |
| Errors (positives) | 253 (12.0%) |
| Best layer | 18 (of 32) |
| **Probe LODO AUROC (v1 labels)** | **0.865 ± 0.13** |
| Baseline mean_logprob (AUROC) | 0.764 |
| Baseline min_logprob (AUROC) | 0.771 |
| Probe CV AUROC (optimistic, random-fold) | 0.923 |

> **Pending:** LODO AUROC under the new (v2 / AUTO-matcher) labels has not yet
> been recorded. The CV number above is post-matcher; the LODO re-run is the
> single outstanding command. The v1 LODO (0.865) is the last confirmed honest
> figure.

---

## 2. What changed this period

### 2.1 Dataset completed via A100 recovery
The finance/credit_agreement domain previously had only 5-7 of 10 documents
(three exceeded A40 GPU memory). These were re-extracted on the larger-VRAM
A100 partition and re-labelled, bringing credit to a full 10 documents. The
pooled set grew from 20 to 25 documents. This tightened LODO variance and raised
the headline LODO AUROC from a prior ~0.828 (20 docs) to 0.865 (25 docs).

A100 jobs require a distinct environment setup (`--export=NONE` plus an
AMD-stack module load and a separate AMD-built virtual environment) because the
A100 nodes use AMD EPYC CPUs while the A40 nodes are Intel. This is now scripted
and working.

### 2.2 Matching algorithm improved (AUTO comparison strategy)
Audit of flagged errors revealed that the labelling step defaulted to strict,
case-sensitive string comparison for ~91% of fields (the benchmark schemas
rarely specify an explicit comparison method). This produced false-positive
errors where the model was in fact correct.

A type-aware default comparison ("AUTO") was added:
1. **Numeric** fields compared with tolerance, after stripping currency symbols
   and thousands separators — so `$1,234` matches `1234`, and `49.7` matches
   `49.70`, while genuinely different numbers (e.g. `11.1` vs `11.3`) stay
   flagged.
2. **Date** fields parsed across formats — so `April 27, 2011` matches
   `2011-04-27` — with a guard that prevents fiscal-period strings (e.g.
   `FY2025 Q2`) from being mis-parsed as dates, since those represent real
   (systematic) period errors that must remain flagged.
3. **String** fallback is case-insensitive (not fuzzy), so case/whitespace
   differences match but token-overlapping real errors stay flagged.

Effect on labels (false positives cleared):

| Domain | Errors before | Errors after |
|---|---|---|
| academic | 26 | 25 |
| finance/10kq | 7,641 | 7,634 |
| finance/credit | 77 | 65 |
| sport/swimming | 13 | 13 |

The change cleared ~20 false-positive errors, concentrated in the legal-text
credit domain. The pooled positive count fell from 273 to 253. The headline
AUROC was essentially unchanged (the false positives were a small fraction),
but the labels are now cleaner and the comparison more defensible.

### 2.3 Field exclusions (systematic gold/model mismatches)
Some fields exhibit systematic gold-vs-model disagreement that is **not** a
genuine trust signal, and are excluded from the pooled probe set so the probe
learns transferable error detection rather than field identity:

- **10kq `segment_name`, `data_period`** — systematic metadata bias
  (already excluded).
- **academic `citations`** — gold stores full bibliographic strings; the model
  emits short author-year keys. A single set-valued field collapses ~11 correct
  short citations and a few genuine garbage entries into one binary label,
  making it uninformative. Now excluded.

---

## 3. Data-quality findings (from manual audit)

A document-by-document audit of flagged errors surfaced several distinct
failure modes. These matter because **systematic** mislabelling corrupts the
probe (it learns a wrong rule), whereas **sporadic** mislabelling is tolerable
noise that affects probe and baselines equally and leaves the relative
comparison valid.

1. **Matcher false positives (equivalence)** — case, number-format, and
   date-format differences flagged as errors. *Fixed* by the AUTO matcher.

2. **Gold under-annotation — swimming `records`** — when an athlete set a
   record, the document shows a "WR"/"CR" marker and the model correctly
   extracts it, but the gold annotation leaves the field empty, producing a
   false "hallucination". Confirmed against the source document in at least two
   PDFs. Candidate for field exclusion (not yet enabled; small effect given
   only 13 swimming errors).

3. **Gold incorrectness — swimming `event_details.length`** — at least one 200m
   Individual Medley event is labelled "50m" in gold; the model correctly
   extracted "200m" and was penalised. This is an annotation error, not a model
   error. Treated as sporadic label noise (a noise floor on absolute AUROC),
   not fixed, since excluding the whole field would discard correct labels too.

4. **10kq omissions (schema mismatch)** — 10-Q gold encodes two periods per
   line item (current + prior comparison); the model emits only the current
   period, producing thousands of "omissions" (the `.1.` prior-period entries).
   These are correctly filtered from probe training. Under separate
   investigation; not a probe-result concern.

**Implication for the write-up:** a fraction of apparent model errors are in
fact gold-standard errors. This places an irreducible noise floor on absolute
AUROC. Because all methods are evaluated against identical labels, the
probe-vs-baseline comparison remains valid and, if anything, conservative
(label noise generally compresses measured differences). The probe's own false
positives were the diagnostic that surfaced these benchmark-quality issues.

---

## 4. Per-layer LODO profile (v1 labels, 25 docs)

The layer profile supports the interpretation that the trust signal lives in
mid-network contextual representations, not surface lexical features:

| Layer | LODO AUROC |
|---|---|
| 1 | 0.694 |
| 8 | 0.826 |
| 14 | 0.849 |
| 16 | 0.850 |
| **18** | **0.865** |
| 20 | 0.845 |
| 24 | 0.837 |
| 32 | 0.764 |

Layer 1 (closest to embeddings) is weakest; the signal peaks at layer 18 and
declines toward the late layers, ruling out a surface-feature explanation.

---

## 5. Engineering notes

- **A100 environment:** jobs need `#SBATCH --export=NONE`, a sourced AMD-stack
  setup (`setup_env_a100.sh`), and the AMD-built venv (`~/nlp_lab_a100`).
  Informational commands in job scripts use `|| true` to avoid SIGPIPE-killing
  the job (a prior failure cause).
- **AUTO matcher wiring:** the comparison strategy enum and dispatch were
  initially incomplete (the `AUTO` value and its dispatch branch were missing),
  so earlier re-label attempts silently produced no change. Now corrected and
  validated against a 10-case test harness.
- **Workflow:** edit locally -> `git push` -> `git pull` on cluster -> run.

---

## 6. Next steps

In rough priority order:

1. **Record the v2 LODO number** (single command) to lock the post-matcher
   honest headline.
2. **Cross-model replication — Llama-3.1-8B.** Full pipeline with the same AUTO
   matcher, so the comparison is clean. Unblocked by the working A100 setup.
3. **Additional trust-signal baselines** (per supervisor request): combined
   probe + logprob classifier (cheapest, likely AUROC bump, new baseline);
   then hand-crafted features, P(True), and self-consistency.
4. **Selective regeneration (Stage 5):** threshold sweep producing the
   cost-quality Pareto curve vs baseline-guided regeneration — the project's
   intended deliverable.
5. **Optional:** enable swimming `records` exclusion; continue the 10kq
   omission investigation.

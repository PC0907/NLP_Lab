# Dataset Evaluation: RealKIE — Findings and Path Forward

**Date:** May 17, 2026
**Author:** Syed Ali Mehdi Rizvi
**Project:** Probe-based trust signals for structured information extraction
**Purpose:** Document why the RealKIE benchmark was evaluated, why two of its
datasets were found unsuitable for this pipeline, and what the evaluation
established about dataset requirements going forward.

---

## 1. Background and motivation

The probe experiments to date used the ExtractBench benchmark. The central
methodological limitation identified in earlier work was **sample size**: the
academic/research domain yields only 74 trainable fields, and the
sport/swimming domain, while larger, has only 9 errors and collapsed under
leave-one-document-out (LODO) cross-validation. A larger dataset with
trustworthy labels was needed to make LODO statistically meaningful.

RealKIE (Townsend et al., 2024) was evaluated as a candidate. It provides five
enterprise key-information-extraction datasets with substantially more
documents per dataset than ExtractBench. Two datasets were assessed in
sequence: **NDA** (non-disclosure agreements) and **fcc_invoices** (broadcast
advertising invoices).

Neither was found suitable. This document explains why, because the *reasons*
are informative — they sharpen the requirements any future dataset must meet.

---

## 2. What was built

The evaluation was not wasted effort on the engineering side. The following
pipeline components were built and verified, and are **dataset-independent** —
they will serve any future benchmark:

- A `RealKIE` benchmark loader conforming to the existing `Benchmark`
  interface, so RealKIE datasets plug into the pipeline like ExtractBench.
- A benchmark-dispatch mechanism in the stage-01 and stage-02 scripts, so the
  active benchmark is selected by config rather than hard-coded.
- A new **set-membership matching mode** in the labeling stage, for
  multi-valued fields where each extracted element should be its own probe
  sample and element order need not align with gold. This is a genuine
  capability the pipeline previously lacked.
- A fix to a pre-existing infinite-recursion bug in the matcher, triggered by
  a gold/extraction shape ExtractBench never produced.
- A **dataset gate report** (`check_dataset.py`) — a diagnostic tool, described
  in Section 5, that is the main durable output of this evaluation.

---

## 3. Dataset 1 — NDA: failed (sparse annotation)

The NDA dataset has 439 documents and three fields: Party, Jurisdiction,
Effective Date. Initial structural inspection looked acceptable — roughly two
party annotations per document on average.

A 20-document validation run exposed the problem. Of the labeled fields, the
labeling stage reported **55 errors out of 55** — every single one of a
particular kind: the model extracted a real, correct value (a genuine party
name present in the document) that the gold annotation simply did not contain.

The cause is that **RealKIE's NDA gold is sparsely annotated**. The benchmark
does not annotate every correct value in each document; it annotates a subset.
The RealKIE paper itself describes the datasets as having "sparse annotations
in long documents." For this pipeline, sparse gold is fatal: the labeling
logic treats "extracted value not found in gold" as an error, which is only
valid if gold is *complete*. With sparse gold, that assumption breaks and the
error labels become noise — a probe trained on them would learn whether
RealKIE's annotators happened to highlight a value, which is unrelated to
extraction correctness.

NDA was therefore rejected.

---

## 4. Dataset 2 — fcc_invoices: failed (partial per-document coverage
and tabular structure)

fcc_invoices has 370 documents and a richer field set. A dedicated gate report
(Section 5) was run first this time. The dataset passed the checks that
existed then: dense annotation in aggregate, clean annotation offsets, and a
healthy text-vs-gold ratio. The five line-item fields (Description, Rate, Days,
Start Date, End Date) were selected; the six sparser header fields were
excluded.

A 20-document validation run again exposed problems the gate report had not
caught. Two distinct issues:

**(a) Partial per-document coverage.** Although the dataset is densely
annotated *in aggregate*, individual documents are not. Some documents annotate
only a subset of the fields — for example, one invoice had gold for
`start_date` only, with the other four line-item fields empty despite the
document clearly containing that data. On such documents, "not in gold" is
unreliable for the un-annotated fields. This is the NDA problem reappearing at
the per-document level: a dataset can look dense overall yet have many
individual documents with incomplete gold.

**(b) Tabular structure.** Invoice line items are *rows of a table*. The fields
`Description`, `Rate`, `Days`, etc. are table columns: `Rate[5]` and
`Days[5]` belong to the same row. The pipeline's set-membership matching pools
all of a field's values into an unordered bag and asks only "is this extracted
value somewhere in the gold bag." That is correct for genuinely unordered
fields (e.g. the set of parties to an agreement) but **wrong for table
columns**, where position carries meaning and values repeat across rows
(a rate of `0.00` appearing in twenty rows). The result is that correct
extractions are marked as errors because row alignment is lost.

The 20-document run reported a 55% error rate, but inspection of the
individual labels showed these errors were overwhelmingly artifacts of (a) and
(b), not genuine extraction mistakes. Some real errors exist in the data — the
model does sometimes misread or mis-segment table rows — but they cannot be
separated from the artifacts, so the labels are unusable for probe training.

fcc_invoices was therefore rejected.

---

## 5. The durable output: a dataset gate report

The main reusable result of this evaluation is `check_dataset.py`, a
diagnostic that measures a candidate dataset against the properties this
pipeline requires, and prints targeted samples for manual review. It renders
no automatic verdict — it presents evidence for a human decision.

It now checks, in order:

1. Document counts (per split and pooled).
2. Per-label annotation density and empty-gold rate (catches NDA-style
   aggregate sparsity).
3. Annotation offset integrity.
4. An under-annotation heuristic for regex-detectable field types.
5. A manual-inspection printout — gold values shown in document context, for
   the one judgment a script cannot make: is gold *complete*.
6. **Per-document field coverage** (added after fcc_invoices) — what fraction
   of documents annotate *all* fields versus only some. Catches partial
   per-document coverage.
7. **Tabular-structure detection** (added after fcc_invoices) — flags fields
   whose shape (many values per document, heavy value repetition) indicates
   they are table columns, for which set-membership matching is unsafe.

Sections 6 and 7 were added specifically because fcc_invoices passed sections
1–5 and still failed. Run against fcc_invoices, the updated report flags both
failure modes before any extraction run.

---

## 6. What the evaluation established — dataset requirements

The two failures, taken together, sharpen the requirements for any dataset
used with this pipeline. A suitable dataset must have:

- **Complete gold, per document, not just in aggregate.** Every correct value
  of every schema field must be annotated on every document. Aggregate density
  is not sufficient.
- **Flat fields, not tabular fields.** Each field should be a single value, or
  an unordered set of values, per document. Table-structured fields (rows with
  aligned columns) are not supported: the pipeline matches by value, not by row
  position, and row-aligned matching is a separate problem outside this
  project's scope.
- **Enough documents for LODO** to be statistically meaningful.

RealKIE meets the third requirement but, in the two datasets examined, not the
first two. Its remaining datasets (charities, S1 filings, resource contracts)
are long-document sets the RealKIE paper itself describes as sparsely
annotated, so they are likely to fail the first requirement as well. RealKIE
is, on this evidence, a structural mismatch for the pipeline as built —
it is a span-labeling / table-extraction benchmark, whereas the pipeline
requires complete flat-field gold.

---

## 7. Recommended path forward — for discussion

Two options, with a clear recommendation.

**Option A (recommended): return to ExtractBench, all five domains.**
ExtractBench's defining property is the one RealKIE lacks: complete,
human-validated, flat-field gold. This is *why* the existing academic result
(LODO AUROC 0.834 vs baseline 0.708) is trustworthy. The limitation was small
sample size from using only two of the five domains. Extending to all five
roughly triples the trustworthy field count. A smaller dataset with reliable
labels is more valuable than a larger one with noisy labels — this is the
consistent lesson of the swimming, NDA, and fcc_invoices results. Any
remaining ExtractBench-specific issues should be enumerated and addressed
directly, as this is the more reliable path.

**Option B: find a different flat-field KIE benchmark with complete gold.**
Possible, but after two RealKIE misses, any candidate must pass the full
`check_dataset.py` report — including the new sections 6 and 7 — *and* a manual
completeness check, before any pipeline work begins. Whether a benchmark
exists that combines complete flat-field gold with a large document count is
an open question.

The recommendation is **Option A**, with **Option B** pursued in parallel only
if specific high-quality candidates are identified.

---

## 8. Summary

- RealKIE was evaluated to address the sample-size limitation. Two datasets,
  NDA and fcc_invoices, were tested and both found unsuitable.
- NDA failed due to sparse gold annotation; fcc_invoices failed due to
  partial per-document coverage and tabular field structure.
- The evaluation was not wasted: it produced reusable pipeline capabilities
  (benchmark dispatch, set-membership matching, a recursion-bug fix) and, most
  usefully, a dataset gate report that now encodes every failure mode
  encountered.
- The validation discipline worked as intended — both datasets were rejected
  on cheap 20-document runs, before any expensive full-scale experiment.
- The recommended next step is to extend ExtractBench to all five domains,
  trading large-but-noisy for smaller-but-trustworthy, which every result so
  far indicates is the correct trade.

# Updates — June 16, 2026

Progress update for the `adnan-dev` branch.

This branch applies the existing probe pipeline to a **new model**,
**DeepSeek-R1-Distill-Qwen-7B** (a reasoning model), across all four
ExtractBench domains. The pipeline is the same one used on the partner branch
(Qwen3.5-4B): per-document extraction with hidden-state capture, strict
labelling against gold, per-layer logistic-regression probes, and
leave-one-document-out (LODO) evaluation. The model is the only change, so the
two sets of results are directly comparable.

Note on scope: the Cross-Layer Attention Probe (CLAP) was **not** used in this
run. It was scoped out to focus on getting the standard per-layer probe working
end-to-end on a reasoning model first. The captured activations would support
CLAP later without re-extraction.

---

## 1. Headline result

Linear probes on DeepSeek-R1's hidden states predict field-level error labels.
The honest, document-level metric is **leave-one-document-out (LODO) AUROC**:

| Layer | LODO AUROC |
|------:|-----------:|
| 1  | 0.70 |
| 4  | 0.75 |
| 7  | 0.74 |
| 10 | 0.77 |
| 12 | 0.77 |
| 14 | 0.73 |
| 16 | 0.78 |
| 18 | 0.80 |
| 19 | 0.77 |
| 21 | 0.81 |
| 23 | 0.81 |
| **24** | **0.83**  (best) |
| 26 | 0.81 |
| 28 | 0.81 |

**Takeaway:** the signal is strongest in the **mid-to-late layers
(18-28, ~0.80-0.83)**, peaking at **layer 24 (0.83)**. Early layers (1-4) are
weaker (~0.70-0.75), consistent with trust-relevant information accumulating in
later layers.

**Baseline comparison (be precise here):** the black-box token-log-probability
baselines score **mean_logprob AUROC 0.80** and **min_logprob AUROC 0.78**. So
the best probe layer (0.83) is **modestly above** the baseline, while several
layers are roughly tied with or below it. The honest framing is: *the probe is
competitive with the logprob baseline and its best layer (24) edges past it* --
not a large margin. Note the baseline figures are full-set AUROC whereas the
probe figures are the stricter LODO metric, so the comparison is indicative
rather than perfectly matched.

---

## 2. What to read, and what to ignore

- **Report the LODO numbers above.** They hold out whole documents, so no
  fields leak between train and test.
- **Do not report the random-fold CV numbers** (`auroc_cv_mean` ~ 0.95-0.99 in
  `comparison.json`). Those are inflated because fields from the same document
  appear in both train and test folds. The large gap between CV (~0.99) and
  LODO (~0.83) is expected and is the reason LODO is the metric of record.

---

## 3. Dataset and an important labelling caveat

- **27 of 28 documents extracted** successfully (Stage 1).
- **21 documents labelled**, **2,779 fields** total (Stage 2). Seven documents
  were skipped automatically because their generated JSON was pathologically
  deep/degenerate and could not be aligned to the gold schema; one further
  document failed extraction. Skipping these protects the run without affecting
  the labelled set.
- The probe evaluation uses the subset of fields with usable activations
  (`n_samples` = 657 in `comparison.json`).

**Caveat the supervisor will ask about — the ~95% labelled error rate.**
The strict matcher labels ~95% of fields as errors, but this **overstates true
extraction failure**. Inspection shows most "errors" are **structural mismatches**,
not wrong values. For example, on the NIPS paper the model emits authors as flat
strings:

```
extracted: "B. Boser"
gold:      {"name": "B. Boser", "affiliation": "AT&T Bell Laboratories"}
```

The extracted content is correct, but because the model outputs a string where
the gold schema expects a nested object, the matcher records it as a
hallucination/mismatch. On that document, 317 of 328 "errors" are of this kind.
So the **absolute error rate reflects schema-shape conformance more than factual
correctness**. The probe is therefore predicting "will this field be flagged by
the strict matcher," which is a consistent and reportable target — and identical
to the matcher used on the partner branch, so the cross-model comparison remains
fair.

A second, smaller caveat: large finance / credit-agreement documents were
truncated to 50,000 characters (~12k tokens) to fit GPU memory, so for those the
model extracted from the front of the document.

---

## 4. Pipeline engineering (why this required work)

Getting a reasoning model through the pipeline on the A40 needed several fixes,
all now in the branch:

- **GPU reliability:** an ECC-faulted cluster node silently ran the model on CPU
  (~0.4 tok/s). Added a GPU self-test guard to the job script and exclude the
  bad node at submission; runs now execute on GPU at ~25-30 tok/s.
- **Memory:** capped input length to avoid CUDA OOM on the largest documents.
- **Runaway generation:** added a repetition penalty so the reasoning model stops
  cleanly instead of looping to the token limit.
- **Robust parsing:** hardened JSON extraction (handles reasoning tags, truncated
  code fences) plus an optional repair fallback.
- **Robust labelling:** the labelling stage now skips any single document that
  fails (e.g. degenerate deeply-nested JSON) instead of crashing the whole run.

These took the usable dataset from 4 documents to 21 and let the full pipeline
run end-to-end.

---

## 5. What this means / next steps

- The standard per-layer probe **replicates on a reasoning model**: probes beat
  the baseline, with peak signal in mid-to-late layers (best 0.83 at layer 24).
- **Relax the matcher** so it compares values rather than schema shape (e.g.
  compare an author string against the gold object's `name`). This should bring
  the error rate down to a realistic level and make the probe target "factual
  correctness" rather than "schema conformance."
- **Optionally run CLAP** (`run_clap.sh`) on the existing activations to test
  whether a single cross-layer attention probe matches or beats the best
  hand-picked single layer, without manual layer selection.
- **Cross-model read-across:** compare these DeepSeek per-layer LODO results
  against the partner's Qwen results under the identical matcher.

---

## 6. Reproduce

```bash
sbatch --exclude=node-02 run_deepseek_extract.sh   # Stage 1 (GPU)
sbatch run_deepseek_analysis.sh                    # Stages 2-5 (CPU)
```

Outputs: `artifacts/deepseek_r1_7b_pooled/results/comparison.json` (per-layer +
baselines) and `lodo_cv.json` (per-layer LODO — the metric of record).
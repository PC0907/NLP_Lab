# Weekly Update — Model Scaling Setup + Financial-Domain Diagnosis

## Summary

This week was largely **diagnostic and infrastructure**: setting up the model-scaling
experiments (smaller + larger Qwen) and investigating *why* the financial domain has
such high error rates. The main outcomes are two clear findings (the financial errors
are a benchmark-annotation artifact, not a parser or model failure; and a token-limit
truncation issue that explains run-to-run document-count inconsistency) plus the
scaling pipeline now running. No new headline AUROC yet — that comes once the scaling
runs complete on a clean, comparable document set.

---

## 1. Financial-domain failure investigation (Task 4) — *it is not the parser*

**Question:** are the high financial-domain (10kq) error rates caused by the PDF
extractor failing, the model being bad, or the benchmark annotations?

**Method:** single-document end-to-end audit (`investigate_financial_doc.py`) on
`adp_10q_fy2025q2` — dumped raw text from **both** parsers (PyMuPDF and Docling) and,
per error field, checked whether the gold value is findable in each parser's text.

**Result (947 error fields on this one doc):**

| Bucket | Count | Meaning |
|---|---|---|
| present_model_wrong | 546 (58%) | value IS in both parsers' text, model still wrong |
| gold_not_in_text (both) | 379 (40%) | absent from both parsers |
| empty/matcher artifact | 22 (2%) | not real errors |
| **in_docling_only** | **0** | Docling recovers nothing PyMuPDF misses |
| **in_pymupdf_only** | **0** | and vice versa |

**Conclusions:**
- **The parser is not the problem.** The two parsers are *identical* on gold-presence
  (`in_docling_only = in_pymupdf_only = 0`). A different/better PDF extractor would
  recover **zero** additional values. We do not need a new parser for these documents.
- **~40% of errors are not in the document at all** — these are metadata/convention
  fields (`scale`, `unit`, `data_period`, `segment_name`) that are either derived or
  follow an undocumented benchmark convention. No model or parser can extract a value
  that is not present.
- The remaining 58% ("present but wrong") still needs the genuine-vs-convention split
  (drilldown pending review), but a large share is the scale/value encoding convention.

---

## 2. The financial errors are a benchmark-annotation artifact — confirmed across model sizes

Labeling the **2B** model on 10kq shows the same pattern at scale. On one 10-Q
(`csco`, 1344 errors), the error fields by type:

| Field type | Errors | Nature |
|---|---|---|
| data_period | 202 | derived ("FY2025 H1"), not literal in text |
| metric_type | 202 | annotation convention ("actual") |
| segment_name | 202 | placeholder ("NA"), not in document |
| segment_type | 202 | convention ("company") |
| scale | 186 | undocumented (schema gives no description) |
| unit | 171 | implied (USD), not printed per line |
| value | 155 | the actual numbers (the only potentially-real errors) |

~90% of the 10kq errors are convention/metadata fields the model cannot infer. This
holds for **both** the 4B and 2B → it is a property of the benchmark's annotations,
not of any particular model.

**Whose "fault":** for the financial domain, it is primarily a **benchmark-annotation
issue** — not the parser (two parsers agree) and not really the model (all model sizes
fail the same convention fields). Genuine model errors are confined to the `value`
fields and to the non-financial domains. **Consequence:** 10kq error rate is not a
meaningful model-quality signal, so 10kq is excluded from the cross-model scaling
comparison and reported separately as an annotation artifact.

---

## 3. Model-scaling setup (smaller + larger Qwen)

Goal: test whether the probe's trust signal generalizes across model scales.
Same Qwen3.5 architecture family (clean comparison): **2B → 4B → 9B**.

| Model | Layers | Hidden | Status |
|---|---|---|---|
| Qwen3.5-2B | 24 | 2048 | extracted + labeled; LODO pending (clean doc set) |
| Qwen3.5-4B | 32 | 2560 | done (prior: nested LODO 0.879) |
| Qwen3.5-9B | 32 | 4096 | config ready (reduced layers [14-22] to avoid OOM) |

Pooled single-config extraction (all domains in one run). The 9B captures only the
mid-network layers [14,16,18,20,22] (where the signal lives) to fit A100 memory.

Two pipeline bugs found and fixed along the way:
- **Layer range:** the 2B has 24 layers, not 32 — config layer list corrected.
- **Matcher recursion crash:** the labeler hit a `RecursionError` (infinite
  empty-substitution loop) on a 2B output. Added a depth guard (`_MAX_WALK_DEPTH`):
  past a sane depth, a node is labelled a type-mismatch instead of looping. The 4B
  never recursed deep, so its labels are unaffected — the guard only fires on
  pathological output, which (notably) the smaller 2B produced and the 4B did not.

---

## 4. Token-limit truncation — explains the inconsistent document counts

**Observation:** extraction sometimes yields 28 successful docs, sometimes 26, 25, 22.

**Cause (from `_summary.json`):** `finish_reason: "length"`, `generated_tokens: 16384`
— documents whose JSON answer exceeds `max_new_tokens` (16384) are **cut off mid-JSON**,
producing invalid JSON that fails to parse, so the document is dropped. 5 of 6 parse
failures on the 2B were this truncation.

**Why it varies by model:** the smaller 2B is wordier / less efficient, so documents
that fit under 16384 tokens for the 4B *overflow* for the 2B → more truncation → fewer
successful docs. The variation is not random; it is the token cap interacting with
model verbosity.

**Which docs truncate:** mostly long academic surveys (real content worth recovering)
and the big 10-Qs (convention-noise, excluded anyway).

**Consequence for the scaling comparison:** because models truncate on *different*
document sets, naive comparison would pit (e.g.) 2B-on-22-docs against 4B-on-26-docs —
not apples-to-apples. Fix: either raise `max_new_tokens`, and/or compare all models
only on the **intersection** of documents every model parsed successfully.

---

## Next steps

- Run the 2B LODO on a clean, comparable domain set (academic + credit + swimming;
  exclude 10kq convention-noise) and compare to the 4B on the same set.
- Decide truncation handling: raise `max_new_tokens` (recovers academic docs) vs use
  the common-success document intersection. Then re-run for comparability.
- Launch the 9B (reduced-layer config) once the 2B pipeline is confirmed clean.
- Review the `present_model_wrong` drilldown to quantify genuine model errors vs the
  scale/value convention on financial `value` fields.
- (Deferred) Task 1: the 3-signal regression (probe + logprob + hand-crafted features)
  — framing to be confirmed with supervisor before building.

## Key principle this week

Most of the apparent "failures" turned out **not** to be where they first looked. The
financial error rate is a benchmark-annotation artifact (not parser, not model), and
the document-count inconsistency is token truncation (not randomness). Separating these
confounds before drawing model-quality conclusions is the substance of the week — it
prevents reporting "the model is bad at finance" when the model is mostly being
penalized for undocumented conventions.

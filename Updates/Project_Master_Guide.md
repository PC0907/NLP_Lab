# Project Master Guide — Understand Everything (with examples)
## Probe‑Based Trust Signals for Structured Information Extraction (DeepSeek‑R1 / Reasoning‑Trace Track)

**Purpose:** a single document to master the project end‑to‑end. Every concept is
explained plainly and then shown on **one running example** so nothing stays
abstract. The last section is a large **expected‑questions bank** with answers.

**How to read it:** §1 is the 2‑minute version. §2–§3 build intuition. §4 is the
SOB dataset in depth (worked example). §5–§11 walk the whole pipeline on that same
example. §12 is results. §13 is the "why it means what it means." §14 limitations.
§15 the Q&A bank.

---

## 1. The 2‑minute version

- **Task:** an LLM reads text and outputs a **structured JSON** record following a
  schema (this is *structured information extraction*).
- **Problem:** some fields come out wrong (wrong number, invented name), but the
  JSON still *looks* fine. The model gives no reliable per‑field confidence.
- **Our solution:** train a tiny **linear probe** (a logistic regression) on the
  model's **internal activations** for each field, to predict *is this field
  wrong?* — no gold answer needed at deployment. This is a **trust signal**.
- **Use of the trust signal:** **selective regeneration** — only re‑ask the model
  for the fields the probe flags as risky, instead of regenerating everything.
- **Our specific angle:** we use a **reasoning model** (DeepSeek‑R1) that writes an
  explicit chain of thought `<think>…</think>` before answering, on a dataset
  (**SOB**) where the answer genuinely requires reasoning. New question: *does the
  reasoning trace carry extra error signal we can probe?*
- **Answer we found:** pooling the reasoning at the **document level** does nothing
  (mechanical reason below); but **localizing the reasoning to each field** gives a
  small **statistically significant** improvement. That contrast is the novelty.

---

## 2. The core idea, by analogy

Imagine a student (the LLM) filling in a form from a textbook page. Sometimes they
write a confident but wrong answer. You can't re‑grade every answer against the
textbook every time (that's expensive). So instead you watch the student's **brain
activity** (the hidden activations) and learn the pattern that lights up when they
are about to be wrong. That learned detector is the **probe**. Our extra twist: the
student also **thinks out loud** first (the reasoning trace), and we check whether
listening to *how they reasoned about a specific answer* helps predict if that
answer is wrong.

Key vocabulary you must be fluent in (each expanded later with an example):
- **Activation / hidden state** — the internal vector the model computes at a token
  (here 3584 numbers per token, per layer).
- **Probe** — a simple linear classifier trained on those vectors.
- **Field** — one leaf value in the output JSON (e.g. `instrument_name`).
- **Label `is_error`** — 1 if the field's value disagrees with gold, else 0.
- **AUROC** — how well a score ranks wrong fields above correct ones (0.5 random,
  1.0 perfect).
- **LODO** — leave‑one‑document‑out; the honest generalization test.
- **Reasoning trace** — the `<think>…</think>` text the model writes before its JSON.

---

## 3. The running example (used everywhere below)

We will follow ONE SOB record through the entire system.

**Question:** *"what percussion instrument called zils was used?"*

**Context (passage):**
> "Xylophone: a wooden instrument played with mallets. Tambourine: a frame drum
> fitted with pairs of small metal jingles called zils…"

**Schema (what JSON to produce):**
```json
{
  "type": "object",
  "properties": {
    "instrument_name":   {"type": "string"},
    "instrument_family": {"type": "string"},
    "main_components":   {"type": "array", "items": {"type": "string"}}
  }
}
```

**Gold answer (human‑verified):**
```json
{
  "instrument_name": "Tambourine",
  "instrument_family": "percussion",
  "main_components": ["frame", "zils"]
}
```

This is **multi‑hop**: to answer you must connect "zils" (in the Tambourine
sentence) → identify the Tambourine → know its family is percussion → list its
components. That's exactly why a reasoning model is interesting here.

---

## 4. The SOB dataset — in depth

**Name:** SOB = *Structured Output Benchmark* (`interfaze-ai/sob`), the
**text / multi‑hop** subset, derived from **HotpotQA** (Wikipedia passages).

**What one record contains:**
| Field in the record | Meaning | Example |
|---|---|---|
| `question` | a multi‑hop question | "what percussion instrument called zils was used?" |
| `context` | the source passage(s) | the Xylophone/Tambourine text |
| `json_schema` | the target schema (per record!) | the 3‑property object above |
| `ground_truth` | human‑verified gold JSON | `{"instrument_name":"Tambourine", …}` |
| `source_dataset` | provenance | `hotpotqa` |
| `schema_complexity` | difficulty tag | `hard` |

**Four properties that make SOB the right choice for us:**
1. **Reasoning is required.** Unlike reading a value off an invoice, you must
   combine facts across sentences. So the `<think>` trace actually contains useful
   reasoning to probe. On a pure‑lookup dataset the trace would be empty of signal.
2. **Per‑record schema.** The schema *changes every record* (one has instrument
   fields, the next has composer fields, etc.). This is harder than a fixed
   per‑domain schema and closer to real, heterogeneous extraction.
3. **Clean text.** No PDFs → no OCR/layout noise polluting the activations. Any
   error the probe detects is a genuine reasoning/extraction error.
4. **Complete, human‑verified gold.** We can label every field reliably.

**A subtle data‑engineering point (worth knowing — the guide may ask):** in the raw
parquet, `json_schema` and `ground_truth` are stored as **JSON strings**, not
nested objects (because the schema shape varies per record and can't be a fixed
Arrow column). Our loader parses those strings back into dicts (`_as_dict`). If you
forget this, every record's schema/gold reads as an unusable string — this actually
bit us once and is now handled + unit‑tested.

**How a record becomes a pipeline `Document`:** the question is **prepended** to the
context so the standard extraction prompt sees what to answer:
```
Question: what percussion instrument called zils was used?

Context:
Xylophone: a wooden instrument… Tambourine: a frame drum … called zils…
```
The record's own `json_schema` and `ground_truth` are attached to that Document.

**This run's scale:** 300 documents (test split). 299 labeled (1 dropped for a
malformed‑JSON generation). 1,842 fields total. ~919‑token contexts on average.

**Contrast with Ali's ExtractBench (know this cold):**
| | SOB (this track) | ExtractBench (Ali) |
|---|---|---|
| Input | question + Wikipedia passage (text) | PDF documents |
| Schema | **per record** | per domain |
| What it demands | **multi‑hop reasoning** | mostly field lookup |
| Noise | clean text | PDF/OCR artifacts |
| Why chosen | to make the reasoning trace meaningful | broad flat‑field coverage |

---

## 5. The model — DeepSeek‑R1 and its reasoning trace

**DeepSeek‑R1‑Distill‑Qwen‑7B:** a 7‑billion‑parameter **reasoning** model, 28
transformer layers, hidden dimension 3584. Run in `bfloat16`, **greedy** decoding
(temperature 0 → deterministic, so reruns reproduce), up to 4096 new tokens, on one
A100 GPU (Bender cluster).

**What "reasoning model" means here:** before emitting the answer it writes a chain
of thought between `<think>` and `</think>`. For our example it might produce:

```
<think>
The question asks for a percussion instrument associated with "zils".
The context says the Tambourine is a frame drum fitted with small metal
jingles called zils. Zils are the jingles, so the instrument is the
Tambourine. Its family is percussion. Its main components are the frame
and the zils.
</think>
{"instrument_name": "Tambourine",
 "instrument_family": "percussion",
 "main_components": ["frame", "zils"]}
```

**What we capture during generation (Stage 1):**
- The generated JSON (the answer).
- Per‑token **log‑probabilities** (used for the baselines).
- For each output **field**, the model's **hidden‑state vector** at 14 layers, taken
  at the **last token of that field's value** (the "answer token"). E.g. for
  `instrument_name = "Tambourine"`, the vector at the last token of "Tambourine".
- **Reasoning‑trace representations**: (a) pooled vectors — the mean over all
  `<think>` tokens and the `</think>` summary position (used in the *document‑level*
  experiment); and (b) — added this cycle — the **per‑token** hidden states of the
  reasoning tokens for 4 layers, plus their surface strings (used for
  *field‑localized* attribution).

**Why the last token of the value?** By the time the model finishes writing a
field's value, that position's hidden state summarizes the model's "state of mind"
about that value — the most informative single position to probe.

---

## 6. Pipeline overview

```
 SOB record ─▶ [1] Extraction ─▶ [2] Labeling ─▶ [3] Probe training ─▶ [4] Eval vs baselines
                    │                  │                                      │
        hidden states + reasoning   is_error per field              [5/6] LODO (honest) + doc-level fusion
        (answer tokens + trace)                                             │
                                                              [7] Field-localized reasoning attribution
```
Stages 1–4 give the core trust signal. Stages 5/6 are the honest evaluation +
the document‑level reasoning test (null). Stage 7 is the field‑localized reasoning
attribution (the significant result).

---

## 7. Stage 2 — Labeling (turning outputs into `is_error`), with the structure‑aware fix

For each field we compare the model's value to gold and set `is_error ∈ {0,1}`.

**The problem we hit first.** Naïve **exact string match** reported a **~95% error
rate** — obviously an artifact. Two causes, both visible on our example:
- **Flat vs nested.** Sometimes gold stores a value as a nested object while the
  model returns the flat leaf (or vice versa). Exact‑match calls that "wrong" even
  though the *content* agrees.
- **Formatting.** `"tambourine"` vs `"Tambourine"`, `"percussion"` vs
  `"Percussion"`, `["zils","frame"]` vs `["frame","zils"]` (order), numbers like
  `5` vs `5.0`, dates in different formats.

**The fix — three labeling modes (we record all three for transparency):**
- **strict** — exact string match (harshest, original).
- **auto** — type‑aware leaf comparison: numeric tolerance, date parsing,
  case‑insensitive text, set‑equality for lists, fuzzy match.
- **structure_aware** — everything in `auto`, **plus** match a flat predicted value
  against the *leaf values* of a gold object (fixes flat‑vs‑nested).

**On our example, structure_aware labeling:**
| Field | Model value | Gold | is_error |
|---|---|---|---|
| instrument_name | `"tambourine"` | `"Tambourine"` | 0 (case‑insensitive match) |
| instrument_family | `"Percussion"` | `"percussion"` | 0 |
| main_components | `["zils","frame"]` | `["frame","zils"]` | 0 (set‑equal) |

If instead the model had said `instrument_name = "Xylophone"`, that field would be
`is_error = 1` — a genuine error the probe should catch.

**Result of the fix (300‑doc run):** error rate strict **44.4%** → auto **41.6%** →
structure_aware **41.4%**. The residual ~41% is a **genuine, balanced** error rate —
ideal for training a probe (not trivially easy, not degenerate). `structure_aware`
is our primary mode.

**Why 41% and not lower?** Because SOB is genuinely hard multi‑hop, and DeepSeek‑R1
is only 7B — it makes real reasoning mistakes on many fields. That's good for us: a
balanced dataset of correct and wrong fields to learn from.

---

## 8. Stage 3 — The probe (the trust signal itself)

For one layer, take each field's activation vector `h ∈ ℝ³⁵⁸⁴` and train a
**logistic regression** `P(error) = σ(w·h + b)`, class‑balanced, on the `is_error`
labels. One probe per layer (we try 14 layers).

**Why *linear*?** If a *linear* classifier can read "is this wrong?" off the
activation, then that information is **linearly present** in the model's
representation — a clean, strong, interpretable scientific claim. A big non‑linear
probe could "manufacture" signal and overfit; a linear probe is a conservative,
credible test. (Ali's future work lists MLP/mass‑mean/CLAP probes as extensions.)

**On our example:** the vector at the last token of "Tambourine" goes in; the probe
outputs, say, `P(error) = 0.08` (confident it's right). For a hallucinated field it
would output a high `P(error)`.

---

## 9. Stage 3/4 — Baselines (what we must beat)

The free competitor is the model's **own confidence**, from token log‑probabilities:
- **mean_logprob** — average token log‑prob over the field's tokens.
- **min_logprob** — the least confident token in the field.

Intuition: if the model was "unsure" (low probability) while writing a value, maybe
it's wrong. It's cheap but crude — a fluent model can be confidently wrong. A trust
signal is only interesting if it **beats** these. (Ali also compares P(True) and
self‑consistency; our track focuses on the log‑prob baselines.)

---

## 10. Evaluation — how we measure success (the part guides love to probe)

**AUROC (Area Under the ROC Curve).** Rank all fields by the score (probe or
baseline). AUROC = probability that a randomly chosen *wrong* field is ranked above
a randomly chosen *correct* one. 0.5 = coin flip, 1.0 = perfect separation. We use
AUROC because the classes are imbalanced and we care about *ranking* risky fields,
not a hard yes/no.

**Two evaluation protocols:**
- **5‑fold Cross‑Validation (CV):** shuffle all fields, 5 folds, standard check.
- **Leave‑One‑Document‑Out (LODO):** the **honest** test. Hold out one *document*,
  train on all the others' fields, predict the held‑out doc's fields. Repeat for
  every document. Why stricter than CV: fields from the same document are
  correlated (same context, same reasoning), so random CV can "peek" at a document
  it's also testing on. LODO forces generalization to **entirely unseen documents**
  — the realistic deployment condition.

**Two LODO metrics (you MUST be able to explain the difference):**
- **per‑doc AUROC** — compute AUROC *within* each held‑out document, then average.
  Answers: *"within one document, can we rank which field is wrong?"* This is what
  matters for choosing which field to regenerate in a given document.
- **pooled‑OOF AUROC** — collect every out‑of‑fold prediction across all documents
  into one pool and compute a single AUROC. Answers: *"across the whole dataset, can
  we globally rank risky fields?"* This is what a single global regeneration
  threshold consumes.

**Why they can disagree — the crux of our whole result:** a feature that is
**constant within a document** (like a document‑level reasoning vector) cannot
change the *within‑document* ranking (per‑doc), because it adds the same number to
every field's score in that document. It could only ever help the *global* ranking
(pooled). This single fact explains why document‑level fusion was null and why
field‑level attribution works — see §13.

**"n = 199 valid folds" explained:** a per‑doc AUROC only exists if the held‑out
document has **both** a correct and a wrong field (you can't rank one class). Of
~295 documents, 199 satisfy that; the rest are all‑correct or all‑wrong and are
skipped for the per‑doc average (they still contribute to pooled‑OOF).

---

## 11. Stages 6 & 7 — Bringing in the reasoning trace

**Stage 6 — document‑level fusion (the negative result).** Concatenate the pooled
`<think>` vector (mean / `</think>`) onto the answer‑token vector, retrain the
probe. Result: **no real gain** (Δ ≈ +0.006 to +0.013, not significant). Reason
(from §10): the pooled reasoning vector is **the same for every field** in a
document, so it cannot re‑rank fields within the document.

**Stage 7 — field‑localized attribution (the significant result).** For each field,
**find where its value is mentioned inside the `<think>` trace** and pool the hidden
states of exactly those reasoning tokens. Now the reasoning feature **varies per
field**, so it *can* help within‑document ranking.

**On our example:** the trace mentions "Tambourine", "percussion", "frame", "zils".
- For field `instrument_name = "Tambourine"`: we locate the token(s) "Tambourine" in
  the trace and mean‑pool their hidden states → that field's **attribution vector**.
- For `instrument_family = "percussion"`: pool the reasoning tokens around
  "percussion".
- Each field gets a *different* attribution vector → legitimate within‑doc signal.

**Two signals per field produced by the attribution module:**
1. **attr_vec** — the pooled hidden states of the value‑mention tokens (the main,
   powerful signal). If the value is never mentioned, it's a zero vector.
2. **interpretable scalars** — e.g. `mentioned` (was the value reasoned about at
   all?), `mention_count`, where in the trace it first/last appears. The headline
   scalar is **"value present in the JSON but absent from the reasoning trace"** — a
   natural **hallucination red‑flag**. (86.8% of values are mentioned; the 13.2%
   that aren't are prime suspects.)

**Matching details (so you can defend it):** matching is done on lowercased,
whitespace‑normalized text; it is **word‑bounded** so "art" doesn't match inside
"Mozart"; if the full value isn't found it falls back to the **longest word**
(e.g. full "lake tahoe, california" absent but "california" present → partial
match). Booleans/None aren't searchable; integers render without a trailing ".0".

**The five variants Stage 7 compares (under LODO, paired test vs `answer`):**
`answer` (baseline), `fused_attr` (answer + attr_vec), `fused_scalars` (answer +
scalars), `fused_both` (answer + attr_vec + scalars), `scalars_only`.

---

## 12. Results (all the numbers, in one place)

**(a) The probe works — beats baselines (5‑fold CV):**
| Signal | AUROC |
|---|---|
| Baseline mean_logprob | 0.771 |
| Baseline min_logprob | 0.768 |
| Probe layer 16 | 0.822 |
| **Probe layer 19 (best)** | **0.850** |
| Probe layer 23 | 0.834 |
| Probe layer 26 | 0.843 |

Signal rises with depth, peaks mid‑late (≈18–26), then plateaus. Beats baselines by
**~+0.08**.

**(b) Honest LODO (answer‑only):** per‑doc **0.81**, pooled‑OOF **0.84** (layer 19).
Holds on unseen documents.

**(c) Document‑level reasoning fusion (Stage 6): NULL.** Δ +0.006 to +0.013, not
significant. Explained mechanically.

**(d) Field‑localized attribution (Stage 7, layer 19), paired vs answer:**
| Variant | per‑doc | Δ | p | pooled‑OOF |
|---|---|---|---|---|
| answer | 0.8118 | — | — | 0.8422 |
| **fused_attr** | **0.8325** | **+0.0207** | 0.059 | 0.8561 |
| **fused_both** | **0.8345** | **+0.0227** | **0.044 ✅** | 0.8553 |
| fused_scalars | 0.8108 | −0.001 | 0.59 | 0.8418 |
| scalars_only | 0.7338 | −0.078 | 0.002 | 0.7167 |

Consistent across all 4 layers (16/19/23/26), both metrics (all deltas positive,
+0.012 to +0.031 per‑doc). 86.8% of values mentioned in the trace.

---

## 13. Why the results mean what they mean (the intellectual core)

The whole track hinges on **one mechanical fact**:

> A feature that is **constant across a document's fields** adds the same value to
> every field's probe score in that document. Adding a constant to every score does
> **not change their relative order** → it cannot change the **within‑document**
> (per‑doc) ranking.

- **Document‑level reasoning** is exactly such a constant‑within‑doc feature →
  provably ~null on per‑doc, and empirically flat on pooled too. **(Stage 6)**
- **Field‑localized reasoning** produces a **different** vector per field → it *can*
  re‑rank fields within a document, and it does: significant, consistent gain.
  **(Stage 7)**

So our contribution is not just "reasoning helps" — it is the sharper, more
defensible claim: **reasoning helps for structured‑extraction error detection only
when it is attributed to the specific field; pooled at the document level it is
useless, and here is the exact reason.** That mechanistic story is what makes it
publishable rather than a lucky number.

---

## 14. Limitations (be the first to say them)

1. **Modest effect size.** ~+0.02 AUROC. `fused_both` is significant (p = 0.044);
   `fused_attr` is borderline (p = 0.059).
2. **Multiple comparisons.** Five variants were compared; a strict correction would
   soften the single p‑value. Our current defense is the **consistency across four
   independent layers and both metrics**; the real fix is more data (next step).
3. **Scale.** 300 documents, 199 valid per‑doc folds. Scaling to ~1,000 is the
   planned power boost.
4. **String‑match attribution.** We locate a value in the trace by (normalized,
   word‑bounded) string match with a longest‑word fallback. It can miss paraphrased
   mentions; attention‑based attribution is a future refinement.
5. **One model, one dataset so far.** Cross‑dataset (ExtractBench) and other model
   families are future work.

---

## 15. Expected questions & answers (study this section)

### Conceptual / motivation
**Q. In one sentence, what is this project?**
A. We train a linear probe on an LLM's internal activations to flag which fields of
its structured‑extraction output are wrong, and we study whether the model's
reasoning trace adds signal — finding it helps only when localized to each field.

**Q. Why not just use the model's confidence (log‑prob)?**
A. Log‑prob is cheap but crude — a fluent model is often confidently wrong. Our
probe reads the hidden state, not just the output probability, and beats the
log‑prob baselines by ~0.08 AUROC.

**Q. Why a *linear* probe and not a neural network?**
A. A linear probe tests whether the error information is *linearly present* in the
representation — a conservative, interpretable claim that avoids overfitting.
Bigger probes are listed as future work.

**Q. What is a "trust signal"?**
A. A per‑field score for "is this field correct?" produced without access to the
gold answer, usable at deployment to decide what to regenerate.

### Dataset
**Q. What is SOB and why did you pick it?**
A. Structured Output Benchmark, text/multi‑hop subset from HotpotQA: a multi‑hop
question + passage → JSON with a per‑record schema and human‑verified gold. We chose
it because answering requires genuine reasoning (so the `<think>` trace is
meaningful), the text is clean (no PDF noise), the schema varies per record (harder,
more realistic), and the gold is complete.

**Q. Give a concrete SOB example.**
A. Q: "what percussion instrument called zils was used?"; passage describes a
Tambourine as a frame drum with zils; schema has `instrument_name`,
`instrument_family`, `main_components`; gold is `{"Tambourine","percussion",
["frame","zils"]}`. You must connect "zils" → Tambourine → percussion → components.

**Q. How is SOB different from Ali's ExtractBench?**
A. ExtractBench is PDF documents with per‑domain schemas and mostly lookup;
SOB is clean text with per‑record schemas and multi‑hop reasoning. Same task
(text → JSON), different regime — which is why it's the right place to test
reasoning‑trace probing.

**Q. Any data‑engineering gotcha in SOB?**
A. Yes — `json_schema` and `ground_truth` are stored as JSON *strings* in the
parquet (schemas vary per record, so no fixed columns). Our loader parses them back
to dicts; missing this makes every schema/gold unusable.

### Model
**Q. Why DeepSeek‑R1 specifically?**
A. It's a reasoning model that emits an explicit `<think>` chain, giving us a new
place to look for error signal that Ali's non‑reasoning Qwen models don't have.

**Q. What exactly do you capture from the model?**
A. The JSON, per‑token log‑probs (baselines), each field's hidden state at 14 layers
(last token of the value), and reasoning‑trace representations — pooled vectors and
(new) per‑token reasoning states for 4 layers plus their token strings.

**Q. Why the last token of the value?**
A. By the time the value is fully written, that position's hidden state best
summarizes the model's state about that value.

**Q. Why greedy decoding / temperature 0?**
A. Determinism — reruns reproduce the same JSON and labels exactly, so the
re‑extraction for attribution matches the first run field‑for‑field.

### Labeling
**Q. Why was the error rate 95% at first, and how did you fix it?**
A. Naïve exact match counted flat‑vs‑nested and formatting differences as errors.
Our structure‑aware, type‑aware matcher (numeric tolerance, dates,
case‑insensitivity, set‑equality, flat‑against‑leaf) brought it to a genuine 41.4%.

**Q. Is 41% too high — is the model that bad?**
A. It's a real rate for a 7B model on hard multi‑hop extraction, and it's *balanced*,
which is exactly what you want for training a probe. It is not an artifact.

**Q. Could the matcher be too lenient now?**
A. We keep all three modes (strict/auto/structure_aware) recorded, so any leniency
is transparent; strict is only 3 points higher (44.4%), so the fix is modest and
principled, not a whitewash.

### Method / evaluation
**Q. What is LODO and why is it better than cross‑validation here?**
A. Leave‑one‑document‑out: train on all documents but one, test the held‑out one.
Fields within a document are correlated, so random CV can peek; LODO forces
generalization to unseen documents — the deployment condition.

**Q. per‑doc vs pooled‑OOF AUROC — what's the difference?**
A. per‑doc = rank fields *within* each document, then average (which field in this
doc to regenerate). pooled‑OOF = one global ranking across all documents (what a
single global threshold uses). They differ because a document‑constant feature can
move the global ranking but not the within‑document one.

**Q. What does "n = 199 folds" mean?**
A. A per‑doc AUROC needs the held‑out document to contain both a correct and a wrong
field; 199 of ~295 documents qualify, the rest are skipped for the per‑doc average.

### Reasoning trace (the novelty)
**Q. Why did document‑level reasoning fusion fail?**
A. The pooled reasoning vector is identical for every field in a document, so it
adds the same constant to each field's score and cannot change within‑document
ranking. It's a null *by construction*, not a bug.

**Q. What is field‑localized attribution and why does it work?**
A. For each field, we find where its value is mentioned in the `<think>` trace and
pool exactly those reasoning tokens. That vector varies per field, so it *can*
re‑rank fields within a document — and it gives a significant gain.

**Q. How do you attribute a value to reasoning tokens?**
A. Normalized, word‑bounded string match of the value in the trace text (with a
longest‑word partial fallback), then mean‑pool the hidden states of the matched
tokens. Un‑mentioned values get a zero vector and a hallucination flag.

**Q. Is the improvement significant?**
A. `fused_both` reaches p = 0.044 (paired Wilcoxon, per‑doc, layer 19);
`fused_attr` is borderline at p = 0.059. More importantly the effect is positive at
all four layers and on both metrics — consistent, not a single lucky number.

**Q. Isn't +0.02 AUROC small?**
A. Yes, it's modest. The value is (1) the mechanistic insight — pooled fails,
localized works, with a clear reason — and (2) consistency. Scaling to ~1,000 docs
is the immediate step to harden significance.

**Q. Do the hand‑crafted scalars help?**
A. Barely — `fused_scalars` ≈ answer and `scalars_only` is weak (0.73). The lift
comes from the attribution *vector* (the reasoning hidden states), which is the more
interesting result: the model's reasoning representation carries the signal.

**Q. Could the attribution vector be leaking the answer?**
A. No — it's hidden states from the reasoning tokens (written *before* the JSON),
standardized with train‑only statistics, evaluated under LODO. It reflects how the
model reasoned about the value, not the value itself.

### Stats / rigor
**Q. Why a paired test?**
A. We compare each variant to `answer` on the *same held‑out documents*, so the
per‑document differences are paired; a paired Wilcoxon has far more power than
comparing two noisy means with large between‑document variance.

**Q. What about multiple comparisons?**
A. Fair point — five variants. A strict Bonferroni would soften the single p‑value.
Our current defense is cross‑layer/both‑metric consistency; the fix is more data.

### Results / positioning
**Q. What's your single headline number?**
A. The answer‑probe: LODO **0.81 per‑doc / 0.84 pooled**, beating log‑prob (0.77);
and the novelty: field‑localized reasoning adds a significant **+0.02** (p = 0.044)
where document‑level pooling was null.

**Q. How does this fit with Ali's work?**
A. One trust‑signal method validated across model families (Qwen → DeepSeek‑R1),
datasets (ExtractBench → SOB), and signal sources (answer token → reasoning trace).
His future‑work slide literally lists SOB and other models — this is that.

### Limitations / next
**Q. Biggest weakness?**
A. Modest effect and borderline single p‑value at current scale; fixed by more
documents. Also single‑model/single‑dataset so far.

**Q. What's next?**
A. (1) Scale to ~1,000 SOB docs to harden significance; (2) plug the probe into
selective regeneration for the practical cost‑quality payoff; (3) replicate the
attribution effect on ExtractBench.

**Q. What would falsify your claim?**
A. If, at larger scale, `fused_attr`/`fused_both` deltas shrink to zero or lose
significance and the cross‑layer consistency disappears, the field‑localized effect
would not hold. That's exactly why we're scaling up.

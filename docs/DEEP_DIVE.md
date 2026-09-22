# Deep Dive — Probe-Based Trust Signals for Structured Information Extraction

**Purpose of this document.** This is the "know it cold" reference. It explains
every part of the system: what each module does, *why* it is built that way,
what the alternatives were, and where the traps are. If someone asks a question
about this project, the answer should be findable here.

The companion document [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) is the
shorter, results-first read for someone opening the repository.

**Scope.** This documents the *reasoning-trace track* only — the DeepSeek-R1 /
SOB work. The repository is shared with a partner track that uses different
datasets and its own scripts; nothing here describes or depends on that work.

---

## Table of contents

1. [The research question](#1-the-research-question)
2. [The mental model, with a worked example](#2-the-mental-model-with-a-worked-example)
3. [The dataset](#3-the-dataset)
4. [The model and how activations are captured](#4-the-model-and-how-activations-are-captured)
5. [Stage 1 — Extraction](#5-stage-1--extraction)
6. [Stage 2 — Labelling](#6-stage-2--labelling)
7. [The reasoning trace](#7-the-reasoning-trace)
8. [Field-localized attribution](#8-field-localized-attribution)
9. [The probe and the LODO protocol](#9-the-probe-and-the-lodo-protocol)
10. [Feature variants](#10-feature-variants)
11. [Controls](#11-controls)
12. [Baselines](#12-baselines)
13. [Stage 7 — Selective regeneration, simulated](#13-stage-7--selective-regeneration-simulated)
14. [Stage 8 — Real regeneration](#14-stage-8--real-regeneration)
15. [Stage 9 — Measuring what regeneration actually did](#15-stage-9--measuring-what-regeneration-actually-did)
16. [Statistics: every test we run and why](#16-statistics-every-test-we-run-and-why)
17. [The two reversals](#17-the-two-reversals)
18. [Data-leakage audit](#18-data-leakage-audit)
19. [Limitations we disclose](#19-limitations-we-disclose)
20. [Engineering practices](#20-engineering-practices)
21. [Bugs that shaped the design](#21-bugs-that-shaped-the-design)
22. [Glossary](#22-glossary)
23. [Question drill](#23-question-drill)

---

## 1. The research question

Large language models are increasingly used to turn unstructured documents into
structured records — JSON conforming to a schema. They get a lot of it wrong. On
our corpus, **33.2% of extracted fields are wrong**.

The practical response is to regenerate: ask the model again. But regenerating
every field of every document is expensive. If you could tell *which* fields are
probably wrong, you could spend a small regeneration budget where it matters.

So the project asks two questions, and the second one is the part most work
skips:

> **Q1 (detection).** Can we predict, per field, whether the model got it wrong —
> using only signals available at inference time, without gold labels?
>
> **Q2 (value).** Is that prediction actually *worth* anything? If we spend a
> regeneration budget on the flagged fields, does the record genuinely get
> better?

Our specific angle on Q1: the model we use is a **reasoning model**. DeepSeek-R1
emits a `<think>...</think>` monologue before its JSON answer. Existing probing
work reads hidden states at the *answer* token. We ask whether the hidden states
of the **reasoning trace** carry error-relevant information that the answer
token does not.

Our specific angle on Q2: almost everyone who reports a selective-regeneration
result *assumes* a repair rate — "a re-asked wrong field becomes right with
probability 0.7" — and prices the benefit from that assumption. **We measured it
instead.**

---

## 2. The mental model, with a worked example

Here is the whole system in one example.

**A document** (SOB record, derived from HotpotQA):

```
Question: Which band, formed earlier, had a drummer who later joined
          Led Zeppelin?

Context:  [two Wikipedia passages]
```

**Its schema** (each SOB record carries its own):

```json
{"type": "object",
 "properties": {"band_name": {"type": "string"},
                "formation_year": {"type": "integer"},
                "drummer": {"type": "string"}}}
```

**The model's output**, after a long `<think>` monologue:

```json
{"band_name": "The Yardbirds", "formation_year": 1963, "drummer": "John Bonham"}
```

**Gold** says `formation_year` is 1963 ✓, `band_name` is "The Yardbirds" ✓, and
`drummer` is "Keith Moon" ✗. So field 3 is an error. Error rate for this
document: 1/3.

**What we do with it:**

| Step | What happens |
|---|---|
| Extraction | We record the hidden state at the last token of each value — `"The Yardbirds"`, `1963`, `"John Bonham"` — at 14 layers. We also keep the per-token hidden states of the whole `<think>` trace at 4 layers. |
| Labelling | Walk gold and extracted in parallel; emit one label per leaf: `band_name` → 0, `formation_year` → 0, `drummer` → 1. |
| Attribution | Search the `<think>` text for the string "John Bonham". Pool the hidden states of exactly those trace tokens into one vector for that field. |
| Probe | Logistic regression on [answer state ‖ attribution vector] → P(this field is wrong). |
| Evaluation | Train on 973 documents, predict this one. Never train on a document you predict. |
| Regeneration | Ask the model again, three times, at temperature 0.7. If two of the three agree on a different value for `drummer`, swap it in. |
| Measurement | Re-label the swapped record against gold. Did `drummer` become right? Did anything else break? |

Everything below is the detail of those seven rows.

---

## 3. The dataset

**SOB** (Structured Output Benchmark, `interfaze-ai/sob`, arXiv:2604.25359),
text / multi-hop subset. Loader: `src/probe_extraction/data/sob.py`.

### Why this dataset

The text subset is derived from **HotpotQA**, which means answering requires
combining facts from two passages. That matters for us specifically: we are
probing a *reasoning* trace, so the task has to actually require reasoning. On a
task where the answer can be copied straight out of the document, the `<think>`
block would be filler and there would be nothing to probe.

### What one record holds

| Field | Contents |
|---|---|
| `question` | The multi-hop question |
| `context` | Wikipedia passages |
| `json_schema` | **A JSON Schema specific to this record** |
| `ground_truth` | The gold JSON answer |
| `record_id` | sha256 hex, used as `doc_id` |
| `source_dataset` | becomes the domain, e.g. `sob/hotpotqa` |

### Two structural consequences

1. **The schema is per record, not per domain.** Most benchmarks have one schema
   per domain. SOB has one per record. So `get_schema(domain)` returns `{}` on
   purpose, and Stage 2 reads each `Document.schema` individually. Getting this
   wrong would silently label every document against the wrong schema.

2. **`json_schema` and `ground_truth` are stored as JSON *strings*** in the
   parquet, because their shapes vary per record and Arrow needs a fixed struct.
   `_as_dict()` parses them and tolerates an already-parsed dict.

### How the question reaches the model

`record_to_document()` folds the question into the document text:

```
Question: {question}

Context:
{context}
```

This is deliberate: it means the standard extraction prompt ("extract per the
schema from the document below") elicits the answer with **no prompt change**,
so the same prompt path serves SOB and every other benchmark in the repo.

### Offline use

Compute nodes on the cluster have no internet. `scripts/00_download_sob.py` is
run on the **login node** and caches the dataset with `datasets.save_to_disk`;
the loader uses `load_from_disk`. `HF_DATASETS_OFFLINE=1` is set in every job
script so a cache miss fails loudly instead of hanging on a network call.

### Corpus sizes as they actually are

| Count | Value | Why it differs from the one above it |
|---|---|---|
| Records requested | 1,000 | `data.max_documents` |
| Extraction files on disk | 997 | 3 documents failed to produce parseable JSON |
| Documents in the analysis | **974** | requires per-token reasoning states at **all four** capture layers, plus at least one labelled field that the model actually emitted |
| Fields in the analysis | **5,071** | leaves with `extracted_present=True` and an activation at every requested layer |
| Errors | **1,685 (33.2%)** | |

---

## 4. The model and how activations are captured

**Model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` — 28 layers, hidden
dimension 3,584, bfloat16, no quantization, `device_map="auto"`.
Wrapper: `src/probe_extraction/models/hf_model.py`.

**Decoding for extraction:** `temperature=0.0` (greedy), `top_p=1.0`,
`max_new_tokens=4096`. Greedy matters — it means a resumed or re-run extraction
reproduces the same output, so the corpus can be built across several jobs.

### The activation-capture detail worth understanding

`transformers` returns hidden states from `generate(...)` in an awkward shape:
a **tuple over generation steps**, where each element is itself a **tuple over
layers**, and each of those is a tensor. For step 0 the tensor covers the whole
prompt; for later steps it covers a single new token.

The wrapper transposes this into `{layer: (n_generated, hidden_dim)}` — one
matrix per layer, one row per generated token, aligned 1-to-1 with
`generated_token_ids`. Every downstream module depends on that alignment:
slicing a field's activation, pooling the reasoning trace, and attributing a
value to trace tokens all index into it by token position.

**Layer indexing is 1-based** and validated against `model.config.num_hidden_layers`
*before* the first forward pass, so a bad config fails in seconds rather than
after an hour of GPU time.

### What gets captured, and the disk cost

Two different things are captured, and confusing them is easy:

| What | Layers | Shape per document | Purpose |
|---|---|---|---|
| **Answer-token activations** | all 14 configured: `[1,4,7,10,12,14,16,18,19,21,23,24,26,28]` | one `(3584,)` vector per field | the answer-token probe |
| **Per-token reasoning states** | only `REASONING_TOKEN_LAYERS="16,19,23,26"` | `(n_trace_tokens, 3584)` per layer, capped at `REASONING_TOKEN_CAP=2048` tokens | field-localized attribution |

The second is what costs disk: 2,048 tokens × 3,584 dims × 2 bytes × 4 layers
≈ **59 MB per document** before compression, so ~50 GB for 1,000 documents. This
is why the capture layers are a subset, and why the extract job checks free disk
before starting.

**A trap this creates:** Stage 5 requires *every* requested layer to be present
in *every* document. If a later run used a narrower `REASONING_TOKEN_LAYERS`,
the documents written by that run would hold only a subset, and each layer would
end up with a different document set — which would silently break the paired
tests that compare layers to each other. `load_attribution_docs()` therefore
skips any document that does not have all requested layers, and logs the count.

---

## 5. Stage 1 — Extraction

`scripts/01_extract.py` → `src/probe_extraction/extraction/extractor.py`

Per document:

1. **Build the prompt.** `build_prompt_for_document()` → `build_extraction_prompt()`
   → `llm.format_chat()`. The system message tells the model to be conservative,
   emit JSON only, use `null` for missing nullable fields and `[]` for missing
   arrays, and invent nothing. The user message carries the schema (pretty-printed)
   and the document text.
2. **Generate with activation capture.**
3. **Parse the JSON** (`parse_json_output`).
4. **Locate each leaf field's token span** (`locate_fields`).
5. **Slice per-field activations** from the captured states.
6. **Pool the reasoning trace** and **store per-token reasoning states**.

### Field localization — the part that makes per-field probing possible

To read a hidden state "for a field", we need to know which generated tokens
produced that field's value. `parser.py` does this:

- `_strip_to_json()` finds the JSON inside whatever the model emitted (handling
  markdown fences and stray prose), using balanced-brace matching.
- `_iter_leaves()` walks the parsed object, yielding `(path, value)` for every
  leaf. `path` is a list like `["members", 0, "name"]`; `path_str` is its dotted
  form `"members.0.name"`.
- `_locate_one_value()` finds that value's character offset in the JSON text.
- `_compute_token_char_starts()` + `_char_offset_to_token_idx()` convert the
  character span into a **token span** `(start, end)`.

`path_str` is the join key for the entire project: activations are stored under
`"{path_str}__layer{N}"` in the `.npz`, labels are keyed by `path_str`, and the
log-prob baseline joins to the same key. If localization fails for a document it
is caught and logged, and the document keeps its parsed JSON but contributes no
fields — best-effort by design, because a crash here should not lose an
expensive generation.

### The two `position` strategies

`last_token` (used) reads the activation at the field's final token;
`mean` averages across the span. Last-token is the convention in the probing
literature — it is the position where the model has committed to the value.

### Artifacts written

```
artifacts/<experiment>/extractions/{doc_id}.json
    doc_id, domain, prompt_token_count, generated_token_count, finish_reason,
    elapsed_seconds, raw_generated_text, parsed_json, parse_error,
    token_logprobs, captured_layers,
    fields: [{path, path_str, value, is_empty, token_span, activation_layers}]

artifacts/<experiment>/activations/{doc_id}.npz
    "{path_str}__layer{N}"            -> (3584,)    per-field answer states
    "__reasoning_tokens__layer{N}"    -> (T, 3584)  per-token trace states

artifacts/<experiment>/activations/{doc_id}.rtokens.json
    the surface strings of those T trace tokens, in order
```

The `__reasoning_tokens__` prefix cannot collide with a real `path_str` because
JSON keys do not begin with `__` in these schemas — a small thing that prevents
a very confusing bug.

---

## 6. Stage 2 — Labelling

`scripts/02_label.py` → `src/probe_extraction/labeling/matcher.py`

This decides the ground truth the whole project is measured against: for each
extracted leaf, is it right or wrong?

### The walk

`Matcher._walk()` descends gold and extracted **in parallel**:

| Situation | Behaviour |
|---|---|
| both dicts | recurse over the **union** of keys (so both hallucinated and omitted keys surface), in sorted order for determinism |
| both lists of objects | pair **positionally** — `extracted[i]` against `gold[i]` |
| lists of primitives | compared as **sets**, emitted as a single field |
| one side missing/empty | recurse with an empty substitute so per-element labels are still emitted |
| types genuinely disagree | `type_mismatch` |
| leaf | compare values → one `FieldLabel` |

Positional pairing for arrays is a real modelling choice. If the model emits the
right objects in the wrong order, positional pairing marks them wrong. The
alternative — optimal assignment — would be more forgiving but would also hide
ordering errors, which are genuine extraction failures.

### Error types

`match`, `value_mismatch`, `hallucination` (model has a value, gold does not),
`omission` (gold has a value, model does not), `type_mismatch`. Hallucinations
and omissions both count as errors.

### Comparison strategies

`value_compare.py` defines `EXACT`, `CASE_INSENSITIVE`, `FUZZY`, `NUMBER`
(relative tolerance), `URL`, `EMAIL`, `DATE`, and `AUTO`. `AUTO` is type-aware:
numeric → date → case-insensitive string, so `$1,234 == 1234` and
`"New York" == "new york"`, while `FY2025 Q2 != Q1`.

A schema may declare `evaluation_config`; the matcher maps those strings
(`string_exact`, `number_tolerance`, `array_llm`, …) onto these strategies.
`string_semantic` and `array_llm` ideally want an LLM judge; we approximate both
with `FUZZY` and say so.

### The three match modes, and why we use `structure_aware`

`scripts/02_label.py` labels every document under **all three** modes (it is
cheap, CPU-only) and writes `labels/_definition_comparison.json` so the error
rate under each definition is on record. Only the configured `match_mode` is
saved as the labels used downstream.

| Mode | `leaf_default` | `structure_aware` |
|---|---|---|
| `strict` | EXACT | off |
| `auto` | AUTO | off |
| **`structure_aware`** (ours) | AUTO | **on** |

**What `structure_aware` fixes.** DeepSeek-R1 frequently emits a primitive where
the schema wants an object — `"B. Boser"` where gold is
`{"name": "B. Boser", "affiliation": ...}`. Under strict matching that is a
wholesale shape error, and *every* leaf under it counts wrong. That single
behaviour was producing a **~95% error rate** — which is not a measurement of
the model, it is a measurement of a labelling artefact. With `structure_aware`,
a primitive appearing where the other side is an object is matched against that
object's leaf values instead. The error rate becomes **33.2%**, which is a real
number about the model.

This is worth being able to explain precisely, because "your error rate changed
from 95% to 33%" sounds like moving the goalposts. It was a bug fix, the
comparison across all three definitions is written to disk on every run, and the
95% figure is not a result anyone should cite.

### Artifacts

```
artifacts/<experiment>/labels/{doc_id}.json
    per label: path, path_str, is_error, error_type, comparison_strategy,
               gold_present, extracted_present, gold_value, extracted_value
artifacts/<experiment>/labels/_summary.json
artifacts/<experiment>/labels/_definition_comparison.json
```

---

## 7. The reasoning trace

`src/probe_extraction/extraction/reasoning_trace.py`

### Finding the boundary

`find_reasoning_end_token()` accumulates token surface strings until the literal
`</think>` appears, then returns how many tokens that took. It reconstructs text
token-by-token rather than searching a decoded string because the marker is
often **split across BPE tokens** (`["</", "think", ">"]`) — a naive per-token
equality check would never find it.

It returns `0` when there is no `</think>`, which makes the whole reasoning
pipeline a no-op for non-reasoning models rather than an error.

### Document-level pooling

`reasoning_pooled_vectors()` produces two summaries per layer:

- `reasoning_mean` — mean over all trace tokens
- `reasoning_last` — the state at the `</think>` position

**Both are constant across a document's fields.** This is the single most
important fact about them, and it is why Stage 6 (document-level fusion) was
nearly null on ranking *within* a document: a constant cannot reorder fields.
It can still shift a document's overall level, which is why the effect is not
exactly zero and why it showed up once the corpus was large enough.

---

## 8. Field-localized attribution

`src/probe_extraction/extraction/reasoning_attribution.py`

This is the module that makes the reasoning trace produce something that
**varies across fields inside one document**.

### The idea

For a field whose extracted value is `"John Bonham"`, find where that string is
mentioned in the `<think>` trace, and pool the hidden states of exactly those
trace tokens. That vector is "how the model reasoned about *this* value".

### Matching, step by step

1. **Normalize** — lowercase, collapse whitespace. Both the value and the trace
   go through the same normalization, so `"Johann  Sebastian  Bach"` matches
   `"johann sebastian bach"`.
2. **Render the value** (`value_to_text`) — `None` and booleans become `""`
   (not searchable); a float that is integral renders as `5`, not `5.0`, so it
   still matches the digit in the text.
3. **Build a char→token map** (`_char_to_token`) — concatenate normalized token
   strings, recording which token owns each character. This is what lets a
   character-level regex match come back as a set of *token indices*.
4. **Find spans** (`_find_spans`) — regex with word boundaries when the value's
   edges are alphanumeric. The boundary matters: without it, the value `"art"`
   matches inside `"Mozart"` and the attribution vector is pooled from
   completely unrelated tokens.
5. **Partial fallback** (`_longest_word`) — if the full value is absent, retry
   with its longest alphanumeric word (≥ 4 chars). `"lake tahoe, california"`
   may be absent while `"california"` is present. Recorded as
   `match_type="partial"`.
6. **Pool** — mean of the matched tokens' hidden states, per layer, computed in
   float32 and stored as float16.
7. **Zero vector** when nothing matches. 13.9% of fields are in this state.

### The seven scalar features

Fixed order (`FEATURE_NAMES`) so callers can build a stable matrix:

| Feature | Meaning |
|---|---|
| `mentioned` | was the value found at all |
| `match_full` | full match vs partial-word fallback |
| `mention_count` | number of occurrences |
| `n_matched_tokens` | trace tokens covered |
| `first_frac`, `last_frac` | where in the trace, as a fraction of its length |
| `value_char_len` | length of the value — **a control**, so the probe can't mistake "long strings are rarer" for signal |

These are cheap, model-agnostic and human-readable. The hypothesis behind
`mentioned` is that **a value appearing in the final JSON but absent from the
reasoning trace is a hallucination red flag**.

### What the mention statistic actually showed

| Match type | n | error rate |
|---|---|---|
| full | 4,161 | 28.9% |
| never mentioned | 706 | 44.3% |
| **partial** | 204 | **83.3%** |

Risk ratio 1.41 for never-mentioned vs full, Fisher exact **p = 3.7 × 10⁻¹¹**.
Partial matches are dramatically worse — intuitively, the model reasoned about
*part* of the answer and then produced something else.

**But** — and this caveat must always travel with the statistic — a probe on the
7 scalars alone reaches AUROC **0.7166**, which is *below* the free
`min_logprob` baseline at 0.7529. So the mention statistic is an **interpretable
diagnostic**, not a competitive detector. Reporting it as a detector would be
overclaiming.

---

## 9. The probe and the LODO protocol

### The probe

`sklearn.linear_model.LogisticRegression`, `C=1.0`, `class_weight="balanced"`,
`max_iter=200` inside LODO (1,000 in the standalone Stage 3).

Why linear: a linear probe answers "is this information **linearly decodable**
from the representation?" A deeper probe could manufacture signal that is not
really there, and would make it impossible to claim anything about the
representation itself.

`class_weight="balanced"` matters because the classes are 33/67.

### Standardization, and where leakage would hide

```python
def _standardize(train, test):
    mu = train.mean(axis=0); sd = train.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (train - mu)/sd, (test - mu)/sd
```

μ and σ are computed on the **training fold only**. Fitting the scaler on all
data before splitting is one of the most common silent leaks in probing papers,
and it is exactly the kind of thing that inflates a number without any visible
symptom.

### LODO — leave-one-document-out

This is the evaluation protocol for every headline number.

```python
def _fit_fold(full_X, full_y, lo, hi, C):
    mask = np.ones(len(full_y), bool)
    mask[lo:hi] = False          # the held-out document's rows, all of them
    ...
```

**Why documents and not fields.** Fields inside one document share a reasoning
trace, a schema, and a gold record. A random field-level split would put some of
a document's fields in train and others in test, and the probe could learn that
document rather than the phenomenon. Grouping by document removes that.

**Two metrics, and they answer different questions:**

- **Per-document AUROC** — rank the fields *inside* each document, then average
  over documents. This is the deployment question: "given this record, which of
  *its* fields should I regenerate?" Documents whose fields are all correct or
  all wrong are degenerate (AUROC undefined) and are excluded; 681 of 974
  documents yield a valid paired fold.
- **Pooled out-of-fold AUROC** — concatenate every held-out prediction and
  compute one AUROC. This is the global question: "across the whole corpus,
  which fields are worst?"

Both are reported. They can disagree, and where they do, that is informative
rather than embarrassing.

**Cost.** 974 folds × several variants × several layers. The first
implementation took an estimated ~20 hours; with `joblib.Parallel` over folds,
`max_iter=200` and a reduced layer set it runs in ~25 minutes on 32 cores.

---

## 10. Feature variants

`build_features(doc, layer, variant)` in `scripts/05_reasoning_attribution_lodo.py`.
All operate at one layer; `ans` is `(n_fields, 3584)`, `attr` likewise.

| Variant | Features | Width |
|---|---|---|
| `answer` | answer state only | 3,584 |
| `scalars_only` | the 7 mention scalars | 7 |
| `fused_attr` | `[ans ‖ attr]` | 7,168 |
| **`fused_decomposed`** | `[ans ‖ mean(attr) ‖ attr − mean(attr)]` | 10,752 |
| `fused_scalars` | `[ans ‖ scalars]` | 3,591 |
| `fused_both` | `[ans ‖ attr ‖ scalars]` | 7,175 |

### Why `fused_decomposed` exists — the decomposition insight

Write a field's attribution vector as

```
attr_i = m_d + r_i
```

where `m_d` is the **document mean** of the attribution vectors (shared by every
field in the document) and `r_i` is the **field-specific residual**.

`fused_attr` hands the probe the *sum*. The two parts are then inseparable, and
L2 regularization over 7,168 highly correlated dimensions crushes the small
residual in favour of the large shared direction. Geometrically: within-document
cosine between fields' attribution vectors is **0.6132**, and the residual is
only **0.7934** of the mean's norm — the vectors really are mostly the same
vector wearing different labels.

So `fused_attr` **could never have tested the field-localization hypothesis**.
It was structurally unable to. Splitting `m_d` and `r_i` into separate blocks
lets the probe weight and regularize them independently — and that recovered
**60% more effect** (+0.0227 → +0.0364).

This is the single most important methodological point in the project, and it is
also a fair criticism of the earlier version of the work. It is disclosed as a
*motivated fix to an identified defect*, not a variant search: we did not try
twenty variants and report the best one.

### Results at layer 19 (per-document AUROC, Δ vs `answer`)

| Variant | Δ | p |
|---|---|---|
| **`fused_decomposed`** | **+0.0364** | **1.7 × 10⁻⁵** (Holm 1.2 × 10⁻⁴) |
| `fused_both` | +0.0242 | 0.0034 |
| `fused_attr` | +0.0227 | 0.0056 |
| `fused_scalars` | −0.0015 | ns |
| `scalars_only` | −0.0746 | — |

`fused_decomposed` is best on both metrics at both tested layers (19 and 23).

---

## 11. Controls

`scripts/06_attribution_controls.py`. Every control **holds the dimensionality
fixed** and varies only the content, so a gain cannot be explained by "you gave
the probe more numbers".

| Control | Block substituted for `attr` | Question it answers |
|---|---|---|
| `ctrl_docmean` | the document mean, repeated for every field | does per-field assignment add anything over a document-level summary? |
| `ctrl_tracemean` | the mean of the **whole** trace, repeated | does selecting value-mention tokens beat pooling everything? |
| `ctrl_centered` | `attr − mean(attr)` — residual only | is the field-specific part alone informative? |
| `ctrl_shuffled` | attribution vectors shuffled **within** the document | is the field↔vector assignment what matters? |
| `ctrl_random` | random noise, same width | is the gain just extra dimensions? |
| `ctrl_docmean_pad` | document mean padded to `fused_decomposed`'s width | width-matched comparison for the decomposed variant |

### What they showed

| Comparison | Δ | p |
|---|---|---|
| `ctrl_docmean` vs answer | +0.0263 | 5.2 × 10⁻⁴ |
| `ctrl_tracemean` vs answer | +0.0257 | 4.9 × 10⁻⁴ |
| `ctrl_centered` vs answer | +0.0124 | ns |
| `ctrl_shuffled` vs answer | −0.0083 | ns |
| `ctrl_random` vs answer | −0.0065 | ns |
| **`fused_attr` vs `ctrl_docmean`** | **−0.0036** | **0.71** |
| **`ctrl_docmean` vs `ctrl_tracemean`** | **+0.0006** | **0.96** |
| `fused_decomposed` vs `ctrl_docmean` | +0.0101 | 0.23 |
| `fused_decomposed` vs width-matched noise | +0.0236 | 0.013 (Holm-7 0.052 / Holm-3 0.026) |

**Read these carefully, because they refute our own earlier claim.** Shuffled and
random are null — good, the effect is not a dimensionality artefact. But
`ctrl_docmean` matches `fused_attr` exactly, and `ctrl_tracemean` matches
`ctrl_docmean`. So the reasoning signal that helps is **document-level**, and it
does not even require selecting the value-mention tokens.

The honest statement is: *reasoning-trace hidden states carry error-relevant
information beyond the answer token, but that information is a document-level
property of the trace, not a field-localized one.*

### Geometry diagnostic

`attribution_geometry()` reports why:

| Quantity | Value | Meaning |
|---|---|---|
| mean within-document cosine | 0.6132 | fields' attribution vectors point largely the same way |
| residual / mean norm | 0.7934 | the field-specific part is not tiny, but it is not dominant |
| fraction zero-attr | 13.9% | value never found in the trace |

Field structure **exists**; it is just not error-relevant.

---

## 12. Baselines

`src/probe_extraction/baselines/token_logprob.py`

For each field, using the token span from Stage 1:

- `mean_logprob` — average log-probability over the field's tokens
- `min_logprob` — the least-confident single token

**Sign convention.** Log-probs are ≤ 0 and higher means *more* confident. The
probe outputs P(error), where higher means *more likely wrong*. So the baselines
are **negated** before comparison. Getting this backwards would invert the
AUROC (0.75 → 0.25) — an obvious failure, but worth stating because it is the
kind of thing that gets silently patched by "flipping until it looks right".

These are the baselines that matter, because they are **free**: the model
already produced them. Any probe has to beat them to justify its existence.

`random` (a shuffled score, ≈0.5) and `oracle` (the true labels, 1.0) bracket
the range. **`oracle` is a ceiling, never a result** — it uses gold labels and
exists only to show how much of the remaining gap is detection and how much is
repair.

### Pooled LODO AUROC, layer 19

| Signal | AUROC |
|---|---|
| `random` | 0.5065 |
| `mean_logprob` | 0.7500 |
| `min_logprob` | 0.7529 |
| `probe_answer` | 0.7990 |
| **`probe_fused` (`fused_decomposed`)** | **0.8271** |
| `oracle` | 1.0000 |

The probe beats the best free baseline by **+0.074**.

---

## 13. Stage 7 — Selective regeneration, simulated

`scripts/07_selective_regeneration_sob.py`

Turns out-of-fold probe scores into a cost–quality curve.

### Two budget regimes

- **`global`** — rank *all* fields in the corpus, regenerate the top *b*. The
  deployment view when you have one budget for a batch.
- **`per_doc`** — within each document, regenerate its top *b* fraction. Matches
  the per-document AUROC and is realistic when documents arrive one at a time.

`_flag_per_doc` rounds **up** (`ceil`), because you cannot regenerate a fraction
of a field — so any non-zero budget flags at least one field per document.

### Tie-breaking

```python
order = np.lexsort((rng.random(n), -scores))
```

Ties are broken **randomly**, not by array order. If they fell back to array
order, a degenerate constant score would inherit the document ordering and score
better than chance — a signal that is really nothing would look like something.

### What it reports per budget

`errors_caught` / `recall`, `precision`, `selective_risk` (error rate among the
fields you did *not* flag), `final_error_rate` under an assumed repair model,
and the **break-even repair rate**.

### The assumption — and why it is the problem

```
residual = (total_err − caught) + caught·(1 − π) + (k − caught)·δ
```

with `π` = repair rate (assumed 0.7) and `δ` = damage rate (assumed 0.05).
**Those numbers were invented.** Every downstream figure inherited the
invention. Stages 8 and 9 exist to replace them.

### Its own results (recall at a 20% budget, global)

| Signal | recall | AURC ↓ |
|---|---|---|
| `probe_fused` | **49.0%** | 0.1410 |
| `probe_answer` | 46.5% | 0.1548 |
| `min_logprob` | 40.0% | 0.1714 |
| `mean_logprob` | 39.3% | 0.1715 |
| `random` | 19.8% | 0.2866 |
| `oracle` | 60.2% | 0.0638 |

Stage 7 also writes **`results/oof_field_scores.json`** — the per-field
out-of-fold score for every signal. Stage 9 ranks by that file rather than
recomputing the LODO sweep, which keeps the two stages from drifting apart and
makes Stage 9 a fast, re-runnable CPU job.

---

## 14. Stage 8 — Real regeneration

`scripts/08_regenerate.py` (GPU)

For every document with an extraction, re-run the model `k=3` times at
temperature 0.7 and save each attempt's parsed JSON. **Nothing is selected and
nothing is compared here** — this stage only produces raw material, so one
regeneration pass serves every budget and every scoring signal.

### Three design points the result depends on

1. **Temperature must be > 0.** Extraction is greedy, so a greedy re-run
   reproduces it token for token and could never repair anything. The script
   **exits with code 2** rather than run at temperature 0.
2. **The prompt is identical to extraction.** Both go through
   `build_prompt_for_document()`. If the prompts differed, a measured "repair
   rate" would partly reflect the prompt change rather than a second attempt.
3. **No hidden states are captured.** The probe has already scored the original
   fields; regeneration only needs values. Skipping capture makes the pass
   cheaper and the artifacts small (4 MB for 997 documents, versus ~50 GB for
   Stage 1).

### Robustness

Per-document writes, `--resume` (counts existing samples), `--shard I/N`
(1-based) so several GPU jobs split the work into disjoint files. A failure in
one sample is caught and recorded rather than losing the run.

### What the run produced

997 documents × 3 samples = 2,991 generations. **1** hit `max_new_tokens`,
**13** failed to parse. 99.5% usable — far better than expected.

---

## 15. Stage 9 — Measuring what regeneration actually did

`scripts/09_regen_evaluate.py` (orchestration) +
`src/probe_extraction/regen/evaluate.py` (the accounting)

### The outcome vocabulary

| Outcome | Meaning |
|---|---|
| `repaired` | was wrong, now right |
| `damaged` | was right, now wrong |
| `unchanged` | the label did not move |
| `unavailable` | the resample had no value at that path — nothing was swapped |
| `lost` | the swap changed the record's shape and the leaf no longer exists to label |

Plus a **status** axis: `swapped`, `identical` (the resample returned the same
value — budget that buys literally nothing), `unavailable`, `set_failed`,
`label_failed`.

`identical` is tracked separately on purpose: "the model repeated itself" is a
completely different finding from "the model changed its mind and was still
wrong", and collapsing them would hide the most striking fact in the data
(60% of resampled fields came back byte-identical).

### Addressing a single leaf

`src/probe_extraction/utils/jsonpath.py` — `json_get`, `json_set`,
`with_replacements`. Three deliberate behaviours:

- **`json_set` refuses to create a missing path.** Creating a key would
  fabricate a field the model never emitted — exactly what this analysis must
  not do.
- **Booleans are not list indices.** `True == 1` in Python, so `record["members"][True]`
  would silently address element 1. Explicitly rejected.
- **A stored `None` is distinguishable from an absent path.** A regenerated
  `null` is a real answer; a missing path means the regeneration had nothing to
  offer. Conflating them would misclassify a genuine replacement as
  `unavailable`.

`with_replacements` deep-copies, so each budget builds its own hybrid from the
same untouched original and budgets cannot contaminate each other.

### Which resamples are allowed to count

`usable_samples()` excludes:

- **truncated** (`finish_reason == "length"`) — its JSON exists only because the
  parser repaired a cut-off generation; crediting or blaming the model for text
  it never produced would be wrong
- **parse failures**
- **non-object JSON**

### The three strategies

| Strategy | Rule |
|---|---|
| `first` | the first usable resample — the honest **k = 1** deployment cost |
| `vote` | plurality value across usable resamples (self-consistency) |
| `vote_strict` | the same, but only when ≥ 2 agree **and** they form a strict majority; otherwise leave the field alone |

`first` **fixes the sample and then reads the path** — it must not fall through
to a later sample when the first lacks that path, or `first` would quietly
become best-of-k and overstate what a single call buys. There is a test for
exactly this.

`vote` votes on a canonical JSON serialization (`json.dumps(..., sort_keys=True)`)
because values can be unhashable lists or dicts, and breaks ties by first
appearance so the result does not depend on dict iteration order.

### `net_delta` — the accounting that makes the budget curve cheap and honest

Naively, every (strategy, signal, regime, budget) would need its own joint
re-labelling of all 974 documents — hundreds of thousands of matcher runs.

Instead: **measure each field's swap once, alone**, and record `net_delta`, the
change in *that document's error count over the whole scored field set*.

Two consequences:

- Because `net_delta` is computed over the whole scored set, a swap is
  automatically **charged for collateral damage** to its neighbours (array
  re-alignment, for instance).
- Any budget is then just the sum of the net deltas of the fields it flags.

That is an **additivity assumption**, and it is not taken on faith.

### The additivity check

`--checkpoint-budgets` (default 10%, 20%, 100%) re-does selected budgets
**jointly**: swap every flagged field at once, re-label the document once, and
compare against the additive prediction. Budget 100% is always checked because
regenerating everything is the most interaction-heavy case possible.

**Result: exact agreement — +0 at every checkpoint, for all three strategies.**
This is the strongest single piece of evidence that the measurement is correct,
because the joint pass uses none of the per-field arithmetic.

### Two conventions, stated

- A scored field missing from the after-labels (`lost`) **keeps its original
  label** — the swap is neither credited nor blamed for a field it destroyed.
  These are counted and reported.
- Leaves the swap **created** are ignored. The scored set is what the probe
  ranked; it must stay fixed or budgets would not be comparable.

### Two guards that refuse rather than warn

1. **Incomplete regeneration.** If any scored document lacks a Stage 8 file,
   the stage exits 1. A partial run would silently report a number computed on a
   subset. `--allow-partial` overrides it deliberately.
2. **The self-check.** Re-label each **original** record and require it to
   reproduce the labels Stage 2 stored. Everything this stage reports is a
   difference between a before-label and an after-label; if the two are not
   produced under identical conditions — same gold, same schema, same matcher
   settings — every repair and damage count is an artefact of that difference,
   and it would not announce itself. One labelling per document buys the guard.
   It passed on all 5,071 labels.

This guard is not theoretical. During development, a run with an empty benchmark
produced a plausible-looking 25% baseline built entirely from **missing gold**.
The self-check is what catches that.

### Controls

- **The full resample, no probe** — replace each record wholesale with its
  resample: **44.1%** error rate versus the original's **33.2%**. The second
  attempt is markedly worse than the first.
- **Budget 1.0** — regenerate everything: `first` 42.0%, `vote` 37.0%,
  `vote_strict` 32.5%.
- **`random` and log-prob flagging** at every matched budget.

---

## 16. Statistics: every test we run and why

| Test | Where | Why that test |
|---|---|---|
| **Paired Wilcoxon signed-rank** on per-document AUROC | Stages 7, 8 | Variants are compared on the *same* documents, so the comparison is paired. Wilcoxon is non-parametric — per-document AUROCs are bounded in [0,1] and far from normal. |
| **Holm–Bonferroni** | Stages 7, 8, 12 | We run a family of comparisons. Holm is uniformly more powerful than plain Bonferroni while still controlling family-wise error. Implemented once in `06_attribution_controls.py:holm_bonferroni` and imported by Stage 9, so the correction is identical everywhere. |
| **Document-level bootstrap CI** | Stages 8, 12 | Documents are the unit of independence; fields inside one share a trace, a schema and a gold record. Resampling fields would give intervals that are far too narrow. |
| **Paired bootstrap** for signal comparisons | Stage 9 | Within one replicate, all signals are evaluated on the *same* resampled corpus. Comparing two independently resampled corpora would drown a 1-point difference in between-corpus variance. |
| **Fisher exact** | mention analysis | A 2×2 contingency table with small cells. |
| **AURC** (area under the risk–coverage curve) | Stages 9, 12 | Summarizes the whole selective-prediction curve in one number, rather than cherry-picking a budget. |

### How the bootstrap works in Stage 9

```
for each of 2,000 replicates:
    resample 974 documents WITH replacement
    gather their fields
    re-flag the top-b within that replicate      (global regime)
    final_errors = base_errors + sum(net_delta[flagged])
    record the reduction, and each paired signal difference
```

Flags are **recomputed inside each replicate** for the global regime, because a
global budget is defined over whatever corpus you have. Per-document flagging
does not depend on the rest of the corpus, so it is computed once and reused —
the same masks, not an approximation.

A single fixed tie-break vector is drawn once, so ties resolve identically in
every replicate and the comparison stays paired.

p-values are two-sided percentile bootstrap: `2 · min(P(x ≤ 0), P(x ≥ 0))`.
When many replicates land exactly on zero this is **conservative**, which is the
safe direction.

---

## 17. The two reversals

This project changed its story twice. Being able to narrate that honestly is
worth more than pretending the first version was right.

### Reversal 1 — the "clean null" was a power artefact

Update 06 reported that document-level reasoning fusion was null, and framed
that as a clean negative result motivating field localization.

It was underpowered. The point estimates were **positive all along**
(+0.003 to +0.013) — the corpus (300 documents) simply could not resolve them.
At 994 documents the same test gives **p = 1.2 × 10⁻⁴**.

**Lesson:** "not significant" is not "no effect", and at n=300 with a small
effect you cannot tell the difference.

### Reversal 2 — field localization does not work as a mechanism

Update 07 claimed field-localized attribution beats the answer-token probe, and
attributed the gain to the localization.

The controls refuted the mechanism. `ctrl_docmean` (the same vectors collapsed
to one per document) matches `fused_attr` exactly (Δ = −0.0036, p = 0.71), and
`ctrl_tracemean` (pooling the *whole* trace) matches `ctrl_docmean`
(Δ = +0.0006, p = 0.96). **The effect is document-level.**

And `fused_attr` could never have tested the hypothesis anyway, for the
decomposition reason in §10.

**What survives:** the reasoning trace carries error-relevant information the
answer token does not, and `fused_decomposed` is the best way we found to
extract it. What does *not* survive is the claim that the mechanism is
field-level localization.

---

## 18. Data-leakage audit

Performed 2026-09-08, recorded in `Update_08.md` §10.

| Component | Verdict |
|---|---|
| LODO | **Clean.** `_fit_fold` masks the held-out document's rows entirely; `_standardize` fits μ/σ on the training fold only. |
| Document-mean features | **Legitimate.** Built from model outputs only — the extracted values versus the model's own trace. No gold anywhere. Computable at deployment time. |
| Log-prob baselines | **Clean.** No training at all; ranked over the same out-of-fold field set. |
| Stage 9 flagging | **Clean.** Flags come from Stage 7's out-of-fold scores. Gold is used only to *label* the outcome, never to *select* what to regenerate. |
| **Stage 3 cross-validation** | **LEAKY — disclosed.** `StratifiedKFold` over *fields*, not grouped by document. Diagnostic only; a docstring warning lives in `probes/linear.py`. Never use it for probe-vs-baseline. |
| **Layer 19** | **Disclosed.** Pre-committed from the 300-document pilot, which is a subset of the 994. Mitigations: the effect holds at layer 23; the CV optimum is layer 21, so we are not reporting a best case; paired tests use the same layer for every variant. |
| **`fused_decomposed`** | **Disclosed.** Not pre-registered. A motivated fix to an identified defect, not a variant search. |

The distinction that matters most for Stage 9: **the probe predicts which
fields are wrong; we never use gold to choose what to regenerate.** Gold enters
only afterwards, to score whether the swap helped. `oracle` is a labelled
ceiling and is always presented as such.

---

## 19. Limitations we disclose

1. **One model, one dataset.** DeepSeek-R1-Distill-Qwen-7B on SOB. A second
   model was explored (Qwen3.5) and dropped — see §21.
2. **The downstream gain from reasoning features is not significant.**
   `fused_decomposed` beats `answer` clearly on AUROC (+0.0364, p = 1.7 × 10⁻⁵),
   but after regeneration the error-rate difference is +0.37 points with a 95%
   CI of [−0.18, +0.92], Holm p = 0.19. **AUROC gains compress when passed
   through a weak repair operator.** This must be volunteered, not waited for.
3. **The absolute improvement is 1.6 points** (33.2% → 31.6%). Real, measured,
   significant — and modest.
4. **`vote_strict` is post-hoc.** It was designed after seeing that damage was
   high. The pre-planned strategies are `first` and `vote`, and `vote` is
   significant on its own.
5. **Temperature was never tuned.** 0.7 by convention. A cooler temperature is
   an obvious and untested lever.
6. **Most errors are not repairable by resampling.** 60% of fields come back
   byte-identical; the oracle ceiling is 26.5%.
7. **3 of 5,071 fields could not be re-labelled** after their swap
   (`RecursionError` on a deeply nested record). Excluded from the
   repair/damage denominators and reported as lost coverage.
8. **Layer 19 pre-commitment** and **Stage 3 CV leakage**, as above.

---

## 20. Engineering practices

### Restartability

Every long job is resumable and sharded, because SLURM wall clocks are shorter
than the work:

- `--resume` skips completed documents (Stage 1 additionally checks that the
  `.rtokens.json` sidecar exists, so an attribution run resuming over an older
  extraction re-does it rather than silently shrinking the corpus)
- `--shard I/N` (**1-based**) splits work into disjoint files
- per-document writes, so a wall-clock kill loses only the document in flight

### Shard values must be positional, never environment variables

`#SBATCH --export=NONE` strips the environment. A value passed as `VAR=x sbatch job.sh`
never reaches the job. `sbatch` *does* forward positional arguments. This bit us
twice (see §21), and both run scripts now take positional arguments.

### Delivery to the cluster

Work is developed locally and delivered as a self-contained `install_*.sh`:
each file embedded as base64 (byte-exact, no quoting hazards), sha256-verified
after writing, and backed up **only when the content actually differs** — so
re-running the installer is a no-op that creates no `.bak` clutter. Every
installer is round-trip verified locally before it is sent: install into a
scratch copy, diff against source, run the test suite there.

### Tests

**271 tests** pass with

```bash
PYTHONPATH=$PWD/src python -m pytest tests/ -q --ignore=tests/test_extract_bench.py
```

(16 more in `test_extract_bench.py` need PyMuPDF, which belongs to the partner
track's PDF benchmark.)

The testing philosophy that matters here: **this project's failure mode is not a
crash, it is a plausible wrong number.** So the tests pin down arithmetic
(repair/damage accounting, budget counts, break-even), addressing
(`json_get`/`json_set` edge cases), and — crucially — the cases where a test
could *manufacture* a result. For example, `test_no_improvement_gives_a_ci_that_contains_zero`
asserts that a repair operator which does nothing comes out **not significant**.

---

## 21. Bugs that shaped the design

Each of these left a permanent mark on the code, and each is worth being able to
explain.

| Bug | Consequence | Fix |
|---|---|---|
| **Stage 5's `VARIANTS` tuple missing `fused_decomposed`** | The variant was added to `build_features()` but not to the tuple the main loop iterates, so a 90-minute GPU job re-swept the old five variants. | Added to `VARIANTS`, plus a `--variants` flag; three post-loop iterations switched from `VARIANTS` to the selected list. |
| **`MODEL=... sbatch`** | `--export=NONE` stripped it; the config's `PLACEHOLDER-SET-ME` reached the HF Hub as a 401. | Positional arguments + a placeholder guard that exits before any network call. |
| **`SHARD=... sbatch`** | Same class. Documented sharding silently did nothing; three "shards" would each have processed the whole corpus. | Positional. |
| **Split instability** | Dev/held-out was cut by rank, so the midpoint moves as the corpus grows and boundary documents flip sides — contaminating any held-out claim. A test caught it. | Threshold on each document's **own** salted hash (`splits.py`, salt `sob-reasoning-track-v1`, **frozen**). Cost: halves are balanced only in expectation. |
| **`np.trapz` removed in NumPy 2.0** | Cluster and laptop on different sides of the change. | `_trapezoid` compatibility shim. |
| **Recursion limit not inherited by joblib workers** | `sys.setrecursionlimit(20000)` was set in the script, but workers are **fresh interpreters that inherit nothing**. They ran at the default 1,000 and a completed 13-hour GPU run died at the evaluation step. | Set at **module scope** in `probe_extraction/regen/evaluate.py` — a worker must import that module to unpickle `measure_document`, so the limit is guaranteed in place. Regression test spawns a real subprocess, because an in-process assertion would pass even with the bug present. |
| **Unguarded labeller call in the control block** | The one remaining `labeler()` without a try/except killed the run at the very end. | All labeller calls now record `label_failed` and continue; failures are counted and reported as lost coverage. |
| **Update_08 §4 mixed a leaky CV number with clean baselines** | Found during the leakage audit. | Headline replaced with the matched LODO comparison. |

### The Qwen exploration, and why it was dropped

A second model (Qwen3.5) was investigated for generality. Findings:

- `enable_thinking=False` was **hardcoded** in `hf_model.py`. Qwen3.x suppresses
  `<think>` unless the chat template asks for it, and `find_reasoning_end_token`
  keys off the literal `</think>`. A Qwen run would have extracted fine and then
  produced a **silent null** at Stage 5. Fixed: `ModelConfig.enable_thinking`,
  default `False` so DeepSeek is unchanged, threaded through to `format_chat`.
- Qwen3.5-4B and 9B both ran, but at ~22 tok/s — about **6.6× slower per
  document** than DeepSeek because the traces are far more verbose.
- **~1/3 of documents were truncated** at `max_new_tokens=4096`, and truncated
  generations produced *fake* "parsed" JSON through the repair fallback — a
  data-integrity hazard.

The track was dropped to focus on real regeneration for DeepSeek. `enable_thinking`,
`splits.py` and the CLAP removal were kept.

---

## 22. Glossary

| Term | Meaning |
|---|---|
| **Probe** | A small classifier (here: logistic regression) trained on a model's hidden states to test whether some property is linearly decodable from them. |
| **LODO** | Leave-one-document-out. Train on all documents but one, predict that one, repeat. |
| **OOF** | Out-of-fold. A prediction made for a data point by a model that never saw it. |
| **`path_str`** | Dotted JSON path, e.g. `members.0.name`. The join key across every artifact. |
| **`attr_vec`** | Attribution vector: the mean hidden state of the trace tokens where a field's value is mentioned. |
| **`m_d` / `r_i`** | Document mean and field residual of the attribution vectors: `attr_i = m_d + r_i`. |
| **`fused_decomposed`** | The proposed feature set: `[answer ‖ m_d ‖ r_i]` as separate blocks. |
| **AUROC** | Probability a randomly chosen wrong field is ranked above a randomly chosen right one. 0.5 = chance. |
| **AURC** | Area under the risk–coverage curve. Lower is better. |
| **Selective risk** | The error rate among the fields you did *not* flag. |
| **Repair rate** | P(a regenerated wrong field becomes right). Measured: 0.215 / 0.233 / 0.385. |
| **Damage rate** | P(a regenerated right field becomes wrong). Measured: 0.234 / 0.167 / 0.080. |
| **`net_delta`** | Change in a document's error count over the scored field set caused by one field's swap — includes collateral damage. |
| **Holm–Bonferroni** | Step-down multiple-comparison correction; more powerful than Bonferroni, same family-wise guarantee. |
| **Hallucination / omission** | Model has a value where gold does not / gold has a value where the model does not. Both count as errors. |

---

## 23. Question drill

**"What is the one-sentence contribution?"**
A linear probe on a reasoning model's hidden states detects wrong extracted
fields better than the model's own confidence (AUROC 0.827 vs 0.753), and —
unlike prior work, which assumes a repair rate — we *measured* what acting on
that detection is worth: a significant but modest 1.6-point error reduction,
about three times smaller than the standard assumption predicts.

**"Why should I believe the 33.2% error rate?"**
It is multi-hop QA with a per-record JSON schema, scored leaf by leaf. We label
under three definitions on every run and write the comparison to disk. An
earlier labelling artefact (primitives emitted where the schema wanted objects)
had put it near 95%; `structure_aware` matching fixed that, and 33.2% is the
honest number.

**"How do you know the probe never saw the answers?"**
LODO: every fold masks the held-out document's rows entirely, and
standardization is fitted on the training fold only. Features come from the
model's own outputs — its trace and its extracted values — never from gold.

**"Why is the improvement only 1.6 points?"**
Because the repair operator is weak, not the detector. With *perfect* error
detection the oracle only reaches 26.5%. We can decompose exactly how much of
the remaining gap belongs to detection and how much to repair — that separation
is itself a contribution.

**"Isn't the reasoning-trace idea the whole point? And it isn't significant
downstream."**
Correct, and we say so. It is clearly significant on detection (+0.0364,
p = 1.7 × 10⁻⁵) and not significant on downstream error rate (+0.37,
p = 0.19). The honest reading is that AUROC gains compress through a weak repair
operator — which is itself an argument for measuring repair rather than
assuming it.

**"Your story changed twice. Why should I trust version three?"**
Because both changes came from controls we ran on ourselves, and both are
documented. The first reversal was a power artefact at n=300; the second was a
control showing our claimed mechanism was wrong. Version three is the one that
survived controls that were designed to kill it.

**"How do you know the Stage 9 measurement is arithmetically right?"**
Two independent computations agree to the unit. Each field's swap is measured
alone; a joint pass then re-labels every flagged field at once using none of
that arithmetic. They matched exactly at every checkpoint, for all three
strategies. Re-labelling the untouched originals also reproduced all 5,071
stored labels.

**"How is this different from self-consistency?"**
Self-consistency regenerates everything. Here a probe spends a budget on the
fields most likely to be wrong. Regenerating everything with a single resample
makes the record *worse* — 42.0% versus 33.2% — so the selection is what carries
the result.

**"Is the 20% budget cherry-picked?"**
It was fixed in Stage 7 before any of this was measured, and the full curve from
0 to 100% is reported. The result also holds at 10%.

**"What would you do next?"**
A stronger repair operator, since that is now demonstrably the bottleneck:
lower temperature, targeted re-prompting that names the suspect field, or
retrieval instead of blind resampling. Then a second model for generality.

**"What is `vote_strict` and did you invent it after seeing the results?"**
Yes — and it is labelled post-hoc everywhere it appears. It only overwrites a
field when at least two of the three resamples agree. The pre-planned strategy
is plain `vote`, which is significant on its own; `vote_strict` is reported as a
robustness finding, not as the headline.

**"Why temperature 0.7?"**
Convention, and it was never tuned. That is a stated limitation. The data
suggests it is too hot: 60% of resamples were byte-identical and the damage rate
for a single resample (0.234) exceeded its repair rate (0.215).

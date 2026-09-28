# Cross-Model Label Contamination: The Llama-3.1-8B Structural Mismatch

_Diagnostic note · Probe-based trust signals project · 9 June 2026_

---

## 1. Summary

When replicating the extraction pipeline on Llama-3.1-8B (cross-model check
vs the primary Qwen3.5-4B), three of four domains showed implausibly high
"error" rates:

| Domain | Qwen error rate | Llama error rate |
|---|---|---|
| sport/swimming | ~10% | 10.2% |
| academic/research | low | 44% |
| finance/10kq | ~12% | 85% |
| finance/credit_agreement | (moderate) | 86% |

The finance/academic rates are not real extraction errors. They are an
artefact of **structural / key-naming divergence** between Llama's output and
the gold annotation, which the path-based matcher counts as wholesale
mismatches. Swimming is unaffected because its schema is flat (no nested
objects to relabel).

**Consequence:** a Llama probe trained on the pooled four-domain set would
learn from labels that are ~85% wrong on the finance domains — i.e. it would
learn a labelling artefact ("Llama finance output = error"), not a genuine
trust signal. We therefore restrict the Llama cross-model result to the
swimming domain, where labels are trustworthy, and document the divergence
here as a finding/limitation.

---

## 2. The structural mismatch, concretely

The matcher compares model output to gold by **JSON path**. A field is an
"error" if the value at a given path differs from gold, and a "hallucination"
if the model has a value at a path where gold has nothing.

Llama and the gold annotation represent the same information at **different
paths**, so nearly every nested field is flagged as a hallucination even when
the *content is correct*.

Example — the "administrative agent" party in a credit agreement:

| | Path | Structure |
|---|---|---|
| Gold / schema convention | `parties.administrative_agent` | snake_case key, flatter |
| Qwen output | `parties.administrative_agent` | matched gold → 0 errors |
| Llama output | `parties.Administrative Agent.address` | Title-Case key **with a space**, extra nesting |

The extracted *value* was correct (e.g. the agent's address
`350 5th Ave #6740 New York, NY 10118`), but it sat at a path
(`parties.Administrative Agent.address`) that does not exist in gold. The
matcher therefore recorded it as a hallucination. Multiplied across every
party, every lender, and every nested `terms` sub-object, this produced the
85–99% "error" rates.

Two distinct divergences are at play:

1. **Key naming.** Llama renames the schema's literal keys into
   natural-language Title Case (`"Administrative Agent"` instead of
   `administrative_agent`). A string mismatch at the key level.
2. **Nesting depth.** Llama sometimes adds structure (objects with
   `name` / `address` / `contact` sub-fields) where gold is flatter. A
   structural mismatch, not just a string difference.

Key naming alone could in principle be normalised (lowercase, replace spaces
with underscores). The nesting difference cannot — it changes the path
*shape*, so no per-key string normalisation recovers the alignment.

---

## 3. Why this is a labelling problem, not a model-quality problem

It is important to be precise about what failed:

- Llama did **not** (necessarily) extract wrong information. In the audited
  cases the values were plausible/correct.
- The pipeline's **path-based comparison** assumed the model would emit the
  schema's literal key structure. Qwen did; Llama did not.
- So the "errors" are a mismatch between Llama's output convention and the
  gold's path convention — upstream of the probe entirely.

This matters for interpretation: it is **not** evidence that Llama is a worse
extractor, nor that the probe fails on Llama. It is evidence that the
extraction→gold alignment is sensitive to a model-specific formatting choice.

---

## 4. Why swimming is unaffected

Swimming's schema is essentially **flat**: a championship, event details, and
a list of result objects with primitive fields (rank, time, athlete, country,
year, team, records). There are no deeply nested named sub-objects like
`parties.<role>.<attribute>` for the model to relabel. With no nested keys to
paraphrase, Llama's output aligns with gold's paths, and the labels are
trustworthy (10.2% error, matching Qwen).

This is the diagnostic signature: the divergence scales with **schema
nesting / number of named object keys**, which is why swimming (flat) is
clean, academic (some nesting) is moderately contaminated (44%), and finance
(deeply nested parties/terms) is severely contaminated (85%+).

---

## 5. Would another LLM have this problem?

**This cannot be answered with certainty without testing the specific model.**
What we can say with reasoning:

- The behaviour is a **model-specific instruction-following trait**: how
  strictly the model preserves the schema's literal key strings versus
  paraphrasing them into natural language. Qwen preserved them; Llama
  paraphrased.
- It is therefore **plausibly model-dependent, not universal.** Different
  models, with different instruction-tuning and different amounts of
  structured-output / JSON / function-calling training, will sit at different
  points on the "preserve keys" ↔ "paraphrase keys" spectrum.
- Models heavily optimised for strict JSON or function-calling output may
  preserve keys more faithfully; models optimised more for natural-language
  helpfulness may paraphrase more.
- **But predicting any particular model's behaviour (GPT-4-class, Gemini,
  Mistral, etc.) is an empirical question that requires running that model on
  the schemas and inspecting the output.** We have a two-model sample (Qwen
  preserves, Llama paraphrases); generalising beyond that would be
  speculation.

The honest position: we observed this divergence between two models; we cannot
assume a third behaves like either without testing; but the mitigation in
Section 6 makes the pipeline robust to the behaviour regardless of which model
is used.

---

## 6. Solutions (in order of robustness)

### 6.1 Constrained / schema-guided decoding (most robust, model-agnostic)
Force the model to emit JSON conforming to the schema's exact keys, using a
grammar- or schema-constrained decoder (e.g. JSON-schema-constrained
generation, guided decoding). The model then *cannot* rename keys or alter
nesting — the output structure is guaranteed to match gold's paths. This
removes the problem at the source and works for any model. It is the
principled long-term fix and the one we would recommend for a paper-quality
pipeline. Cost: requires a decoding library that supports schema constraints
and some integration work.

### 6.2 Prompt-level key enforcement (cheap, partial)
Strengthen the extraction prompt to explicitly require the schema's literal
keys: include the exact key names, instruct "use these keys verbatim; do not
rename, translate, or add nesting." This nudges the model toward alignment but
does not *guarantee* it — a model inclined to paraphrase may still drift.
Cheap to try; would require re-running extraction (re-sampling activations)
since it changes the prompt and therefore the context the model sees.

### 6.3 Post-hoc structural alignment (salvage, brittle)
Write a mapping layer that normalises model output keys to the schema's
convention before matching: lowercase, spaces→underscores, and a hand-built
map for known renamings (`"Administrative Agent"` → `administrative_agent`).
This recovers the *key-naming* divergence but **not** the *nesting* divergence
(extra object levels), and it is brittle (every new renaming needs a rule).
Useful as a partial rescue, not a clean solution.

### 6.4 Restrict to structurally-aligned domains (what we did, honest)
Report the cross-model result only on domains where the model's output
structurally aligns with gold (swimming). Document the divergence on the
others as a limitation. This is honest and immediate, at the cost of a
narrower cross-model claim. Appropriate under time constraints; should be
upgraded to 6.1 for a final paper.

---

## 7. Recommendation

- **Now (this round):** restrict Llama cross-model replication to swimming
  (clean labels). Report Qwen-swimming vs Llama-swimming as the same-domain
  cross-model comparison. Cite this divergence as a documented limitation.
- **For a paper:** adopt schema-constrained decoding (6.1) so extraction output
  is guaranteed to match the gold path structure for every model, removing the
  contamination at the source and enabling a full multi-domain cross-model
  comparison.

The finding itself is worth stating positively: the probe-based trust signal
generalises across model architectures **when extraction output is
well-formed**; the bottleneck for cross-model generalisation is
extraction-formatting consistency (a property of the model's instruction
following), not the probe.

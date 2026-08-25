# Paper Skeleton — Reasoning-Trace Trust Signals for Structured Extraction

**Status:** drafting scaffold, written 2026-08-16 while the controls (job 243526)
and the 1,000-document extraction (job 243527) run.
**Target:** NLP lab course deliverable, written to conference-submission standard.
**Convention:** `[[SLOT: name]]` marks a number that lands from a pending run.
Every slot names the artifact it comes from, so filling them in is mechanical.

---

## 1. The claim ladder

Write the paper as four claims, strongest evidence first. Each rung is
independently defensible, so a weak rung damages only itself.

| # | Claim | Evidence | Status |
|---|-------|----------|--------|
| C1 | Per-field errors in structured extraction are **linearly decodable** from a reasoning model's hidden states, better than the model's own confidence. | Probe 0.850 CV / 0.81–0.84 LODO vs 0.771 best log-prob baseline (+0.08) | **Done** |
| C2 | Naive use of the reasoning trace **does not help**, for a reason that is structural rather than empirical. | Stage 06 doc-level fusion null; doc-constant vector cannot re-rank within a document | **Done** |
| C3 | **Localizing** the reasoning to each field recovers a real gain — and it is the localization, not the added capacity, that does it. | Stage 07 (Δ+0.0227, p=0.044) + Stage 08 controls | Stage 08 pending |
| C4 | The signal is **worth something**: it buys a better cost-quality tradeoff for selective regeneration than free baselines. | Stage 09 curves | Pending |

C1 and C2 are already secure. C3's *existence* is established and its
*attribution to localization* is what job 243526 decides. C4 is new.

**If C3's controls come back null** (the shuffle control matches `fused_attr`),
do not bury it. Re-frame the paper around C1 + C2 + the labeling contribution,
and report the dimensionality confound as a finding — "field-localized pooling
gives no gain beyond added capacity" is a publishable negative given that C2
already set up the question. Ali's branch has no equivalent result either way.

---

## 2. Title candidates

1. *Reading the Reasoning: Field-Localized Trace Attribution as a Trust Signal for Structured Extraction*
2. *Where the Model Thought About It: Localizing Reasoning Traces for Per-Field Error Detection*
3. *Trust Signals from Reasoning Traces in Structured Information Extraction*

(1) leads with the mechanism and is the most specific. Keep the supervisor's
project title intact wherever the course requires it.

---

## 3. Abstract template

> Large language models increasingly emit structured records, but expose no
> reliable per-field confidence, so downstream systems cannot tell which fields
> to trust. We train linear probes on the hidden states of a reasoning model
> (DeepSeek-R1-Distill-Qwen-7B) to predict per-field extraction errors on SOB, a
> multi-hop structured-output benchmark. The probe reaches **0.85 AUROC** under
> cross-validation and **0.81–0.84** under leave-one-document-out, outperforming
> token-log-probability baselines by **0.08 AUROC**. We then ask whether the
> model's explicit reasoning trace carries error-relevant signal beyond the
> answer token. Pooled at the document level it does not — a null we explain
> structurally: a document-level vector is constant across that document's
> fields and cannot change their relative ranking. Localizing the trace to each
> field instead — pooling the hidden states of the reasoning tokens where that
> field's value is mentioned — yields a consistent improvement of
> **[[SLOT: delta]] AUROC** (paired Wilcoxon **p = [[SLOT: p]]**), which
> survives controls that hold feature dimensionality fixed and destroy only the
> field-to-reasoning correspondence. Applied to selective regeneration, the
> signal catches **[[SLOT: recall]]%** of errors within a 20% regeneration
> budget versus **[[SLOT: baseline_recall]]%** for the best free baseline. We
> additionally report a structure-aware labeling correction that reduces an
> apparent 95% error rate to a genuine 41%, and a probe-free hallucination
> indicator: values absent from the reasoning trace are wrong
> **[[SLOT: risk_ratio]]×** as often.

Slots come from `results/attribution_controls.json` and
`results/selective_regeneration.json` (both regimes; quote `per_doc`).

---

## 4. Section plan

### 1. Introduction
Problem: structured extraction is deployed, per-field confidence is not
available. Selective regeneration needs it. Contributions as a bulleted list
mirroring the claim ladder. State the negative result **in the intro** — it is
part of the contribution, not an appendix apology.

### 2. Related work
Four buckets. Two to three sentences each; this is where a lab report usually
under-invests and a reviewer notices.
- **Probing / interpretability of hidden states** — linear probes as evidence of
  linear decodability.
- **Hallucination & error detection in LLM outputs** — log-prob and entropy
  baselines, self-consistency; position ours as *internal-state* rather than
  *sampling-based*, and note the cost asymmetry (self-consistency needs k
  generations; a probe needs zero extra).
- **Reasoning models and chain-of-thought faithfulness** — the live question of
  whether the trace reflects the computation. Our field-localized result is a
  small piece of evidence that it partly does. **This is the framing that makes
  the paper interesting beyond the application.**
- **Structured extraction benchmarks & evaluation** — SOB, ExtractBench; the
  flat-vs-nested scoring artifact we correct.

### 3. Method
3.1 Task and notation. 3.2 The linear probe. 3.3 Reasoning-trace capture
(`<think>` boundary, pooled vs per-token). 3.4 **Field-localized attribution** —
the core: value → normalized string match in the trace → token span → mean-pool
those hidden states; plus the seven interpretable scalars. Include the
worked tambourine/zils example from `Project_Master_Guide.md` — a concrete
example here is worth a paragraph of prose. 3.5 Evaluation protocol: LODO, why
per-doc *and* pooled AUROC measure different things, paired Wilcoxon.

### 4. Experimental setup
Model, dataset, layers, decoding, hardware. **Structure-aware labeling** goes
here with the three-mode table — presented as a method choice with a
sensitivity analysis, not as a bug fix.

### 5. Results
5.1 C1: probe vs baselines, per-layer curve. 5.2 C2: doc-level fusion null +
the structural argument. 5.3 C3: field-localized attribution + **controls**.
5.4 C4: selective regeneration. 5.5 The mention analysis (probe-free signal).

### 6. Analysis / discussion
Why mid-to-late layers peak. Why the hand-made scalars are weak while the
pooled hidden states are not (the *content* of the reasoning matters, not
whether it merely occurred). What the localization result implies for trace
faithfulness.

### 7. Limitations
Written honestly and specifically — see §6 below.

### 8. Conclusion

---

## 5. Figures and tables

| # | Type | Content | Source |
|---|------|---------|--------|
| F1 | Diagram | Pipeline: SOB record → extraction w/ trace → attribution → probe → regeneration | hand-drawn |
| F2 | Line | Probe AUROC vs layer, with baseline horizontals | `probes/_summary.json` |
| F3 | Bar + CI | `answer` / `fused_attr` / `fused_both` / three controls, per-doc AUROC w/ bootstrap CIs | `attribution_controls.json` |
| F4 | Line | **Risk–coverage curve**, all signals, `per_doc` regime | `selective_regeneration.json` |
| T1 | Table | Labeling modes: strict / auto / structure_aware error rates | `_definition_comparison.json` |
| T2 | Table | Layer robustness: Δ vs answer at layers 16/19/23/26 | `reasoning_attribution_lodo.json` |
| T3 | Table | Paired tests w/ Holm-corrected p and bootstrap CIs | `attribution_controls.json` |

**F3 is the paper's key figure.** It shows the claim and its three controls in
one frame — a reader should be able to see the argument without the text.
**F4 is the one a practitioner cares about.**

---

## 6. Limitations — draft them now, honestly

Writing these before the numbers land keeps them from being defensive.

1. **Single model, single benchmark.** One 7B reasoning model on one dataset.
   ExtractBench replication is directional only (~28 documents — too few for a
   paired test), and we should say so rather than dress it up.
2. **Modest effect size.** ~+0.02 AUROC. Consistent across layers and metrics,
   but small. Report it plainly; consistency and the controls are the argument,
   not magnitude.
3. **String matching is a coarse localizer.** A value the model reasoned about
   *by paraphrase* is scored as unmentioned. This is a floor on the method, and
   an obvious next step (embedding-based localization).
4. **Simulated regeneration.** Stage 09's cost-quality curve uses an explicit
   repair/damage model rather than actually re-running the model on flagged
   fields. State the parameters, report the parameter-free upper bound, and
   flag actual regeneration as future work.
5. **Post-hoc layer selection.** Layer 19 was chosen on the first run's CV. The
   effect holds at all four layers examined, which mitigates but does not
   eliminate this.
6. **Error labels inherit the matcher.** Structure-aware matching is a judgement
   call; we report all three modes so the reader can see the sensitivity.

---

## 7. What is NOT in this paper

Guard against scope creep in the final two weeks:
- CLAP (implemented, unstable, dropped — one sentence in future work at most).
- Ali's ExtractBench/Qwen results (his track; cite as companion work, do not
  absorb).
- Nested-LODO hyperparameter selection (Ali's `05b`; ours fixes C=1.0 — mention
  in limitations, do not run it now).

---

## 8. Immediate write-up order

1. §3 Method and §4 Setup — **fully result-independent, write these first.**
2. §2 Related work — also result-independent.
3. §5 Results — as jobs land; Update_08.md is the raw version of this.
4. §1 Intro and abstract — write last, once the numbers are known.

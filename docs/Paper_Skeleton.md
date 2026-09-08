# Paper Skeleton — Reasoning-Trace Trust Signals for Structured Extraction

**Status:** updated 2026-09-08 against the final 994-document results. The
version written on 2026-08-16 asserted a field-localization claim that the
controls later overturned; that framing has been removed rather than softened.
**Target:** NLP lab course deliverable, written to conference-submission standard.
**Every number below is final and traced to a result file** — no slots remain.

---

## 1. The claim ladder

Four claims, strongest evidence first. Each rung is independently defensible, so
a weak rung damages only itself.

| # | Claim | Evidence | Status |
|---|-------|----------|--------|
| C1 | Per-field extraction errors are **linearly decodable** from a reasoning model's hidden states, better than the model's own confidence. | Pooled LODO 0.7990 vs 0.7529 best log-prob baseline (+0.046) | **Done** |
| C2 | The **reasoning trace carries error signal beyond the answer token**. | `fused_decomposed` +0.0364, p=1.7e-05, Holm 1.2e-04, n=681 | **Done** |
| C3 | The gain is **document-level, not field-localized**, and is not a dimensionality artifact. | Controls: docmean matches fused_attr (p=0.71); tracemean matches docmean (p=0.96); shuffled and random both null | **Done** |
| C4 | The signal is **worth something**: it buys a better cost-quality tradeoff for selective regeneration than free baselines. | 49.0% of errors caught at a 20% budget vs 40.0%; error rate 33.2% -> 22.0% | **Done** |

**C3 is the paper's intellectual centre, and it is a negative-within-positive.**
We proposed field-localized attribution, built it, and found it works no better
than the crudest possible document-level summary. Two earlier conclusions were
reversed by scaling and controls. That is the contribution, and it should be
stated in the introduction rather than buried.

---

## 2. Title candidates

1. *Reasoning Traces Improve Error Detection in Structured Extraction — But Not Where You'd Expect*
2. *Document, Not Field: What a Reasoning Model's Chain of Thought Says About Its Own Errors*
3. *Trust Signals from Reasoning Traces in Structured Information Extraction*

(1) leads with the reversal, which is the most defensible framing. Keep the
supervisor's project title intact wherever the course requires it.

**Do not use** any title built on "field-localized" or "localizing" — that claim
did not survive the controls.

---

## 3. Abstract (final numbers, no slots)

> Large language models increasingly emit structured records, but expose no
> reliable per-field confidence, so downstream systems cannot tell which fields
> to trust. We train linear probes on the hidden states of a reasoning model
> (DeepSeek-R1-Distill-Qwen-7B) to predict per-field extraction errors on SOB, a
> multi-hop structured-output benchmark of 974 documents and 5,071 fields. Under
> leave-one-document-out the probe reaches **0.799 AUROC**, beating
> token-log-probability baselines by **0.046**. We then ask whether the model's
> explicit reasoning trace carries error-relevant signal beyond the answer
> token. It does: adding a pooled representation of the trace raises detection to
> **0.827**, a paired improvement of **+0.036 AUROC (p = 1.7e-05**, Holm-corrected
> across the control family**)**. Controls that hold feature dimensionality fixed
> while destroying content — reasoning vectors shuffled across fields, and
> scale-matched noise — are both null, so the gain is not an artifact of added
> capacity. However, the effect is **document-level rather than field-specific**:
> collapsing every field's reasoning vector to a single document mean performs
> identically (p = 0.71), and pooling the entire trace performs identically to
> pooling only the tokens where a field's value is mentioned (p = 0.96). A
> geometric analysis shows field-specific structure does exist (within-document
> cosine 0.61) but is not error-relevant. Applied to selective regeneration, the
> signal catches **49.0%** of errors within a 20% regeneration budget versus
> 40.0% for the best free baseline, cutting the field error rate from **33.2% to
> 22.0%**. We additionally report a structure-aware labeling correction that
> reduces an apparent 95% error rate to a genuine 43%, and a probe-free
> indicator: values only *partially* present in the reasoning trace are wrong
> **83%** of the time.

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
5.1 C1: probe vs baselines under LODO, plus the per-layer curve (labelled as an
optimistic diagnostic). 5.2 C2: the reasoning trace helps -- the variant table.
5.3 C3: the controls, and the two reversals they forced. 5.4 C4: selective
regeneration. 5.5 The mention analysis (probe-free indicator, with the caveat
that it is not a competitive detector).

### 6. Analysis / discussion
Why mid-to-late layers peak. Why the hand-made scalars are weak while the pooled
hidden states are not (the *content* matters, not merely that reasoning
occurred). Why decomposing the vector into document-mean and field-residual
blocks recovers 60% more effect than handing over their sum -- L2 over 7,168
correlated dimensions otherwise discards the small component, which is why the
original comparison could not have detected a field-level effect. The geometry
result: field-specific structure exists but is not error-relevant. What a
document-level effect implies for chain-of-thought faithfulness -- the trace
appears to encode something closer to "this record was hard" than per-field
evidence.

### 7. Limitations
Written honestly and specifically — see §6 below.

### 8. Conclusion

---

## 5. Figures and tables

| # | Type | Content | Source |
|---|------|---------|--------|
| F1 | Diagram | Pipeline: SOB record → extraction w/ trace → attribution → probe → regeneration | hand-drawn |
| F2 | Line | Probe AUROC vs layer, with baseline horizontals | `probes/_summary.json` |
| F3 | Bar + CI | 7 variants (`fused_decomposed`, `fused_attr`, and five controls), Δ vs answer w/ bootstrap CIs and Holm stars | merged from `decomposition_test.json` + `selection_test.json` + `attribution_controls.json` |
| F4 | Line | **Risk–coverage** + errors-caught, all signals, both budget regimes | `selective_regeneration_final.json` |
| T1 | Table | Labeling modes: strict / auto / structure_aware error rates | `_definition_comparison.json` |
| T2 | Table | Layer robustness: variants at layers 19 and 23 | `reasoning_attribution_merged.json` |
| T3 | Table | Paired tests w/ Holm-corrected p and bootstrap CIs | `decomposition_test.json` |

**F3 is the paper's key figure.** It puts the proposed method and every control
in one frame, all adding the same number of dimensions — so a reader can see,
without the text, that the two document-level grey bars match the coloured one
while shuffled and random sit at zero. That single image is the C3 argument.
**F4 is the one a practitioner cares about.**

---

## 6. Limitations

Volunteered rather than extracted. Items 5-7 come from the leakage audit in
`Update_08.md` §10.

1. **Single model, single benchmark.** One 7B reasoning model on one dataset.
   ExtractBench replication would be directional only (~28 documents — too few
   for a paired test), and we say so rather than dress it up.
2. **Modest effect size.** +0.036 AUROC. Highly significant and consistent
   across both layers tested, but small. Report it plainly; the controls and the
   mechanism are the argument, not magnitude. The field-specific residual is
   weaker still and reported as suggestive (+0.012 alone, p=0.19).
3. **String matching is a coarse localizer.** A value the model reasoned about
   *by paraphrase* is scored as unmentioned. This is a floor on the method, and
   an obvious next step (embedding-based localization).
4. **Simulated regeneration.** Stage 09's cost-quality curve uses an explicit
   repair/damage model rather than actually re-running the model on flagged
   fields. State the parameters, report the parameter-free upper bound, and
   flag actual regeneration as future work.
5. **Layer pre-commitment, not post-hoc selection — but with a caveat.** Layer 19
   was fixed from the 300-document pilot, whose documents are a *subset* of the
   final 994, so the choice saw data inside the evaluation set. Mitigations: the
   effect holds at layer 23 as well; the CV optimum on the full corpus is layer
   21, so we do not report the best-case layer; and every paired test compares
   variants at the *same* layer, so no variant gets a private choice.
6. **The per-layer CV curve is optimistic and must be labelled as such.** Its
   folds are stratified over fields rather than grouped by document, so
   within-document leakage inflates it. It appears only as a
   where-in-the-network diagnostic; every claim rests on leave-one-document-out,
   where the held-out document is fully masked and per-fold standardization is
   fitted on training rows only. The probe-vs-baseline comparison is made on the
   LODO out-of-fold scores, where both signals are ranked over an identical
   field set.
7. **`fused_decomposed` was not pre-registered.** It was introduced after
   observing that `fused_attr` matched the document-mean control, as a motivated
   fix to an identified defect in how features were presented to the probe —
   not as a search over variants. Stated plainly rather than implied.
8. **Error labels inherit the matcher.** Structure-aware matching is a judgement
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
3. §5 Results — all runs are complete; `Update_08.md` is the raw version of this
   section and every number is final.
4. §1 Intro and abstract — §3 above is the finished abstract; the intro should
   mirror the claim ladder and state the reversal up front.

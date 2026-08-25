# Probe‑Based Trust Signals for Structured Information Extraction
## Weekly Update 08 — DeepSeek‑R1 / Reasoning‑Trace Track

**Author:** Adnan
**Date:** 2026‑08‑25
**Scope:** The corpus was scaled from 300 to **994 documents**, and the
attribution result was put through a full set of controls. Three conclusions
from Update 07 changed. The track's central claim is now **stronger and more
significant than before**, but the *mechanism* behind it is not the one we
proposed — and the controls are what revealed that.

---

## 1. Headline

> A linear probe on DeepSeek‑R1's hidden states detects per‑field extraction
> errors at **0.799 AUROC** under leave‑one‑document‑out, beating the model's own
> token log‑probabilities by **+0.046**. Adding a representation of the reasoning
> trace raises this to **0.827** — a paired improvement of **+0.0364 AUROC,
> p = 1.7 × 10⁻⁵** (Holm‑corrected p = 1.2 × 10⁻⁴, n = 681 documents). Applied to
> selective regeneration, it catches **49.0 % of all errors within a 20 %
> regeneration budget** versus 40.0 % for the best free baseline, cutting the
> field error rate from **33.2 % to 22.0 %**.

---

## 2. What we set out to do

Update 07 reported field‑localized reasoning attribution at p = 0.044 on 300
documents — real, but a single borderline p‑value that would not survive a
strict multiple‑comparison correction. The plan was:

1. **Scale the corpus** to ~1,000 documents to settle the significance question.
2. **Run controls** that hold feature dimensionality fixed and destroy only the
   property we claim matters, so the gain cannot be dismissed as "you added
   3,584 more features".

Both are done. The scale‑up confirmed the effect. The controls changed our
explanation of it.

---

## 3. Corpus and setup

| | |
|---|---|
| Documents extracted | 994 (from 300) |
| Fields labeled | 6,132 — error rate 42.7 % (structure‑aware) |
| Fields used for probing | 5,071 in 974 documents, 33.2 % errors |
| Paired LODO folds | 681 |
| Values mentioned in the `<think>` trace | 86.1 % |

Labeling‑mode sensitivity is unchanged in character: strict 46.2 % / auto
42.9 % / **structure‑aware 42.7 %** (primary).

Extraction was made **resumable and shardable**, so the 700 new documents cost
only their own GPU time (4 h 19 m) and a wall‑clock kill loses at most one
document. Layer **19** remains the pre‑committed analysis layer throughout.

---

## 4. Result 1 — the probe works and beats free baselines (C1)

Per‑layer 5‑fold CV on the full corpus (Figure F2):

| Signal | AUROC |
|---|---:|
| Baseline: mean log‑prob | 0.750 |
| Baseline: min log‑prob | 0.753 |
| Probe — layer 12 | 0.753 |
| Probe — layer 16 | 0.792 |
| Probe — layer 19 *(pre‑committed)* | 0.804 |
| **Probe — layer 21 (CV peak)** | **0.806** |
| Probe — layer 28 | 0.784 |

The probe overtakes both baselines around **layer 12** and holds roughly
+0.05 AUROC from layer 16 onward. Under LODO the answer‑token probe reaches
**0.799 pooled**, versus 0.753 for the best baseline.

**Note on layer choice.** The CV peak on this corpus is layer 21 (0.806), not
layer 19 (0.804). The difference is 0.002 — well inside the CV standard
deviation of ≈0.010 — and layer 19 was **pre‑committed** from the 300‑document
pilot. We keep layer 19 everywhere rather than re‑select on the final corpus,
which is exactly the post‑hoc choice pre‑commitment exists to avoid.

---

## 5. Result 2 — the reasoning trace helps, significantly

Merged attribution table, LODO, 974 documents (per‑doc AUROC, pooled in
brackets):

| Variant | Layer 19 | Layer 23 | Δ vs answer (L19) | p |
|---|---:|---:|---:|---:|
| answer (baseline) | 0.7912 [0.7990] | 0.7608 [0.7670] | — | — |
| **fused_decomposed** | **0.8276 [0.8271]** | **0.8189 [0.8127]** | **+0.0364** | **1.7e‑05** |
| fused_both | 0.8154 [0.8254] | 0.7998 [0.8103] | +0.0242 | 0.0034 |
| fused_attr | 0.8140 [0.8248] | 0.8030 [0.8104] | +0.0227 | 0.0056 |
| fused_scalars | 0.7898 [0.8006] | 0.7598 [0.7695] | −0.0015 | 0.39 |
| scalars_only | 0.7166 [0.7008] | 0.7166 [0.7008] | −0.0746 | 6.4e‑07 |

Every reasoning variant improves on the answer‑only probe at **both** layers.
`fused_decomposed` is best on both metrics at both layers.

*(`scalars_only` is byte‑identical across layers, as it must be — the mention
scalars do not depend on layer. A useful correctness check that passed.)*

---

## 6. Result 3 — the controls, and what they overturned

Seven variants, all adding the **same 3,584 dimensions** to the same answer
block at the same layer, with only the *content* of the reasoning block
differing (Figure F3, Holm‑corrected over the family):

| Variant | Δ vs answer | 95 % CI | p (Holm) |
|---|---:|---|---:|
| **fused_decomposed** — doc mean + field residual | **+0.0364** | [+0.019, +0.055] | **0.00012** ✅ |
| ctrl_docmean — same vectors, one per document | +0.0263 | [+0.014, +0.040] | 0.00052 ✅ |
| ctrl_tracemean — **whole trace**, one per document | +0.0257 | [+0.013, +0.039] | 0.00049 ✅ |
| fused_attr — field‑localized | +0.0227 | [+0.006, +0.039] | 0.022 ✅ |
| ctrl_centered — field residual alone | +0.0124 | [−0.006, +0.030] | 0.56 |
| ctrl_shuffled — vectors on the wrong fields | −0.0083 | [−0.022, +0.005] | 0.36 |
| ctrl_random — scale‑matched noise | −0.0065 | [−0.026, +0.013] | 1.0 |

**Three findings, two of which reverse Update 07.**

**(a) It is not a dimensionality artifact.** `shuffled` and `random` are both
null. Adding 3,584 dimensions of *anything* buys nothing; the content matters.
This was the objection the controls were built to answer, and it is answered.

**(b) The gain is document‑level, not field‑localized.** ⚠️ *Reverses Update 07.*
Collapsing every field's attribution vector to a **single document mean**
performs as well as attributing the right reasoning to the right field
(`fused_attr` vs `ctrl_docmean`: Δ = −0.0036, p = 0.71). At 300 documents
`ctrl_docmean` looked null; that reading was underpowered.

**(c) Which tokens you pool does not matter either.** ⚠️ *Reverses Update 06.*
Pooling the **entire** `<think>` trace performs the same as pooling only the
value‑mentioning tokens (`docmean` vs `tracemean`: Δ = +0.0006, p = 0.96). And
`ctrl_tracemean` **is** Update 06's document‑level fusion experiment — at 994
documents it is strongly significant (p = 1.2 × 10⁻⁴). Update 06's "clean null"
was a power artifact: its point estimates were positive all along (+0.003 to
+0.013), we simply could not resolve them at 300 documents.

### Why `fused_attr` looked like the answer, and what fixed it

Write the attribution vector as `attr_i = m_d + r_i` — a document mean plus a
small field‑specific residual. `fused_attr` hands the probe **the sum**. Under
L2 over 7,168 correlated dimensions the optimizer rides the high‑variance shared
direction and the residual is regularized away, so `fused_attr ≈ ctrl_docmean`
*whether or not* `r_i` carries signal. That comparison could not have come out
otherwise.

Splitting the two into **separate blocks** — `fused_decomposed` — lets the probe
weight and regularize them independently, and recovers **60 % more effect**
(+0.0364 vs +0.0227) from exactly the same information.

The field‑specific residual is **weak but not nothing**: +0.0124 alone (p = 0.19),
+0.0101 on top of the document mean (p = 0.23), and +0.0236 against a
width‑matched noise control (p = 0.013, Holm over 7 tests = 0.052). Consistently
positive, individually below significance. We report it as suggestive.

### Geometry — why localization has so little to add

A diagnostic measured directly on the vectors, which rules out the trivial
explanation:

| | |
|---|---|
| Mean within‑document cosine between fields' attribution vectors | **0.613** |
| Residual‑to‑mean norm ratio | **0.793** |
| Fields whose value never appears in the trace (zero vector) | 13.9 % |

The vectors are **not** near‑identical — there is substantial field‑specific
structure (cosine 0.61, residual ≈ 79 % of the mean). So localization does not
fail by construction. The field‑specific variation simply is not very
error‑relevant. That is a mechanism, not a shrug.

---

## 7. Result 4 — what the signal is worth (C4)

Selective regeneration on the LODO out‑of‑fold scores (Figure F4), base error
rate 33.2 %:

| Signal | pooled AUROC | Errors caught @20 % (global) | AURC ↓ |
|---|---:|---:|---:|
| **probe_fused** (decomposed) | **0.8271** | **49.0 %** | **0.125** |
| probe_answer | 0.7990 | 46.5 % | 0.139 |
| min log‑prob | 0.7529 | 40.0 % | 0.155 |
| mean log‑prob | 0.7500 | 39.3 % | 0.155 |
| random | 0.5065 | 19.8 % | 0.270 |
| oracle | 1.0000 | 60.2 % | 0.048 |

**The practical numbers:**

- At a **20 % regeneration budget**, the probe catches **49.0 %** of all errors
  versus **40.0 %** for the best free baseline — **+9.0 points**. Precision among
  flagged fields is **0.81**.
- Simulated post‑regeneration error rate falls from **33.2 % to 22.0 %** (repair
  rate 0.7) — a **34 % relative reduction** for a 20 % spend.
- **Break‑even repair rate 0.01**: the spend pays for itself if the regenerator
  fixes even 1 % of what it touches.
- The same ordering holds in the per‑document budget regime (46.9 % vs 40.2 %,
  error rate → 22.9 %).

---

## 8. Result 5 — a probe‑free hallucination indicator

Pure string matching against the reasoning trace, no hidden states required:

| Field's value in the trace | n | Error rate |
|---|---:|---:|
| Fully mentioned | 4,161 | **28.9 %** |
| Never mentioned | 706 | **44.3 %** |
| **Only partially mentioned** | 204 | **83.3 %** |

Risk ratio (unmentioned vs mentioned) **1.41**, Fisher exact **p = 3.7 × 10⁻¹¹**.

A value that appears only *partially* in the reasoning — the model reasoned
about something adjacent but emitted something else — is wrong **83 % of the
time**. This is the single most interpretable result in the project.

**Honest caveat:** as a standalone *detector* these features reach only AUROC
0.717, **below** the 0.753 log‑prob baseline. They are a diagnostic and an
explanation, not a competitive scoring function, and we will present them that
way.

---

## 9. What changed since Update 07 — stated plainly

| Update 07 said | Update 08 finds |
|---|---|
| Field‑localized attribution improves error detection (p = 0.044) | The improvement is real and now p = 1.7e‑05 — but it is **document‑level**; per‑field assignment adds nothing over the document mean |
| Document‑level fusion is a clean null (Update 06) | That null was **underpowered**. At 994 documents it is significant (p = 1.2e‑04) |
| Localization is what makes the reasoning signal usable | **Decomposition** is what makes it usable: separating the shared and field‑specific components recovers 60 % more effect |

The three‑act narrative of Updates 06–07 does not survive scaling. What replaces
it is simpler and better evidenced: **the reasoning trace carries a
document‑level error signal that the answer token does not, and how you extract
it matters far less than that you extract it at all.**

---

## 10. Limitations

1. **Single model, single benchmark.** One 7B reasoning model on SOB.
2. **Modest effect size.** +0.036 AUROC — highly significant, but small.
3. **The residual result is suggestive, not established.** Positive in three
   independent measurements, individually below significance.
4. **Simulated regeneration.** The cost‑quality curve uses an explicit
   repair/damage model; we report the parameter‑free upper bound alongside it.
   Actually re‑running the model on flagged fields is future work.
5. **String matching is a coarse localizer** — a value reasoned about by
   paraphrase counts as unmentioned. A floor on the method, not a ceiling.
6. **Error labels inherit the matcher.** All three labeling modes are reported.

---

## 11. Next steps

1. **Write‑up.** Method and setup sections are result‑independent and can be
   drafted now; the results section follows this document.
2. **Window pooling** (optional, CPU‑only, ~1 day). We pool tokens where a value
   is *stated* — typically where the model records a conclusion. Pooling a
   window *around* each mention targets the reasoning instead. The geometry says
   field‑specific structure exists, so this is the remaining lever for a
   field‑level effect.
3. **Cross‑dataset replication** on ExtractBench with DeepSeek‑R1 — directional
   only; ~28 documents is too few for a paired test, and we will say so.

---

## 12. Artifacts

- `results/reasoning_attribution_merged.json` — the main table (§5)
- `results/decomposition_test.json` — controls + geometry + mentions (§6, §8)
- `results/selection_test.json` — the token‑selection test (§6c)
- `results/attribution_controls.json` — shuffled/random controls (§6a)
- `results/selective_regeneration_final.json` — cost‑quality curves (§7)
- `probes/_summary.json`, `results/comparison.json` — per‑layer CV + baselines (§4)
- `figures/F2_probe_by_layer`, `F3_controls`, `F4_selective_regeneration_*`
  (PDF + PNG + CSV of the plotted values)

## 13. Reproduce

```bash
sbatch run_sob_1k_extract_a100.sh        # GPU, resumable, --resume/--shard
sbatch run_sob_1k_analysis_a100.sh       # label -> attribution -> controls -> regen
sbatch run_sob_1k_selection_a100.sh      # Stages 03/04 + token-selection test
sbatch run_sob_1k_decomposition_a100.sh  # decomposition + geometry
sbatch run_sob_1k_stage07_fix_a100.sh    # fused_decomposed + merged table
python scripts/10_make_figures.py --config configs/exp_deepseek_r1_7b_sob_attr_1k.yaml
```

# Probe-Based Trust Signals for Structured Information Extraction

**Reasoning-trace track — DeepSeek-R1 on SOB.**

Can we tell which fields a language model got wrong when it extracted structured
data — and is knowing that actually worth anything?

This document is the overview: dataset, method, results, limitations, and how to
reproduce. For the full internals — every module, every design decision, and the
reasoning behind them — read [`DEEP_DIVE.md`](DEEP_DIVE.md).

> **Scope.** This repository is shared between two tracks. Everything documented
> here is the reasoning-trace track: `configs/exp_deepseek_r1_7b_sob_attr_1k.yaml`,
> `scripts/01`–`10`, `run_sob_*.sh`, and `src/probe_extraction/`. The partner
> track's experiment files have been removed from this branch; only the shared
> library remains, because `01_extract.py` imports its benchmark loaders.

---

## 1. The problem

LLMs turn documents into JSON records, and they get a lot of it wrong — **33.2%
of fields** on our corpus. Regenerating every field is expensive. If you knew
*which* fields were probably wrong, you could spend a small budget where it
counts.

Two questions follow, and most work answers only the first:

- **Detection** — can we predict, per field, whether the model got it wrong,
  using only what is available at inference time?
- **Value** — if we act on that prediction, does the record actually improve?

## 2. Contributions

1. **A reasoning-trace probe.** DeepSeek-R1 emits a `<think>...</think>`
   monologue before its answer. We probe the hidden states of that trace, not
   just the answer token, and beat the model's own token confidence by a wide
   margin (AUROC **0.827** vs **0.753**).
2. **Real, measured regeneration.** Prior work prices selective regeneration
   using an *assumed* repair rate. We ran the regeneration and measured it.
   The assumption (0.700) was **three times too high**; the measured rate is
   **0.233**. A simulation using the assumed value predicted a 22.9% error rate;
   the truth is 32.3%.
3. **Consensus is what makes regeneration safe.** A single resample damages more
   than it repairs (0.234 vs 0.215). Requiring the model to agree with itself
   lifts the repair-to-damage ratio from **0.9 to 4.8**.
4. **Two self-corrections.** Controls we ran against our own earlier claims
   refuted them, and the corrections are documented rather than quietly dropped.
   See [§8](#8-what-we-got-wrong-and-corrected).

---

## 3. Dataset

**SOB** (Structured Output Benchmark, `interfaze-ai/sob`, arXiv:2604.25359),
text / multi-hop subset, `test` split.

Each record pairs a **multi-hop question** and source context with a **per-record
JSON Schema** and a human-verified gold JSON answer. The text subset is derived
from HotpotQA, so answering genuinely requires combining facts across passages —
which is precisely why it suits a reasoning model whose `<think>` trace we want
to probe.

| | |
|---|---|
| Records requested | 1,000 |
| Extractions on disk | 997 (3 produced no parseable JSON) |
| **Documents analysed** | **974** |
| **Fields analysed** | **5,071** |
| **Errors** | **1,685 (33.2%)** |
| Domain | `sob/hotpotqa` |

A document enters the analysis only if it has per-token reasoning states at all
four capture layers and at least one labelled field the model actually emitted.

**Labelling.** Gold and extracted JSON are walked in parallel; each leaf gets a
correct/wrong label. Arrays of objects are paired positionally; arrays of
primitives are compared as sets; hallucinations and omissions both count as
errors. We label under three strictness definitions on every run and write the
comparison to `labels/_definition_comparison.json`.

The configured mode is **`structure_aware`**, which handles DeepSeek-R1's habit
of emitting a primitive where the schema wants an object (`"B. Boser"` instead
of `{"name": "B. Boser", ...}`). Under strict matching that single behaviour
produced a ~95% error rate — an artefact of the labeller, not a property of the
model. The honest rate is 33.2%.

---

## 4. Method

```
document + schema
        │
        ▼
  DeepSeek-R1-Distill-Qwen-7B     greedy, max_new_tokens 4096
        │
        ├──► <think> … </think>   ──►  per-token hidden states (layers 16,19,23,26)
        │                                        │
        └──► {"field": "value", …} ──►  answer-token hidden state per field
                     │                           │        (14 layers)
                     │                           ▼
                     │                 find where each field's VALUE is
                     │                 mentioned in the trace; pool those
                     │                 tokens  ──►  attr_vec
                     │                           │
                     ▼                           ▼
              label vs gold            [ answer ‖ doc-mean ‖ residual ]
                     │                           │
                     └───────────► linear probe ─┘
                                        │
                                  P(field is wrong)
                                        │
                        ┌───────────────┴───────────────┐
                        ▼                               ▼
              flag top-b by score              log-prob baselines
                        │                      (the free alternative)
                        ▼
              re-ask the model (k=3, T=0.7)
                        │
              swap flagged fields only
                        │
              re-label against gold  ──►  repaired / damaged / unchanged
```

**The probe** is logistic regression (`C=1.0`, `class_weight="balanced"`) — a
*linear* probe on purpose, so the result speaks about what is linearly decodable
from the representation rather than about the classifier's capacity.

**The feature set** (`fused_decomposed`) is the answer-token state concatenated
with the attribution vector **split into two blocks**: the document mean and the
field-specific residual. Handing the probe their *sum* lets L2 regularization
over 7,168 correlated dimensions crush the small residual — splitting them
recovered 60% more effect.

**Evaluation is leave-one-document-out** throughout. Fields inside one document
share a trace, a schema and a gold record, so a field-level split would let the
probe learn the document instead of the phenomenon.

---

## 5. Pipeline

| Stage | Script | Runs on | What it does |
|---|---|---|---|
| 0 | `00_download_sob.py` | login node | Caches SOB to disk (compute nodes have no internet). |
| 1 | `01_extract.py` | **GPU** | Extraction; captures answer-token hidden states, log-probs, and per-token `<think>` states. `--resume` / `--shard`. |
| 2 | `02_label.py` | CPU | Per-field correct/wrong labels under three strictness definitions. |
| 3 | `03_train_probe.py` | CPU | Per-layer probe. **Optimistic CV — diagnostic only.** |
| 4 | `04_evaluate.py` | CPU | Probe vs token-log-prob baselines. |
| 5 | `05_reasoning_attribution_lodo.py` | CPU | The main experiment: LODO over feature variants + paired significance. |
| 6 | `06_attribution_controls.py` | CPU | Controls, Holm correction, bootstrap CIs, geometry diagnostic, mention analysis. |
| 7 | `07_selective_regeneration_sob.py` | CPU | Risk–coverage curves (**simulated** repair). Writes the per-field OOF scores Stage 9 ranks by. |
| 8 | `08_regenerate.py` | **GPU** | **Real** regeneration: re-asks the model k times at T > 0. |
| 9 | `09_regen_evaluate.py` | CPU | Swaps regenerated values into flagged fields, re-labels, counts what actually happened, bootstraps the result. |
| 10 | `10_make_figures.py` | CPU | Figures as PDF + PNG + CSV of plotted values. |

Every stage runs in number order and nothing is skipped. Two earlier stages that
Stage 5 replaced are kept in [`archive/scripts/`](../archive/) rather than in the
live pipeline.

---

## 6. Results

### 6.1 Detection — pooled LODO AUROC, layer 19

| Signal | AUROC |
|---|---|
| `random` | 0.5065 |
| `mean_logprob` | 0.7500 |
| `min_logprob` | 0.7529 |
| Probe, answer token only | 0.7990 |
| **Probe + reasoning (`fused_decomposed`)** | **0.8271** |

**+0.074 over the best free baseline.**

Per-document AUROC: **0.8276** at layer 19, 0.8189 at layer 23.
Versus answer-only: **Δ +0.0364, Wilcoxon p = 1.7 × 10⁻⁵** (Holm 1.2 × 10⁻⁴),
over 681 documents with a valid paired fold.

| Variant | Δ vs answer | p |
|---|---|---|
| **`fused_decomposed`** | **+0.0364** | **1.7 × 10⁻⁵** |
| `fused_both` | +0.0242 | 0.0034 |
| `fused_attr` | +0.0227 | 0.0056 |
| `fused_scalars` | −0.0015 | ns |
| `scalars_only` | −0.0746 | — |

### 6.2 Controls — the effect is document-level, not field-localized

Every control holds dimensionality fixed and varies only content.

| Comparison | Δ | p |
|---|---|---|
| `ctrl_docmean` vs answer | +0.0263 | 5.2 × 10⁻⁴ |
| `ctrl_tracemean` vs answer | +0.0257 | 4.9 × 10⁻⁴ |
| `ctrl_shuffled` vs answer | −0.0083 | ns |
| `ctrl_random` vs answer | −0.0065 | ns |
| `fused_attr` vs `ctrl_docmean` | −0.0036 | **0.71** |
| `ctrl_docmean` vs `ctrl_tracemean` | +0.0006 | **0.96** |
| `fused_decomposed` vs width-matched noise | +0.0236 | 0.013 |

Shuffled and random are null, so the gain is not a dimensionality artefact. But
collapsing the attribution vectors to one per document loses nothing — so the
reasoning signal is a **document-level** property of the trace, not a
field-localized one.

**Geometry** explains why: mean within-document cosine between fields'
attribution vectors is 0.6132, the residual is 0.7934 of the mean's norm, and
13.9% of fields have a zero attribution vector. Field structure exists; it is
just not error-relevant.

### 6.3 Mention analysis

| Value found in the trace | n | error rate |
|---|---|---|
| fully | 4,161 | 28.9% |
| never | 706 | 44.3% |
| **partially** | 204 | **83.3%** |

Risk ratio 1.41, Fisher exact **p = 3.7 × 10⁻¹¹**.

**Caveat, always reported with it:** a probe on these 7 scalars alone reaches
0.7166 — *below* `min_logprob` at 0.7529. It is an interpretable diagnostic, not
a competitive detector.

### 6.4 The measured repair and damage rates

997 documents regenerated, 3 samples each at temperature 0.7.
2,908 of 2,922 resamples usable (1 truncated, 13 unparseable).

| Strategy | Repair | Damage | Ratio | Best error rate | Regenerate everything |
|---|---|---|---|---|---|
| **Assumed by Stage 7** | 0.700 | 0.050 | 14.0 | — | — |
| `first` (k = 1) | 0.215 | 0.234 | 0.9 | 32.4% | 42.0% |
| **`vote` (plurality of 3)** | **0.233** | **0.167** | **1.4** | **31.6%** | 37.0% |
| `vote_strict` (≥2 agree) | 0.385 | 0.080 | 4.8 | 31.4% | 32.5% |

**Control — no probe at all:** replacing each record wholesale with its resample
gives **44.1%** against the original's **33.2%**. The second attempt is markedly
worse than the first; selectivity is the only reason regeneration helps.

**60% of resampled fields came back byte-identical** — at T = 0.7 the model
simply repeats itself more often than not.

### 6.5 Does selective regeneration actually help? — measured, with intervals

2,000 document-level bootstrap replicates, paired across signals,
Holm-corrected within each budget and regime. `vote` strategy, global budget,
20% of fields:

| Quantity | Estimate | 95% CI | Holm p |
|---|---|---|---|
| **Error-rate reduction** | **−1.59 pts** (33.2% → 31.6%) | [0.93, 2.22] | **< 0.001** |
| vs `min_logprob` | +1.34 pts | [0.67, 2.04] | **< 0.001** |
| vs `mean_logprob` | +1.56 pts | [0.86, 2.26] | **< 0.001** |
| vs answer-only probe | +0.37 pts | [−0.18, +0.92] | 0.19 **(ns)** |

The reduction is also significant for `vote` at a 10% budget
(−0.77 pts, [0.44, 1.11]) and in the per-document regime
(−0.86 pts, [0.17, 1.58], Holm p = 0.046), and for `vote_strict` in **all four**
tested configurations. A single resample (`first`) is *not* reliably beneficial:
in the per-document regime at 20% it is −0.32 points with a CI spanning zero.

**The honest negative:** the reasoning-trace features win clearly on detection
but do **not** produce a significant downstream error-rate gain over answer-token
features. AUROC advantages compress when passed through a weak repair operator.

### 6.6 Simulated versus measured — the headline comparison

| | Simulated (repair assumed 0.7) | **Measured** |
|---|---|---|
| Repair rate | 0.700 | **0.233** |
| Damage rate | 0.050 | **0.167** |
| Error rate after regeneration, per-doc @ 20% | 22.9% | **32.3%** |

**The assumption overstates the benefit by 9.4 percentage points** on the same
corpus, the same folds and the same fields.

### 6.7 The ceiling

With **perfect** error detection (an oracle using gold labels), the error rate
bottoms out at **26.5%**. The probe reaches 31.4–31.6%. The distance from 33.2%
to 26.5% is what better detection could buy; everything below 26.5% requires a
better repair operator, not a better detector.

---

## 7. Validation

| Check | Result |
|---|---|
| **Additivity** — per-field arithmetic vs an independent joint re-labelling | **Exact agreement (+0)** at 10%, 20% and 100% budgets, for all three strategies |
| **Self-check** — re-label the untouched originals and compare to stored labels | **All 5,071 labels reproduced** |
| **Coverage guard** | Stage 9 refuses to run if any scored document lacks a regeneration |
| **Unlabelable fields** | 3 of 5,071 (`RecursionError`), excluded from denominators and reported |
| **Leakage** | LODO masks the held-out document entirely; standardization fitted on the training fold only; features are model outputs, never gold |
| **Tests** | 271 passing |

Gold is used only to *label* outcomes, never to *select* what to regenerate.
`oracle` is a labelled ceiling and is always presented as such.

**Disclosed weaknesses:** Stage 3's cross-validation is field-stratified, not
document-grouped — it is leaky and diagnostic only, with a warning in the
docstring. Layer 19 was pre-committed from a 300-document pilot that is a subset
of the final corpus (the effect holds at layer 23, and the CV optimum is layer
21, so we are not reporting a best case). `fused_decomposed` was not
pre-registered; it is a motivated fix to an identified defect.

---

## 8. What we got wrong, and corrected

**The "clean null" was a power artefact.** An earlier report claimed
document-level reasoning fusion was null. The point estimates were positive all
along (+0.003 to +0.013); at 300 documents the corpus could not resolve them. At
994 the same test gives p = 1.2 × 10⁻⁴.

**Field localization does not work as a mechanism.** An earlier report claimed
the gain came from attributing reasoning to individual fields. Controls refute
it: collapsing the attribution vectors to one per document changes nothing
(p = 0.71), and pooling the whole trace instead of the value-mention tokens also
changes nothing (p = 0.96). What survives is that the reasoning trace carries
error-relevant information the answer token does not — as a document-level
property.

Both corrections came from controls designed to kill our own claims.

---

## 9. Limitations

1. One model, one dataset, one temperature (T = 0.7, never tuned).
2. The reasoning-trace advantage does not reach significance downstream
   (p = 0.19), only on detection.
3. The absolute improvement is 1.6 points — real and significant, but modest.
4. `vote_strict` is post-hoc, designed after seeing the damage numbers. The
   pre-planned strategy is `vote`, which is significant on its own.
5. Most errors are not repairable by resampling: 60% of resamples are identical,
   and the oracle ceiling is 26.5%.
6. Stage 3 CV leakage and the layer-19 pre-commitment, as in §7.

**The clearest next step** follows from the oracle ceiling: the bottleneck is the
repair operator, not the detector. Lower temperature, targeted re-prompting that
names the suspect field, or retrieval instead of blind resampling.

---

## 10. Repository layout

| Path | Contents |
|---|---|
| `src/probe_extraction/` | The library: config, data loaders, extraction, labeling, probes, baselines, regeneration measurement |
| `src/probe_extraction/regen/` | The repair/damage accounting (in the package so the parallel pass can import it by name) |
| `src/probe_extraction/utils/jsonpath.py` | Leaf addressing for building hybrid records |
| `scripts/` | Numbered pipeline stages, run in order |
| `configs/exp_deepseek_r1_7b_sob*.yaml` | This track's experiment configs |
| `tests/` | 271 tests, CPU-only |
| `docs/` | This file, `DEEP_DIVE.md`, weekly updates 01–08, paper skeleton |
| `run_sob_*.sh` | SLURM job scripts |
| `artifacts/`, `data/`, `logs/` | Gitignored — large and regenerable |

---

## 11. Reproducing

Dataset first, from a **login node** (compute nodes have no internet):

```bash
python scripts/00_download_sob.py
```

Extraction and analysis:

```bash
sbatch run_sob_1k_extract_a100.sh        # GPU, resumable; ~4h20m for 994 docs
sbatch run_sob_1k_analysis_a100.sh       # label -> attribution -> controls
sbatch run_sob_1k_selection_a100.sh      # Stages 3/4 + token-selection test
sbatch run_sob_1k_decomposition_a100.sh  # decomposition test + geometry
sbatch run_sob_1k_variants_a100.sh    # fused_decomposed + merged results
python scripts/10_make_figures.py --config configs/exp_deepseek_r1_7b_sob_attr_1k.yaml
```

Real regeneration (Stages 8–9):

```bash
sbatch run_sob_regenerate_a100.sh 1/2    # GPU, ~13h total, so split in two
sbatch run_sob_regenerate_a100.sh 2/2
sbatch run_sob_regen_eval.sh             # CPU: Stage 7 rescore + Stage 9
```

Config: `configs/exp_deepseek_r1_7b_sob_attr_1k.yaml`. Per-token reasoning
capture is enabled by environment variables set inside the extract script
(`REASONING_TOKEN_LAYERS="16,19,23,26"`, `REASONING_TOKEN_CAP="2048"`); without
them Stage 5 has nothing to attribute and refuses to run.

**Disk:** per-token reasoning states dominate — roughly 59 MB per document
before compression, so budget 50 GB for 1,000 documents. The extract script
checks free space before starting.

Tests:

```bash
PYTHONPATH=$PWD/src python -m pytest tests/ -q --ignore=tests/test_extract_bench.py
```

`test_extract_bench.py` is excluded because it needs PyMuPDF, which belongs to
the partner track's PDF benchmark.

---

## 12. Result files

Under `artifacts/deepseek_r1_7b_sob_attr/`:

| File | Contents |
|---|---|
| `results/reasoning_attribution_merged.json` | Main variant table (layers 19 and 23) |
| `results/decomposition_test.json` | Controls, geometry, mention analysis |
| `results/selection_test.json` | Whole-trace vs value-mention token selection |
| `results/attribution_controls.json` | Shuffled / random controls |
| `results/selective_regeneration.json` | Cost–quality curves (simulated repair) |
| `results/oof_field_scores.json` | Per-field out-of-fold scores for every signal |
| **`results/regen_evaluation.json`** | **Measured repair/damage rates, budget curves, bootstrap CIs** |
| `probes/_summary.json`, `results/comparison.json` | Per-layer CV and baselines |
| `labels/_definition_comparison.json` | Error rate under all three matchers |

---

## 13. Superseded documents

`Project_Master_Guide.md`, `Update_06.md` and `Update_07.md` carry banners: they
teach the refuted field-localization mechanism and their results tables are out
of date. The Master Guide's pipeline, dataset and example prose remain accurate.
Current explainers are this file and `DEEP_DIVE.md`.

# Reasoning-Trace Track — DeepSeek-R1 on SOB

Entry point for the `adnan-dev` half of the project. The partner track
(Qwen3.5 on ExtractBench) lives on `main`; this document covers only the
reasoning-model work and the files that belong to it.

**Research question.** A reasoning model writes an explicit chain of thought
before answering. Does that trace carry information about which extracted
fields are wrong, beyond what the answer token already tells us — and can a
linear probe read it?

**Answer.** Yes, and the effect is document-level rather than field-specific.
Full write-up in [`Update_08.md`](Update_08.md).

---

## Headline results

Corpus: SOB text/multi-hop, 994 documents extracted, 974 documents /
5,071 fields / 33.2 % error rate used for probing, 681 paired LODO folds.

| Signal | Pooled LODO AUROC |
|---|---:|
| Baseline: mean log-prob | 0.7500 |
| Baseline: min log-prob | 0.7529 |
| Probe, answer token only | 0.7990 |
| **Probe + reasoning (`fused_decomposed`)** | **0.8271** |

Paired improvement of the reasoning representation over answer-only:
**+0.0364 AUROC, p = 1.7 × 10⁻⁵** (Holm-corrected 1.2 × 10⁻⁴, n = 681).

Selective regeneration at a 20 % budget catches **49.0 %** of all errors versus
40.0 % for the best free baseline, cutting the field error rate from
**33.2 % to 22.0 %**. Break-even repair rate 0.01.

**All reported numbers are leave-one-document-out.** See
[`Update_08.md` §10](Update_08.md) for the leakage audit, including the one
place where an optimistic number is reported (the per-layer CV curve) and why
it is never used as a headline.

---

## Repository layout — what belongs to this track

The repository is shared between both tracks, so the root holds run scripts for
each. These are the ones for the reasoning-trace work; everything else at the
root belongs to the partner track (`run_analysis_*.sh`, `run_extraction_*.sh`,
`run_deepseek_*.sh`) or is generic setup.

| Path | Contents |
|---|---|
| `src/probe_extraction/` | The library: config, data loaders, extraction, labeling, probes, baselines |
| `scripts/` | Numbered pipeline stages, run in order |
| `configs/exp_deepseek_r1_7b_sob*.yaml` | This track's experiment configs |
| `tests/` | 177 tests, CPU-only |
| `docs/` | Weekly updates 01–08, this file, paper skeleton, briefing |
| `run_sob_1k_*.sh` | **The current run scripts** — the 994-document pipeline |
| `run_sob_attr_*.sh`, `run_sob_*_a100.sh` | Earlier 300-document runs, kept for reproducibility |
| `figures/` | Generated figures (PDF + PNG + CSV of plotted values) |
| `artifacts/`, `data/`, `logs/` | Gitignored — large and regenerable |

If you only want to reproduce the final results, you need the five
`run_sob_1k_*.sh` scripts and nothing else at the root.

---

## Pipeline

| Stage | Script | Runs on | What it does |
|---|---|---|---|
| 1 | `scripts/01_extract.py` | GPU | Extraction; captures hidden states, log-probs, and the `<think>` trace. `--resume` / `--shard` make long runs restartable. |
| 2 | `scripts/02_label.py` | CPU | Per-field correct/wrong labels under three matching strictnesses. |
| 3 | `scripts/03_train_probe.py` | CPU | Per-layer probe. **Optimistic CV — diagnostic only** (see the docstring in `probes/linear.py`). |
| 4 | `scripts/04_evaluate.py` | CPU | Probe vs token-log-prob baselines. |
| 7 | `scripts/07_reasoning_attribution_lodo.py` | CPU | The reasoning experiment: LODO over feature variants + paired significance. |
| 8 | `scripts/08_attribution_controls.py` | CPU | Controls, Holm correction, bootstrap CIs, geometry diagnostic, mention analysis. |
| 9 | `scripts/09_selective_regeneration_sob.py` | CPU | Risk–coverage and cost–quality curves. |
| 10 | `scripts/10_make_figures.py` | CPU | Figures F2/F3/F4 as PDF + PNG + CSV of plotted values. |

Stages 5 and 6 (`05_lodo_cv.py`, `06_reasoning_fusion_lodo.py`) are earlier
versions, superseded by Stage 7 — see "Superseded" below.

### Feature variants compared in Stage 7

| Variant | Input to the probe |
|---|---|
| `answer` | Answer-token hidden state only (3,584 dims) |
| `fused_attr` | answer + attribution vector |
| `fused_decomposed` | answer + attribution vector **split** into document-mean and field-residual blocks — **the proposed method** |
| `fused_scalars` | answer + 7 interpretable mention scalars |
| `fused_both` | answer + attribution vector + scalars |
| `scalars_only` | the 7 scalars alone |

Controls in Stage 8 (`ctrl_docmean`, `ctrl_tracemean`, `ctrl_centered`,
`ctrl_shuffled`, `ctrl_random`, `ctrl_docmean_pad`) all add the *same* number of
dimensions and vary only the content, so a gain cannot be attributed to width.

---

## Reproducing

Dataset first, from a login node (compute nodes have no internet):

```bash
python scripts/00_download_sob.py
```

Then, in order:

```bash
sbatch run_sob_1k_extract_a100.sh        # GPU, resumable; ~4h20m for 994 docs
sbatch run_sob_1k_analysis_a100.sh       # label -> attribution -> controls -> regen
sbatch run_sob_1k_selection_a100.sh      # Stages 3/4 + the token-selection test
sbatch run_sob_1k_decomposition_a100.sh  # decomposition test + geometry diagnostic
sbatch run_sob_1k_stage07_fix_a100.sh    # fused_decomposed + merged results table
python scripts/10_make_figures.py --config configs/exp_deepseek_r1_7b_sob_attr_1k.yaml
```

Config: `configs/exp_deepseek_r1_7b_sob_attr_1k.yaml`.
Per-token reasoning capture is enabled by env vars set inside the extract
script (`REASONING_TOKEN_LAYERS`, `REASONING_TOKEN_CAP`); without them Stage 7
has nothing to attribute and will refuse to run.

Tests (177, no GPU needed):

```bash
PYTHONPATH=$PWD/src python -m pytest tests/ -q --ignore=tests/test_extract_bench.py
```

`test_extract_bench.py` is excluded because it needs PyMuPDF, which is only
used by the partner track's PDF benchmark.

---

## Result files

Under `artifacts/deepseek_r1_7b_sob_attr/`:

| File | Contents |
|---|---|
| `results/reasoning_attribution_merged.json` | Main variant table (layers 19 and 23) |
| `results/decomposition_test.json` | Controls, geometry, mention analysis |
| `results/selection_test.json` | Whole-trace vs value-mention token selection |
| `results/attribution_controls.json` | Shuffled / random controls |
| `results/selective_regeneration_final.json` | Cost–quality curves |
| `probes/_summary.json`, `results/comparison.json` | Per-layer CV and baselines |
| `labels/_definition_comparison.json` | Error rate under all three matchers |

Artifacts are gitignored (large, regenerable). `figures/` is committed.

---

## Superseded — kept for the record, do not cite

- **Stage 6 document-level fusion was reported as a null.** It is not. At 994
  documents the same comparison is +0.0257, p = 1.2 × 10⁻⁴. The null was a
  power artifact at 300 documents.
- **Field-localized attribution was reported as the mechanism.** It is not.
  The document mean of the same vectors performs identically
  (Δ = −0.0036, p = 0.71).
- **`05_lodo_cv.py`** duplicates the `answer` variant already computed inside
  Stages 6/7.
- **CLAP** (`probes/clap.py`, `06_train_clap.py`, `07_lodo_clap.py`,
  `run_clap.sh`) was implemented but never trained stably — no usable result was
  produced. Kept because revisiting it means debugging training, not comparing
  scores.

Full history, including which numbers are fragile and why, is in
`Update_01.md` … `Update_08.md`. Updates 01–03 are inherited copies of the
partner track's early reports; 04 onward are this track's own.

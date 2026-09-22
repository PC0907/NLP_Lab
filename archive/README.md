# Archive

Work that is no longer part of the live pipeline but is kept because a
documented result came from it. Nothing here is needed to reproduce the final
numbers — see [`docs/PROJECT_OVERVIEW.md`](../docs/PROJECT_OVERVIEW.md) for
that — and nothing in the live pipeline imports from this directory.

## Stage renumbering (2026-09-23)

The pipeline used to run 1, 2, 3, 4, 7, 8, 9, 10, 11, 12 — the gaps were the two
stages that Stage 5 replaced. The live stages have been renumbered to run
contiguously. **Any document dated before 2026-09-23 uses the old numbers.**

| Old | New | Script |
|---|---|---|
| 1 | 1 | `01_extract.py` |
| 2 | 2 | `02_label.py` |
| 3 | 3 | `03_train_probe.py` |
| 4 | 4 | `04_evaluate.py` |
| 5 | — | `05_lodo_cv.py` → `archive/scripts/` |
| 6 | — | `06_reasoning_fusion_lodo.py` → `archive/scripts/` |
| 7 | **5** | `05_reasoning_attribution_lodo.py` |
| 8 | **6** | `06_attribution_controls.py` |
| 9 | **7** | `07_selective_regeneration_sob.py` |
| 11 | **8** | `08_regenerate.py` |
| 12 | **9** | `09_regen_evaluate.py` |
| 10 | 10 | `10_make_figures.py` (now genuinely last) |

The weekly updates (`docs/Update_*.md`), `Project_Master_Guide.md` and
`Meeting_Briefing_2026-08-25.md` were deliberately **not** rewritten: they are
dated records, and renumbering stages inside them would falsify what was
actually done at the time. Use the table above to translate.

## What is here

### `scripts/`

| File | Why it is archived |
|---|---|
| `05_lodo_cv.py` | Plain leave-one-document-out cross-validation over layers. Superseded by Stage 5, which does the same evaluation across feature variants with paired significance testing. |
| `06_reasoning_fusion_lodo.py` | Document-level reasoning fusion — pools the whole `<think>` trace into one vector per document. This produced Update 06's "clean null", which later turned out to be a power artefact: the point estimates were positive all along (+0.003 to +0.013) but 300 documents could not resolve them. At 974 the same test gives p = 1.2 × 10⁻⁴. Kept because that reversal is part of the paper's argument, and `tests/test_reasoning_fusion.py` still exercises it. |

### `run/`

Earlier SLURM job scripts: the 300-document attribution runs (`run_sob_attr_*`),
the first SOB runs (`run_sob_extract*`, `run_sob_analysis*`, `run_sob_finish*`,
`run_sob_stage06*`), and the original ExtractBench runs with DeepSeek
(`run_deepseek_*`). **These reference the old stage numbers** and would need the
mapping above applied before they would run today.

### `configs/`

| File | Used by |
|---|---|
| `exp_deepseek_r1_7b_sob.yaml` | The first SOB run |
| `exp_deepseek_r1_7b_sob_attr.yaml` | The 300-document attribution run, superseded by the `_1k` config |
| `exp_deepseek_r1_7b_pooled.yaml` | The early ExtractBench run, before the track moved to SOB |

### `Problem_with_Dataset.md`

A May 2026 evaluation of the RealKIE benchmark and why two of its datasets were
rejected for this pipeline. Belongs to the earlier, shared phase of the project;
kept because it records the dataset-suitability criteria that were established.

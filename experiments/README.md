# Experiments

ExtractBench-track follow-up studies built on top of the shared pipeline in [`scripts/`](../scripts/). Each script reads cached artifacts
produced by stages 01–02 and writes results under `artifacts/<experiment>/results/`. Run them from the repo root:

```bash
python experiments/<script>.py --help
```

| Script | Question |
|---|---|
| `nested_safe_override.py` | Fully nested safe-override regeneration: does probe-guided replacement give a net gain? |
| `grid_probe.py` | Nested-LODO probe quality across the model × PDF-parser grid |
| `mass_mean_probe.py` | Mass-mean probes vs. logistic regression under identical nested LODO |
| `token_position_probe.py` | Where in a field's token span does the error signal live? |
| `transfer_probe.py` | Cross-dataset transfer: train on ExtractBench, apply to insurance claims |
| `transfer_sob.py` | Zero-shot cross-task transfer between ExtractBench and SOB |
| `steering_smoke.py` | Does the activation-steering hook work, and does the JSON survive? |
| `fit_steering_direction.py` | Fit the probe direction used for steering |
| `steering_experiment.py` | Is the probe direction causally involved in producing extraction errors? |
| `steering_intersection.py` | Recompute steering error rates over a fixed document set |
| `tenkq_recovery.py` | How many 10-K/Q fields are legitimately answerable from the parsed text? |
| `build_pooled_dataset.py` | Build a pooled four-domain experiment directory (Qwen) |
| `build_pooled_dataset_llama.py` | Same, for Llama 3.1 8B |
| `build_lasttoken_2dom.py` | Last-token dataset restricted to a fixed 8-document subset |
| `run_all_tokens.py` | Check that all-token extraction saved multi-token activations |

The matching SLURM launchers are in [`slurm/`](../slurm/) (e.g. `run_nested_regen.sh`, `run_grid_probe.sh`,
`run_transfer_full.sh`, `run_mass_mean.sh`).

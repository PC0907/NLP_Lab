# SLURM jobs

Job scripts used to run every experiment on the cluster. Submit from the repo root, since log paths
(`logs/…`) and config paths are relative to it:

```bash
mkdir -p logs
sbatch slurm/extractbench/run_extraction.sh
sbatch slurm/reasoning/run_sob_1k_extract_a100.sh
```

## Shared

- `setup_env.sh` / `setup_env_a100.sh`: load modules and activate the virtualenv. The ExtractBench jobs source one
  of them. Edit the module versions and venv path for your cluster.
- `install_a100_env.sh`: one-time environment install on an A100 node.

## `extractbench/`

- `run_extraction*.sh`: stage 1 (GPU extraction), sometimes followed by stages 2–4.
- `run_analysis*.sh`: stages 2–4 only (CPU).
- `launch_grid.sh` → `run_grid_cell.sh`: fan out the model × parser grid.
- Other `run_*.sh` scripts launch the matching script in [`experiments/`](../experiments/) or [`tools/`](../tools/).

## `reasoning/`

- `run_sob_1k_*.sh`: extraction and analysis on the 1,000-document SOB corpus (extraction is resumable and
  shardable).
- `run_sob_regenerate_a100.sh`, `run_sob_regen_eval.sh`: measured regeneration (stages 8–9).

These jobs activate `~/nlp_lab_a100` directly rather than sourcing a shared setup script. See
[`docs/reasoning/REASONING_TRACK.md`](../docs/reasoning/REASONING_TRACK.md) for the run order.

Scripts assume the repo is cloned at `~/NLP_Lab`. See [`docs/hpc.md`](../docs/hpc.md) for the full cluster guide.

# SLURM jobs

Job scripts used to run every experiment on the cluster. Submit from the repo root, since log paths
(`logs/…`) and config paths are relative to it:

```bash
mkdir -p logs
sbatch slurm/run_extraction.sh
```

- `setup_env.sh` / `setup_env_a100.sh`: load modules and activate the virtualenv. Every job sources one of them.
  Edit the module versions and venv path for your cluster.
- `install_a100_env.sh`: one-time environment install on an A100 node.
- `run_extraction*.sh`: stage 1 (GPU extraction), sometimes followed by stages 2–4.
- `run_analysis*.sh`: stages 2–4 only (CPU).
- `launch_grid.sh` → `run_grid_cell.sh`: fan out the model × parser grid.
- Other `run_*.sh` scripts launch the matching script in [`experiments/`](../experiments/) or [`tools/`](../tools/).

Scripts assume the repo is cloned at `~/NLP_Lab`. See [`docs/hpc.md`](../docs/hpc.md) for the full cluster guide.

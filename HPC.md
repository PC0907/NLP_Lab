# HPC Cluster Guide — NLP Lab on Bender

Practical guide for running the probe-based trust signals project on the University of Bonn Bender HPC cluster. Written for future reference and onboarding new chat sessions.

---

## 1. Cluster Overview

**Cluster:** Bender (bender.hpc.uni-bonn.de)
**Username:** s44srizv
**Scheduler:** SLURM
**Available GPUs:**

| Partition | GPU | VRAM | Max Time | Use Case |
|-----------|-----|------|----------|----------|
| A40devel | A40 | 48 GB | 1 hour | Quick tests, debugging |
| A40short | A40 | 48 GB | 8 hours | Standard extraction runs |
| A40medium | A40 | 48 GB | 1 day | Long multi-domain runs |
| A100short | A100 | 40/80 GB | 8 hours | Large models (Llama-3.1-8B+) |
| A100medium | A100 | 40/80 GB | 1 day | Overnight runs |

**No SLURM account needed on Bender** (unlike the Marvin cluster).

---

## 2. Connecting

### SSH
```bash
ssh s44srizv@bender.hpc.uni-bonn.de
```

### Faster SSH (add to `~/.ssh/config` on laptop)
```
Host bender
    HostName bender.hpc.uni-bonn.de
    User s44srizv
    ServerAliveInterval 60
```
Then just: `ssh bender`

The `ServerAliveInterval 60` sends a keepalive every 60 seconds, preventing idle disconnects.

---

## 3. Environment Setup

### First-time setup (already done)
```bash
module load Python/3.12.3
python -m venv ~/nlp_lab
source ~/nlp_lab/bin/activate
module load CUDA/12.4.0
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
cd ~/NLP_Lab
pip install -r requirements.txt
pip install -e . --no-deps
```

### Every SSH session
If `~/.bashrc` has the module loads (recommended), nothing to do. Otherwise:
```bash
module load Python/3.12.3
module load CUDA/12.4.0
source ~/nlp_lab/bin/activate
cd ~/NLP_Lab
```

### Recommended `.bashrc` additions
```bash
module load Python/3.12.3
module load CUDA/12.4.0
source ~/nlp_lab/bin/activate
```

### Verify environment is working
```bash
which python    # should show: ~/nlp_lab/bin/python
which pip       # should show: ~/nlp_lab/bin/pip
python -c "import torch; print(torch.cuda.is_available())"  # True (on compute nodes only)
python -c "from probe_extraction.config import load_config; print('OK')"
```

**Important:** `torch.cuda.is_available()` returns `False` on login nodes (no GPU). This is normal. It works on compute nodes when you submit a job.

---

## 4. Project File Layout

```
/home/s44srizv/
├── NLP_Lab/                    ← project repo (git-cloned from GitHub)
│   ├── configs/                ← experiment YAML configs
│   │   ├── exp_qwen35_4b_pymupdf.yaml
│   │   ├── exp_qwen35_4b_swimming.yaml
│   │   ├── exp_qwen35_4b_docling.yaml
│   │   └── exp_llama31_8b.yaml
│   ├── scripts/                ← pipeline stage scripts
│   │   ├── 01_extract.py       ← Stage 1: LLM extraction + activation capture
│   │   ├── 02_label.py         ← Stage 2: compare extracted vs gold
│   │   ├── 03_train_probe.py   ← Stage 3: train logistic regression probe
│   │   ├── 04_evaluate.py      ← Stage 4: AUROC comparison vs baselines
│   │   └── 05_lodo_cv.py       ← Leave-one-document-out cross-validation
│   ├── src/probe_extraction/   ← library code (importable package)
│   ├── data/extract-bench/     ← ExtractBench dataset (gitignored)
│   ├── artifacts/              ← experiment outputs (gitignored)
│   │   ├── qwen35_4b_pymupdf/  ← one dir per experiment
│   │   │   ├── extractions/    ← JSON outputs from Stage 1
│   │   │   ├── activations/    ← .npz files with hidden states
│   │   │   ├── labels/         ← per-field correct/error labels
│   │   │   ├── probes/         ← trained probe .pkl files
│   │   │   └── results/        ← comparison.json, lodo_cv.json
│   │   ├── qwen35_4b_swimming/
│   │   └── qwen35_4b_docling/
│   ├── logs/                   ← SLURM job logs + Python logs
│   ├── setup_env.sh            ← sourced by all job scripts
│   ├── run_extraction.sh       ← SLURM job: full pipeline (academic/research)
│   ├── run_extraction_swimming.sh
│   ├── run_extraction_docling.sh
│   ├── run_analysis.sh         ← SLURM job: stages 2-4 only (no GPU needed)
│   ├── run_analysis_swimming.sh
│   ├── test_extraction.sh      ← SLURM job: 1-doc test run
│   ├── PROGRESS.md
│   ├── Updates.md
│   └── TROUBLESHOOTING.md
│
└── nlp_lab/                    ← Python virtual environment (NOT a git repo)
```

---

## 5. The Git Workflow

All code changes happen locally (laptop), then get synced to the cluster via git.

### The cycle
```bash
# 1. Edit on laptop
nano src/probe_extraction/extraction/parser.py  # or use your IDE

# 2. Push from laptop
git add -A
git commit -m "Fix parser truncation bug"
git push

# 3. Pull on cluster
ssh bender
cd ~/NLP_Lab
git pull

# 4. Submit job
sbatch run_extraction.sh
```

### Common mistake: forgetting to pull
If you edit locally, push, then `sbatch` without `git pull`, the job runs the OLD code. Always pull before submitting.

### Editing directly on the cluster
Sometimes faster for small fixes. Use `nano`:
```bash
nano ~/NLP_Lab/src/probe_extraction/extraction/parser.py
# Ctrl+O to save, Ctrl+X to exit
```
Then push FROM the cluster:
```bash
git add -A
git commit -m "Quick fix on cluster"
git push
```
Then `git pull` on your laptop to stay in sync.

### Verifying a file's contents before submitting
```bash
cat run_extraction_docling.sh  # always check before sbatch
grep -E "job-name|partition|config" run_extraction_docling.sh  # quick check
```

---

## 6. SLURM Job Submission

### Anatomy of a job script
```bash
#!/bin/bash
#SBATCH --partition=A40short       # GPU type and time limit
#SBATCH --time=4:00:00             # max wall time (HH:MM:SS)
#SBATCH --gpus=1                   # number of GPUs
#SBATCH --ntasks=1                 # number of tasks
#SBATCH --cpus-per-task=8          # CPU cores
#SBATCH --mem=32G                  # RAM
#SBATCH --job-name=qwen35_pymupdf  # shows in squeue/sacct
#SBATCH --output=logs/slurm-%j.out # stdout (%j = job ID)
#SBATCH --error=logs/slurm-%j.err  # stderr

source ~/NLP_Lab/setup_env.sh     # load modules + activate venv
cd ~/NLP_Lab
nvidia-smi                         # log GPU info

python scripts/01_extract.py --config configs/exp_qwen35_4b_pymupdf.yaml
python scripts/02_label.py --config configs/exp_qwen35_4b_pymupdf.yaml
python scripts/03_train_probe.py --config configs/exp_qwen35_4b_pymupdf.yaml
python scripts/04_evaluate.py --config configs/exp_qwen35_4b_pymupdf.yaml
```

### What `setup_env.sh` does
```bash
#!/bin/bash
module load Python/3.12.3
module load CUDA/12.4.0
source ~/nlp_lab/bin/activate
export PYTHONPATH=$HOME/NLP_Lab/src:$PYTHONPATH
```

### Submitting a job
```bash
sbatch run_extraction.sh
# Output: Submitted batch job 197701
```

### Checking job status
```bash
# Currently running/queued jobs
squeue --me

# Recent job history (today)
sacct -u $USER --format=JobID,JobName,State,ExitCode,Elapsed -X -S today | tail -10

# Specific job details
sacct -j 197701 --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

### Job states
| State | Meaning |
|-------|---------|
| PD | Pending (waiting for resources) |
| R | Running |
| COMPLETED | Finished successfully (exit code 0) |
| FAILED | Finished with error (exit code != 0) |
| TIMEOUT | Hit the time limit |
| CANCELLED | Manually cancelled |

### Cancelling a job
```bash
scancel 197701        # cancel specific job
scancel -u $USER      # cancel ALL your jobs
```

### Partition choice guide
| Scenario | Partition | Why |
|----------|-----------|-----|
| Quick 1-doc test | A40devel | 1 hour limit, fast queue |
| Full extraction (6 docs) | A40short | 8 hour limit |
| Overnight extraction (35 docs) | A40medium | 1 day limit |
| Large model (Llama-3.1-8B) | A40short or A100short | 8B model fits A40 in bf16 |
| Analysis only (stages 2-4) | A40devel | No GPU needed, fast |
| LODO cross-validation | Login node | No GPU needed, runs in seconds |

---

## 7. Reading Logs

### Where logs go

SLURM job logs land in `~/NLP_Lab/logs/`. The naming depends on the `--output` and `--error` directives in your job script:

```
logs/
├── slurm-197701.out          # if --output=logs/slurm-%j.out
├── slurm-197701.err
├── test-197663.out           # if --output=logs/test-%j.out
├── test-197663.err
├── analysis-197710.out       # if --output=logs/analysis-%j.out
├── analysis-197710.err
└── 01_extract_20260510_222658.log  # Python's own log file
```

### Finding the latest logs
```bash
ls -lt ~/NLP_Lab/logs/ | head -10   # most recent files first
```

### Reading logs
```bash
# Full output
cat logs/slurm-197701.out

# Just the end (where results usually are)
tail -50 logs/slurm-197701.out

# Errors only
cat logs/slurm-197701.err

# Watch a running job's output in real time
tail -f logs/slurm-197701.out
# Press Ctrl+C to stop watching (doesn't kill the job)

# Search for specific patterns
grep -i "error\|fail\|warning" logs/slurm-197701.err
grep "AUROC\|RESULTS" logs/analysis-197710.out
```

### What to look for in `.out` files
```
nvidia-smi output        → GPU was allocated correctly
"Model loaded"           → model initialized
"Extracting: X/Y"        → progress through documents
"Generated N tokens"     → extraction working
"Done: X/Y successful"   → final summary
"=== RESULTS ==="        → comparison.json output (if echoed)
```

### What to look for in `.err` files
```
"Illegal instruction"    → CPU mismatch, wrong partition
"ModuleNotFoundError"    → venv/module not loaded
"CUDA out of memory"     → document too large for GPU
"Traceback"              → Python exception (read the full trace)
"set +x" / "++"          → debug mode output (if set -x is enabled)
```

### Debugging a failed job
```bash
# 1. Check exit code
sacct -j 197701 --format=JobID,State,ExitCode,Elapsed

# 2. Read stderr (most errors land here)
cat logs/slurm-197701.err | tail -30

# 3. If stderr is empty or unhelpful, check stdout
cat logs/slurm-197701.out | tail -30

# 4. For more detail, add debug output to job script:
# Add these lines after source setup_env.sh:
set -e   # exit on first error
set -x   # print every command before running
which python
python --version
python -c "from probe_extraction.config import load_config; print('import OK')"
```

---

## 8. Running Experiments

### Creating a new experiment

1. **Create config:**
```bash
cp configs/exp_qwen35_4b_pymupdf.yaml configs/exp_NEW_EXPERIMENT.yaml
nano configs/exp_NEW_EXPERIMENT.yaml
# Change: experiment.name, model settings, domain, etc.
```

2. **Create job script:**
```bash
cp run_extraction.sh run_extraction_NEW.sh
nano run_extraction_NEW.sh
# Change: job-name, all --config references
```

3. **Verify before submitting:**
```bash
grep -E "job-name|config" run_extraction_NEW.sh
# Every config reference should point to exp_NEW_EXPERIMENT.yaml
```

4. **Submit:**
```bash
sbatch run_extraction_NEW.sh
squeue --me
```

### Running the full pipeline (stages 1-4)

**Option A: All stages in one job script** (recommended for overnight runs)
```bash
#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=8:00:00
#SBATCH --gpus=1
# ... other SBATCH directives ...

source ~/NLP_Lab/setup_env.sh
cd ~/NLP_Lab

python scripts/01_extract.py --config configs/exp_qwen35_4b_pymupdf.yaml
python scripts/02_label.py --config configs/exp_qwen35_4b_pymupdf.yaml
python scripts/03_train_probe.py --config configs/exp_qwen35_4b_pymupdf.yaml
python scripts/04_evaluate.py --config configs/exp_qwen35_4b_pymupdf.yaml

echo "=== RESULTS ==="
cat artifacts/qwen35_4b_pymupdf/results/comparison.json
```

**Option B: Extraction separate from analysis** (when iterating on stages 2-4)
```bash
# First: extraction (needs GPU, slow)
sbatch run_extraction.sh

# After extraction finishes: analysis (no GPU needed, fast)
sbatch run_analysis.sh
```

### Running analysis only (stages 2-4, no GPU)

Use `A40devel` partition since analysis is CPU-only and fast:
```bash
#!/bin/bash
#SBATCH --partition=A40devel
#SBATCH --time=0:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=nlp_analysis
#SBATCH --output=logs/analysis-%j.out
#SBATCH --error=logs/analysis-%j.err

source ~/NLP_Lab/setup_env.sh
cd ~/NLP_Lab

python scripts/02_label.py --config configs/exp_qwen35_4b_pymupdf.yaml
python scripts/03_train_probe.py --config configs/exp_qwen35_4b_pymupdf.yaml
python scripts/04_evaluate.py --config configs/exp_qwen35_4b_pymupdf.yaml

echo "=== RESULTS ==="
cat artifacts/qwen35_4b_pymupdf/results/comparison.json
```

### Running LODO cross-validation (no SLURM needed)

LODO is CPU-only and runs in seconds. Run directly on the login node:
```bash
cd ~/NLP_Lab
python scripts/05_lodo_cv.py --config configs/exp_qwen35_4b_pymupdf.yaml
python scripts/05_lodo_cv.py --config configs/exp_qwen35_4b_swimming.yaml
```

### Running a 1-document test
```bash
# Use the test script with --limit 1
sbatch test_extraction.sh
# test_extraction.sh should contain:
# python scripts/01_extract.py --config configs/exp_qwen35_4b_pymupdf.yaml --limit 1
```

---

## 9. Checking Results

### Extraction summary
```bash
cat artifacts/<experiment>/extractions/_summary.json | python -m json.tool
```
Key fields: `successful`, `parse_failed`, `total_fields_extracted`, `per_document` array.

### Labels summary
```bash
cat artifacts/<experiment>/labels/_summary.json | python -m json.tool
```
Key fields: `n_total_fields`, `n_errors`, `error_rate`, `per_document` array.

### Probe results
```bash
cat artifacts/<experiment>/results/comparison.json | python -m json.tool
```
Key fields: `n_samples`, `n_errors`, `baselines` (AUROC), `probes` per layer (AUROC).

### LODO results
```bash
cat artifacts/<experiment>/results/lodo_cv.json | python -m json.tool
```
Key fields: per-layer `lodo_auroc_mean`, `lodo_auroc_std`, `n_valid_folds`.

### Activation file inspection
```bash
python -c "
import numpy as np
data = np.load('artifacts/<experiment>/activations/<doc_id>.npz')
keys = list(data.keys())
print(f'Keys: {len(keys)}')
for k in keys[:5]:
    print(f'  {k}: shape={data[k].shape} dtype={data[k].dtype}')
"
```

### Label file inspection (eyeball errors)
```bash
python -c "
import json
lab = json.load(open('artifacts/<experiment>/labels/<doc_id>.json'))
for x in lab['labels']:
    if x['is_error'] == 1:
        print(f\"[{x['comparison_strategy']}] {x['path_str']}\")
        print(f\"   gold:      {x.get('gold_value')!r}\")
        print(f\"   extracted: {x.get('extracted_value')!r}\")
        print()
"
```

### Field count funnel (extraction → labeling → probe)
```bash
python -c "
import json
doc = '<doc_id>'
exp = '<experiment>'

ext = json.load(open(f'artifacts/{exp}/extractions/{doc}.json'))
lab = json.load(open(f'artifacts/{exp}/labels/{doc}.json'))

print('Extracted fields:', len(ext.get('fields', [])))
print('Labeled fields:', len(lab.get('labels', [])))
print('Errors:', sum(1 for x in lab.get('labels', []) if x['is_error'] == 1))
"
```

---

## 10. GPU and Resource Management

### GPUs are NOT "allocated" to you between jobs
SLURM allocates GPUs only during job execution. When a job finishes (success or failure), the GPU is released automatically. You don't need to "deallocate" anything.

### Checking GPU availability
```bash
sinfo -p A40short --format="%P %a %D %T %G"   # partition info
squeue -p A40short                               # who's using what
```

### Memory considerations

| Model | VRAM (bf16) | VRAM (4-bit) | Fits A40? | Fits A100? |
|-------|------------|-------------|-----------|------------|
| Qwen3.5-4B | ~8 GB | ~3 GB | Yes | Yes |
| Llama-3.1-8B | ~16 GB | ~5 GB | Yes | Yes |
| Qwen2.5-14B | ~28 GB | ~8 GB | Yes | Yes |
| Qwen2.5-32B | ~64 GB | ~18 GB | No | 80GB only |

Add ~2-10 GB for KV cache depending on input length.

### If a job OOMs
The pipeline catches OOM per-document and continues. Check the extraction summary to see which documents failed:
```bash
cat artifacts/<experiment>/extractions/_summary.json | python -m json.tool | grep -A5 "error"
```

---

## 11. Experiment Configurations

### Config file structure
```yaml
experiment:
  name: "qwen35_4b_pymupdf"    # becomes artifact directory name
  seed: 42
  artifacts_dir: "artifacts"

model:
  name: "Qwen/Qwen3.5-4B"
  dtype: "bfloat16"
  quantization: "none"          # none | 4bit | 8bit
  device_map: "auto"
  trust_remote_code: true       # required for Qwen3.5
  max_new_tokens: 16384
  temperature: 0.0              # greedy decoding

activations:
  layers: [1, 4, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
  position: "last_token"        # which token's hidden state to use
  dtype: "float16"

data:
  benchmark: "extract_bench"
  benchmark_path: "data/extract-bench"
  domains:
    - "academic/research"       # which domain(s) to extract
  max_documents: null
  pdf_extractor: "pymupdf"      # pymupdf | docling

extraction:
  prompt_template: "default"
  include_schema: true
  include_examples: false
  max_input_chars: 600000       # cap for very long documents
```

### Key config knobs and what they do

| Parameter | What it controls | When to change |
|-----------|-----------------|----------------|
| `model.name` | Which LLM to use | Cross-model comparison |
| `model.max_new_tokens` | Max output length | If JSON gets truncated |
| `activations.layers` | Which layers to capture | Layer sweep experiments |
| `activations.position` | Token reduction strategy | Position ablation |
| `data.domains` | Which ExtractBench domains | Adding new domains |
| `data.pdf_extractor` | PDF parsing backend | PyMuPDF vs Docling A/B |
| `extraction.max_input_chars` | Input truncation | If OOM on long docs |

---

## 12. Useful Aliases and Shortcuts

Add to `~/.bashrc` on the cluster:
```bash
# Quick git pull
alias gp='cd ~/NLP_Lab && git pull'

# Check job status
alias sq='squeue --me'
alias sa='sacct -u $USER --format=JobID,JobName,State,ExitCode,Elapsed -X -S today | tail -10'

# Quick log peek
alias ll='ls -lt ~/NLP_Lab/logs/ | head -10'

# Latest log files
latest_out() { cat $(ls -t ~/NLP_Lab/logs/*.out 2>/dev/null | head -1); }
latest_err() { cat $(ls -t ~/NLP_Lab/logs/*.err 2>/dev/null | head -1); }
```

---

## 13. Experiment Checklist

Before submitting any experiment:

- [ ] `git pull` on the cluster (sync with laptop)
- [ ] Config file exists and has correct settings (`cat configs/exp_*.yaml`)
- [ ] Job script references the correct config (`grep config run_*.sh`)
- [ ] Job script has the right partition and time limit
- [ ] Job script runs the correct stages (extraction vs analysis)
- [ ] `logs/` directory exists (`mkdir -p logs`)
- [ ] Previous artifacts backed up or intentionally overwritable
- [ ] `squeue --me` shows no conflicting running jobs

After a job completes:

- [ ] Check exit code: `sacct -j <jobid> --format=State,ExitCode`
- [ ] Read `.err` file for warnings/errors
- [ ] Check extraction summary: `cat artifacts/<exp>/extractions/_summary.json`
- [ ] If analysis ran: check comparison.json for probe vs baseline AUROC
- [ ] Commit any cluster-side changes: `git add -A && git commit -m "msg" && git push`

---

## 14. Context for New Chat Sessions

When starting a new Claude chat about this project, include this information:

```
Project: NLP Lab — probe-based trust signals for structured information extraction.
Cluster: Bender HPC (bender.hpc.uni-bonn.de), user s44srizv.
Project path: ~/NLP_Lab/, venv: ~/nlp_lab/.
Modules: Python/3.12.3, CUDA/12.4.0.
Workflow: edit locally → git push → git pull on cluster → sbatch.
Key files: setup_env.sh (env loader), configs/ (experiment YAMLs),
  scripts/01-05 (pipeline stages), artifacts/ (outputs per experiment).
Current model: Qwen3.5-4B (bf16, thinking disabled).
Reference docs: TROUBLESHOOTING.md, HPC_GUIDE.md, Updates.md, PROGRESS.md.
```

Then describe your specific question or task.

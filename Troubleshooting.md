# Troubleshooting Guide — NLP Lab (Probe-Based Trust Signals)

Common errors encountered during development on the University of Bonn Bender HPC cluster, organized by category. Each entry includes the symptom, root cause, and fix.

---

## 1. Environment & Module Issues

### 1.1 `libpython3.12.so.1.0: cannot open shared object file`

**Symptom:**
```
python: error while loading shared libraries: libpython3.12.so.1.0:
cannot open shared object file: No such file or directory
```

**Root cause:** The Python module isn't loaded in the current shell. The venv's Python binary is dynamically linked against `libpython` from the module — without the module loaded, the shared library can't be found.

**Fix:**
```bash
module load Python/3.12.3
source ~/nlp_lab/bin/activate
```

**Prevention:** Add to `~/.bashrc`:
```bash
module load Python/3.12.3
module load CUDA/12.4.0
source ~/nlp_lab/bin/activate
```

---

### 1.2 `which pip` shows module's pip, not venv's pip

**Symptom:**
```bash
(nlp_lab) $ which pip
/software/easybuild-INTEL_A40/software/Python/3.12.3-GCCcore-13.3.0/bin/pip
```
Despite the `(nlp_lab)` prompt prefix, `pip` points to the system module's pip.

**Root cause:** The `module load Python/...` command adds the module's `bin/` to PATH. If the module is loaded *after* venv activation, the module's PATH entry can override the venv's. The `(nlp_lab)` prefix is just a string set during activation — it doesn't guarantee the venv is actually first on PATH.

**Fix:** Re-activate the venv after loading modules:
```bash
module load Python/3.12.3
module load CUDA/12.4.0
source ~/nlp_lab/bin/activate  # must come AFTER module loads
which pip  # verify: should show ~/nlp_lab/bin/pip
```

**Why it matters:** Running `pip install` with the wrong pip installs packages to the wrong location — either the system site-packages (read-only, falls back to `~/.local/`) or a different Python environment entirely.

---

### 1.3 `ModuleNotFoundError: No module named 'probe_extraction'`

**Symptom:**
```
ModuleNotFoundError: No module named 'probe_extraction'
```

**Root cause (variant A — doubled path):** `pip install -e .` was run with the system pip (not the venv's pip), writing an incorrect path into the editable-install `.pth` file. The `.pth` file contained `/home/s44srizv/NLP_Lab/NLP_Lab/src` (doubled directory name) instead of `/home/s44srizv/NLP_Lab/src`.

**Root cause (variant B — not installed):** The editable install was never performed in the current venv.

**Root cause (variant C — SLURM job):** The job script doesn't activate the venv or load the Python module before running the Python script.

**Diagnosis:**
```bash
# Check what's in the .pth file
cat ~/nlp_lab/lib/python3.12/site-packages/__editable__.probe_extraction*.pth

# Check if installed at all
pip list | grep -i probe

# Verify import works
python -c "from probe_extraction.config import load_config; print('OK')"
```

**Fix:**
```bash
# Ensure venv is active and you're in the right directory
source ~/nlp_lab/bin/activate
which pip  # must show ~/nlp_lab/bin/pip

# Clean up any bad installs
pip uninstall probe-extraction -y
rm -f ~/.local/lib/python3.12/site-packages/__editable__.probe_extraction*.pth
rm -rf ~/.local/lib/python3.12/site-packages/probe_extraction-*.dist-info

# Reinstall correctly
cd ~/NLP_Lab
pip install -e . --no-deps

# Verify
cat ~/nlp_lab/lib/python3.12/site-packages/__editable__.probe_extraction*.pth
# Should print: /home/s44srizv/NLP_Lab/src
```

**Prevention:** Always verify `which pip` shows the venv's pip before running `pip install`.

---

### 1.4 PyTorch / CUDA version mismatch

**Symptom:** PyTorch loads but CUDA operations fail, or `torch.cuda.is_available()` returns `False`, or import errors related to CUDA.

**Root cause:** The PyTorch wheel was built for a different CUDA version than what's available on the cluster. The cluster's maximum CUDA module is 12.4.0, but the PyTorch wheel may have been built for CUDA 13.0.

**Fix:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**Verification:**
```python
import torch
print(torch.cuda.is_available())  # True
print(torch.version.cuda)          # Should match cluster CUDA
```

---

### 1.5 `Defaulting to user installation because normal site-packages is not writeable`

**Symptom:**
```
Defaulting to user installation because normal site-packages is not writeable
```

**Root cause:** `pip install` is using the system pip (from the module), not the venv's pip. The system site-packages is read-only, so pip falls back to installing in `~/.local/lib/python3.12/site-packages/`.

**Fix:** Activate the venv first (see 1.2), then retry the install.

---

## 2. SLURM / Cluster Issues

### 2.1 Job fails instantly with `ModuleNotFoundError`

**Symptom:** SLURM job exits in <15 seconds with `ModuleNotFoundError` in the `.err` file.

**Root cause:** The job script's `setup_env.sh` doesn't properly load modules and activate the venv before running Python.

**Fix:** Ensure `setup_env.sh` contains (in this order):
```bash
#!/bin/bash
module load Python/3.12.3
module load CUDA/12.4.0
source ~/nlp_lab/bin/activate
export PYTHONPATH=$HOME/NLP_Lab/src:$PYTHONPATH  # belt-and-suspenders
```

And the job script sources it:
```bash
#!/bin/bash
#SBATCH --partition=A40short
# ... other SBATCH directives ...
source ~/NLP_Lab/setup_env.sh
cd ~/NLP_Lab
python scripts/01_extract.py --config configs/exp_qwen35_4b_pymupdf.yaml
```

---

### 2.2 `Illegal instruction (core dumped)`

**Symptom:**
```
Illegal instruction (core dumped)
```
Even `tput cols` crashes. Job dies immediately.

**Root cause:** CPU instruction set mismatch. The compute node has an older CPU than the one your Python/venv was compiled for. Common when job lands on a different partition's node type.

**Fix:** Force a specific partition known to work:
```bash
#SBATCH --partition=A40short
```

Do NOT use partitions with mixed or older node types.

---

### 2.3 Changes not reflected in SLURM job output

**Symptom:** You edited a file locally (or on the cluster), but the job output shows the old behavior. Debug lines (`set -x`, print statements) don't appear.

**Root cause:** One of:
- Edited locally but forgot to `git push`
- Pushed but forgot to `git pull` on the cluster
- Edited a file but submitted the wrong script (e.g., `run_analysis_X.sh` instead of `run_extraction_X.sh`)

**Fix / workflow:**
```bash
# On laptop:
git add -A && git commit -m "description" && git push

# On cluster:
cd ~/NLP_Lab
git pull
cat <script_name>.sh  # verify contents before submitting
sbatch <script_name>.sh
```

**Prevention:** Always `cat` the script before `sbatch` to verify it's the version you expect.

---

### 2.4 Submitting the wrong script

**Symptom:** Job finishes in seconds when it should take 30+ minutes. `sacct` shows job name as `nlp_analy+` when you expected `qwen35_...`.

**Root cause:** You submitted `run_analysis_X.sh` (stages 2-4, CPU-only, fast) instead of `run_extraction_X.sh` (stage 1, GPU, slow). Or the file you think is an extraction script is actually a copy of the analysis script.

**Diagnosis:**
```bash
# Check what the script actually does
head -15 run_extraction_docling.sh
# Look for: scripts/01_extract.py (extraction) vs scripts/02_label.py (analysis)
```

**Fix:** Verify the script contains the right stage scripts before submitting. For extraction scripts, you must see `python scripts/01_extract.py`.

---

### 2.5 File created with leading space in name

**Symptom:**
```bash
$ ls
' run_extraction_swimming.sh'   README.md   ...
```
Note the leading space in the filename (shown with quotes by `ls`).

**Root cause:** A `cp` or redirect command had an extra space in the destination filename.

**Fix:**
```bash
mv " run_extraction_swimming.sh" run_extraction_swimming.sh
# Or use tab completion:
mv ' r<TAB>'  run_extraction_swimming.sh
```

---

### 2.6 SSH disconnects / "Connection closed by remote host"

**Symptom:**
```
Connection to bender.hpc.uni-bonn.de closed by remote host.
Connection to bender.hpc.uni-bonn.de closed.
```

**Root cause:** The cluster's SSH server disconnects idle sessions after a timeout. Your running SLURM jobs are unaffected (they run on compute nodes, not the login node).

**Fix:** Just reconnect. Add to `~/.ssh/config` on your laptop:
```
Host bender
    HostName bender.hpc.uni-bonn.de
    User s44srizv
    ServerAliveInterval 60
```
Then connect with `ssh bender`.

---

### 2.7 CUDA OOM on specific documents

**Symptom:**
```
CUDA out of memory. Tried to allocate X GiB.
```

**Root cause:** Document is too long for the GPU's memory. The Zhao 2025 LLM survey (265k input tokens) exceeds even 48GB A40's attention buffers.

**Mitigation options (in order):**
1. Set `max_input_chars: 600000` in config to cap input length
2. Increase `max_new_tokens` if output truncation is the issue
3. Use a bigger GPU (`--partition=A100short`)
4. Accept the failure and document it — some docs are genuinely too large

**Note:** The pipeline's try/except in `01_extract.py` catches OOM and continues to the next document. One OOM doesn't kill the whole run.

---

## 3. Model-Specific Issues (Qwen3.5-4B)

### 3.1 All generations labeled `finish=length` even when complete

**Symptom:** Every extraction shows `finish_reason: "length"` despite the JSON output looking complete. Token count is well below `max_new_tokens`.

**Root cause:** Qwen3.5+ uses `<|im_end|>` (token id ~151645) as the end-of-turn marker, not `<|endoftext|>` (the standard `eos_token_id`). The model wrapper only checked `eos_token_id` and didn't recognize `<|im_end|>` as a stop signal.

**Fix:** In `hf_model.py`, add `<|im_end|>` to the EOS token list:
```python
eos_ids = []
if self.tokenizer.eos_token_id is not None:
    eos_ids.append(self.tokenizer.eos_token_id)
im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
if im_end_id is not None and im_end_id != self.tokenizer.unk_token_id:
    if im_end_id not in eos_ids:
        eos_ids.append(im_end_id)
```

And pass `eos_token_id=eos_ids` to `model.generate()`.

---

### 3.2 Thinking mode output pollutes JSON

**Symptom:** Model output contains `<think>...</think>` blocks before or mixed into the JSON, causing parse failures.

**Root cause:** Qwen3.5's thinking mode is enabled by default. Even with `enable_thinking=False`, the model occasionally emits stray thinking tokens.

**Fix (two layers):**
1. Pass `enable_thinking=False` in the chat template call
2. Strip `<think>` blocks in the parser:
```python
text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
```

---

### 3.3 `Layer 0 out of range` error

**Symptom:**
```
ValueError: Layer 0 out of range for model Qwen/Qwen3.5-4B (has 32 layers)
```

**Root cause:** The Extractor validates that layer indices are in `[1, num_layers]`. Layer 0 (the embedding layer) is rejected.

**Fix:** Use 1-indexed layers in config:
```yaml
activations:
  layers: [1, 4, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
```
Where layer 1 = first transformer block output, layer 32 = last transformer block output. To probe the embedding layer (layer 0), modify the Extractor's validation to accept `layer >= 0`.

---

### 3.4 Config file not found on cluster

**Symptom:**
```
FileNotFoundError: Config file not found: configs/exp_qwen35_4b_pymupdf.yaml
```

**Root cause:** The config file was created locally but never pushed to the cluster via git.

**Fix:**
```bash
# On laptop:
git add configs/exp_qwen35_4b_pymupdf.yaml
git commit -m "Add experiment config"
git push

# On cluster:
git pull
ls configs/  # verify file is present
```

---

## 4. Parser / Extraction Issues

### 4.1 `TypeError: the JSON object must be str, bytes or bytearray, not NoneType`

**Symptom:**
```
TypeError: the JSON object must be str, bytes or bytearray, not NoneType
```
in `parse_json_output` → `json.loads(candidate)`.

**Root cause:** The `_strip_to_json` function returned `None` instead of a string. This happened because the function body was accidentally truncated during editing — it was missing the return statements after the `<think>` regex block.

**Fix:** Ensure `_strip_to_json` has all four handling stages:
```python
def _strip_to_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()

    first_brace = text.find("{")
    if first_brace >= 0:
        extracted = _extract_balanced_braces(text, first_brace)
        if extracted is not None:
            return extracted

    return text  # fallback — let json.loads decide
```

Also harden `parse_json_output` to catch `TypeError`:
```python
except (json.JSONDecodeError, TypeError, ValueError) as e:
    return None, f"{type(e).__name__}: {e}", candidate or ""
```

---

### 4.2 Unclosed ` ```json ` fence causes parse failure

**Symptom:** Model wraps output in ` ```json ... ``` ` but hits the token limit before the closing fence. Parser's fence regex requires both opening and closing fences, so it doesn't match.

**Root cause:** The `_FENCE_RE` regex requires ```` ```...``` ```` (both delimiters). When generation is truncated, only the opening fence exists.

**Current status:** Documented limitation. Fixable by adding a fallback regex that matches ` ```json\n{...` without requiring a closing fence. Deferred.

---

### 4.3 Output truncated at exactly N tokens with `finish=length`

**Symptom:** Generation stops at exactly `max_new_tokens` (e.g., 8192 or 16384) mid-JSON, especially on citation-heavy documents.

**Root cause:** The model is faithfully enumerating hundreds of citations, exceeding the output budget. Not a bug — the model is doing its job, just with more output than we budgeted.

**Mitigation:**
- Increase `max_new_tokens` in config (e.g., 16384 or 32768)
- Accept that some very citation-heavy documents will always truncate
- Optionally: modify the prompt to limit citation count

---

## 5. Git / Workflow Issues

### 5.1 GitHub push protection blocks push (exposed secret)

**Symptom:**
```
remote: error: GH013: Repository rule violations found
Push cannot contain secrets
```

**Root cause:** A HuggingFace token or other secret was committed to a tracked file.

**Fix:**
1. Rotate the exposed token immediately (generate a new one on HuggingFace)
2. Remove the secret from the commit:
```bash
git commit --amend --no-edit  # if it's the most recent commit
# Or for older commits: git rebase -i
```
3. Store secrets in `~/.bashrc` (cluster) or `.env` (gitignored), never in tracked files

---

### 5.2 `fatal: not a git repository`

**Symptom:**
```
fatal: not a git repository (or any parent up to mount point /home)
```

**Root cause:** You ran a git command from outside the repo directory (e.g., from `~` instead of `~/NLP_Lab`).

**Fix:**
```bash
cd ~/NLP_Lab
git pull  # now works
```

---

## 6. LODO / Analysis Issues

### 6.1 LODO script finds 0 documents with usable data

**Symptom:**
```
Layer 1: 0 documents with usable data
```

**Root cause:** The LODO script assumed the wrong JSON structure for label files. It looked for `"fields"` key with `"label": "error"`, but the actual structure uses `"labels"` key with `"is_error": 0/1`.

**Fix:** In the LODO script's `load_doc_data`, use:
```python
fields = labels_data.get("labels", [])  # not "fields"
label = int(field["is_error"])           # not field["label"] == "error"
```

---

### 6.2 Analysis shows same numbers as a different experiment

**Symptom:** Swimming analysis outputs show `n_samples: 74, n_errors: 20` — identical to academic results.

**Root cause:** The analysis script's config path still points to the wrong experiment. Often the `cat` line at the end of the script reads from the wrong artifact directory.

**Diagnosis:**
```bash
cat run_analysis_swimming.sh
# Check EVERY line — the python calls AND the final cat command
# All should reference exp_qwen35_4b_swimming, not exp_qwen35_4b_pymupdf
```

**Fix:** Update all references in the script, including the final `cat` line:
```bash
cat artifacts/qwen35_4b_swimming/results/comparison.json
```

---

### 6.3 `Best layer: 1 (AUROC=nan)` in probe summary

**Symptom:** Probe summary reports `AUROC=nan` for the "best" layer.

**Root cause:** The test set in the train/test split happened to have zero errors (`n_test_errors: 0`). AUROC is undefined when only one class is present. The "best layer" picker read this undefined AUROC.

**Impact:** Cosmetic. The cross-validation AUROC (`cv_auroc_mean`) is computed correctly across folds and is the number to report. Ignore the per-split AUROC when it's NaN.

---

## 7. Quick Reference: Common Command Sequences

### Fresh SSH session setup
```bash
ssh s44srizv@bender.hpc.uni-bonn.de
# If .bashrc has module loads, you're ready. Otherwise:
module load Python/3.12.3
module load CUDA/12.4.0
source ~/nlp_lab/bin/activate
cd ~/NLP_Lab
```

### Full push-pull-submit cycle
```bash
# Laptop:
git add -A && git commit -m "msg" && git push

# Cluster:
cd ~/NLP_Lab && git pull
cat <script>.sh  # verify
sbatch <script>.sh
squeue --me
```

### Check recent job status
```bash
sacct -u $USER --format=JobID,JobName,State,ExitCode,Elapsed -X -S today | tail -10
```

### Read job output
```bash
ls -lt ~/NLP_Lab/logs/ | head -5
cat ~/NLP_Lab/logs/<prefix>-<jobid>.out
cat ~/NLP_Lab/logs/<prefix>-<jobid>.err
```

### Verify environment is correct
```bash
which python   # ~/nlp_lab/bin/python
which pip      # ~/nlp_lab/bin/pip
python -c "from probe_extraction.config import load_config; print('OK')"
```

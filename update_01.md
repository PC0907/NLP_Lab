# Project Progress — Probe-Based Trust Signals for Structured Information Extraction

**Period covered:** May 1-11, 2026

---

## Research Question

> Can probe-based trust signals identify risky extracted fields well enough to improve the cost-quality tradeoff of selective regeneration, compared to black-box baselines (token log-probabilities, self-consistency)?

The probe is a means, not the end. The headline result is a Pareto curve comparing probe-guided selective regeneration against baselines on extraction accuracy vs. compute cost. The current work validates that a probe-based trust signal *exists* before building the regeneration experiment.

---

## Where We Are

A complete pipeline is running end-to-end: PDF extraction → LLM-based structured extraction with hidden-state capture → per-field gold matching → linear probe training → AUROC evaluation against baselines.

Preliminary numbers (Qwen3.5-4B, 3 academic/research papers, 74 probe-trainable fields, 20 errors):

| Method | AUROC |
|---|---|
| Baseline mean_logprob | 0.651 |
| Baseline min_logprob | 0.708 |
| Probe layer 4 (5-fold CV) | **0.862 ± 0.039** |
| Probe layer 32 (5-fold CV) | 0.871 ± 0.076 |

Best probe beats best baseline by ~0.15 AUROC. The direction matches an earlier Kaggle-based run with Qwen2.5-7B (probe 0.785 ± 0.16 vs baseline 0.71). Both are small datasets, so the result is suggestive but not yet statistically conclusive.

---

## What Works

**Infrastructure (~3000 LOC across ~35 files, 111 tests passing):**
- Modular abstract interfaces (`Benchmark`, `LLM`, `Probe`) so components are swappable.
- Pydantic-validated YAML configs — typos fail at load time, not after 20 minutes of generation.
- Per-stage scripts with disk artifacts. Re-run probe training without re-running extraction.
- Per-experiment artifact subdirectories so multiple experiments coexist on disk.
- SLURM job scripts for the Bender cluster; shared `setup_env.sh` for module/venv setup.
- Configurable PDF extraction backend (PyMuPDF or Docling).

**Pipeline stages validated end-to-end:**
- Stage 1 (extraction): Qwen3.5-4B in bf16 on A40, no quantization. Captures hidden states at 14 layers (spread across all 32 layers of Qwen3.5-4B).
- Stage 2 (labeling): per-field correctness via fuzzy matching against gold annotations, with type-aware normalization (emails, URLs, numbers).
- Stage 3 (probe training): logistic regression with 5-fold cross-validation and class-balancing.
- Stage 4 (evaluation): probe AUROC vs. token-logprob baselines.

---

## Engineering Issues Encountered (with resolution)

A meaningful fraction of this period went into infrastructure, not science. Documenting these so the same issues don't recur and so the time accounting is honest.

### ExtractBench data quirks (caught by tests before any GPU was used)

1. **Inconsistent schema wrapping.** Some ExtractBench schemas wrap the actual JSON Schema in `{"name": ..., "schema_definition": {...}}`; others don't. The loader now handles both.
2. **Image-only PDFs in hiring/resume.** 5 of 7 resume PDFs are Google Docs exports without text layers. PyMuPDF returns empty text. The loader flags these via `extraction_error` and the pipeline skips them. We switched the sandbox to `academic/research` (all 6 PDFs have proper text layers).

### Hidden-state alignment

3. **Off-by-one in HuggingFace `generate()`.** When generation ends in EOS, the hidden-state tuple has length N-1, not N. Originally caused every field to be rejected from probe training. Fixed by clamping field token-spans to within available activations.

### GPU memory management

4. **Memory leak across documents.** PyTorch's CUDA allocator was retaining hidden-state tensors between generation calls. Resolved with explicit `del outputs; gc.collect(); torch.cuda.empty_cache(); torch.cuda.ipc_collect()` after each generation.
5. **Attention is O(n²) on P100 (no FlashAttention).** Long surveys OOM'd on Kaggle's P100 GPU. Added input-length cap (configurable). This is no longer a constraint on Bender's A40/A100 GPUs, where FlashAttention is supported.

### Move from Kaggle to Bender

6. **PyTorch / CUDA version mismatch.** The PyTorch wheel I had was built for CUDA 13.0; the cluster's max CUDA module is 12.4.0. Reinstalled PyTorch from the cu124 wheel index.
7. **Venv / module ordering issues.** The Python venv depends on libpython from the loaded module; if the module isn't loaded in the SLURM job's shell, the venv breaks. Resolved by ensuring `module load Python/...` happens before venv activation in `setup_env.sh`, and adding the loads to `~/.bashrc` for interactive shells.
8. **Doubled-path editable install.** `pip install -e .` had once been run from the wrong shell (system pip from a module, not venv pip), writing `/home/.../NLP_Lab/NLP_Lab/src` (doubled) into the editable-install `.pth` file. Symptom: `ModuleNotFoundError: No module named 'probe_extraction'` despite `pip list` showing it installed. Resolved by uninstalling and reinstalling cleanly with venv pip.
9. **GitHub push protection caught a hardcoded HF token.** Token was inadvertently included in a job script. GitHub blocked the push; token was rotated, the bad commit rewritten with `git commit --amend --no-edit`, and the token is now stored in `~/.bashrc` on the cluster only (never in any tracked file).

### Qwen3.5-4B specifics

10. **`enable_thinking=False`** required in the chat template call. We treat Qwen3.5 as a regular instruct model for this experiment to keep methodology consistent with existing probing literature. Probing during the reasoning phase is a methodological question worth exploring later but it's a separate experiment.
11. **`<|im_end|>` as EOS.** Qwen3.5 uses `<|im_end|>` to signal end-of-turn, not `<|endoftext|>`. The model wrapper originally only checked `tokenizer.eos_token_id` (which is `<|endoftext|>`). Generations were labeled `finish=length` even when they had completed naturally; in some cases generation continued past the intended end. Fixed by treating both tokens as EOS in the model wrapper.

### Truncated generation

12. **Citation-heavy surveys exceed `max_new_tokens`.** Two of six academic papers (the VLM survey and the dimensionality reduction survey) hit the 16384-token output cap because the model dutifully enumerates hundreds of citations. The Zhao 2025 LLM survey is too long on input alone (265k tokens) to fit even on A40 attention buffers. These three documents are documented failures; the remaining three extracted cleanly.

---

## Current Results in Detail

**Dataset state (academic/research, 6 papers attempted):**
- 3 successful extractions (NIPS 1989, RAG survey, FlashAttention-3)
- 374 raw fields extracted across the three
- 90 fields labelable against gold (the labeler correctly drops citation-array items where gold-to-extracted pairing isn't possible)
- 74 fields usable for probe training (excludes is_empty fields with synthetic activation positions)
- 20 errors among 74 fields (27% error rate)

**Probe AUROC across layers (5-fold CV, n=74):**

| Layer | Probe AUROC ± std |
|---|---|
| 1 | 0.856 ± 0.046 |
| 4 | **0.862 ± 0.039** (best) |
| 8 | 0.834 ± 0.042 |
| 16 | 0.815 ± 0.060 |
| 20 | 0.848 ± 0.071 |
| 28 | 0.853 ± 0.057 |
| 32 | 0.871 ± 0.076 |

Probe AUROC is roughly flat across layers (0.81-0.87), suggesting the trust signal is broadly distributed rather than localized to a specific layer. With only 74 samples, the noise floor on individual layer differences is large; we cannot confidently say layer 4 outperforms layer 16. The probe-vs-baseline gap, however, is clearly larger than the within-probe variation.

**Why the gap might be wider than the earlier Kaggle run:**
- Higher error rate (27% vs 15.5% on Kaggle's Qwen2.5-7B) — more positive examples to train on
- Hidden-state quality is higher (bf16 instead of 4-bit quantized)
- 14 layers captured instead of 4, so layer-32 representation is now visible

---

## What We Know Doesn't Work / Open Questions

- **Scaling extraction by more output per document doesn't help probe training.** The labeler discards citation-array items where it can't pair extracted to gold. Going from 4k to 16k `max_new_tokens` increased raw field count from 120 to 374 — but trainable fields stayed at 74. The path to more data is more documents, not more output.
- **Per-document trainable yield is fairly stable.** NIPS=30, RAG=33, FlashAttention-3=27. Suggests ~25-35 trainable fields per academic paper. 35 docs (full ExtractBench) would be ~600 trainable fields.
- **Layer-level structure is unclear at n=74.** May become clearer at larger n.
- **One ExtractBench document (Zhao 2025 LLM survey) genuinely doesn't fit on a single 48GB GPU** at any reasonable truncation level. Would need chunked extraction or document filtering.

---

## In-Flight (Currently Running)

- **sport/swimming domain extraction** on Bender A40. Goal: a citation-free, more structured domain — tabular swim-meet results. Tests whether the probe-vs-baseline gap holds outside academic papers. Expected ~50 trainable fields per doc.
- **Docling vs PyMuPDF A/B** (to be submitted today). Tests whether layout-aware PDF extraction changes accuracy on academic/research.

---

## Plan for Next Week

**Days 1-2: Cross-domain validation.**
- Finish sport/swimming and Docling runs (in-flight).
- Run full ExtractBench across all 5 domains with PyMuPDF.
- Goal: ~400-600 probe-trainable fields. At that size, AUROC error bars should drop from ±0.04 to ±0.02 and we can start trusting layer-level differences.

**Days 3-4: Cross-model generalization.**
- Run Llama-3.1-8B on the same documents. Most important "is the result real?" check — a probe gap that holds across model families is much harder to attribute to architectural artifacts.
- Run Qwen2.5-14B for cross-scale comparison.

**Day 5: Ablations.**
- Position strategy: last_token vs mean vs concat(first, last, mean). Re-uses captured activations; no new GPU work needed.
- Layer sweep on all 32 layers (currently we capture 14).

**Days 6-7: Stage 5 — the actual deliverable.**
- Selective regeneration experiment. Use the best probe from the experiments above to flag risky fields, regenerate only those, measure cost-quality tradeoff vs. uniform regeneration and vs. no regeneration. This produces the Pareto curve that is the main result of the project.

---

## Open Questions for the Meeting

1. **Statistical sufficiency for early results.** At n=74, we have a clear probe-vs-baseline gap but wide CIs. Is 75 enough to claim "the signal exists" in writing, or do we hold back until n=300+?

2. **Methodological frame for Qwen3.5 (and reasoning models in general).** We're currently disabling thinking mode. The cleaner experiment is to probe during JSON generation (which is what we do now). The more interesting experiment is probing during the reasoning phase. Worth raising whether this is in scope for this project or a follow-up.

3. **The Zhao 2025 LLM survey is too big for our GPU.** Options: chunked extraction (changes the methodology), document filtering (loses one data point), bigger GPU (asks for compute). Preference?

4. **Strict vs LLM-judge labeling.** Current labeling is strict (fuzzy match against gold). ExtractBench's official metric uses an LLM judge for `string_semantic` and `array_llm` fields. Strict labeling is reproducible and deterministic but produces a stricter notion of "error" than the benchmark intends. Should we add an LLM-judge labeler before the next major experiment, or stay with strict and document the caveat?

5. **Time accounting.** Significant infrastructure effort went into the cluster migration this week (~3 days). The pipeline is now stable on Bender. I'd estimate the engineering load is mostly behind us; the rest of the project is experiments.

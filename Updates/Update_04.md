# Updates — June 10, 2026

Progress update for the `adnan-dev` branch. This work extends the pooled
four-domain probe result from `Updates_3.md` (May 17) in **two directions at
once**, both of which were listed as planned future work in that document:

1. A **new model** — DeepSeek-R1-Distill-Qwen-7B, the project's first
   *reasoning* model — instead of the partner branch's Qwen3.5-4B.
2. A **new probe architecture** — Cross-Layer Attention Probing (CLAP),
   which attends jointly over all captured layers, instead of one independent
   logistic regression per layer.

Everything else (the ExtractBench data, the activation-capture format, the
labelling pipeline, and the LODO evaluation protocol) is held identical to the
partner branch, so the two sets of results are directly comparable.

---

## 1. What is different from the partner branch

| | Partner branch | This branch (`adnan-dev`) |
|---|---|---|
| Model | Qwen3.5-4B (32 layers) | DeepSeek-R1-Distill-Qwen-7B (28 layers) |
| Model type | Instruction model | **Reasoning model** (emits `<think>…</think>`) |
| Probe | Per-layer logistic regression (14 independent probes) | **CLAP** — one cross-layer attention transformer |
| Layer selection | Manual (report the best single layer) | **Learned** by attention (no manual choice) |
| Hidden dim | 2560 | 3584 |
| Pipeline / data / labels / LODO | — | Identical, for fair comparison |

The two changes are complementary. The new model tests whether the probe
trust signal **replicates across architectures** (the strongest robustness
check flagged in `Updates_3.md` §6). The new probe tests whether we can drop
the **manual layer-selection** step that the per-layer approach depends on.

---

## 2. The CLAP probe

CLAP is taken from Suresh et al., *"Cross-Layer Attention Probing for
Fine-Grained Hallucination Detection"* (TRUST-AI @ ECAI 2025,
arXiv:2509.09700). The original paper detects hallucinations in open-domain QA
by probing the EOS token's hidden states. Here the same architecture is
applied to **per-field extraction error detection**, at the last content token
of each field value in the generated JSON — the exact `last_token` position the
project's logistic-regression probes already use.

**Architecture** (implemented in `src/probe_extraction/probes/clap.py`):

1. Down-project each layer's activation vector from the model hidden size to a
   small `d_model` (128), with LayerNorm.
2. Prepend a learnable CLS token, giving a sequence of `(1 + n_layers)` vectors.
3. Pass through a small Transformer encoder (pre-norm, full self-attention)
   so every layer can attend to every other layer.
4. Classify from the CLS output only.

**Why this matters for the project.** The per-layer approach trains 14
separate probes and then *we* pick the best layer after the fact. CLAP instead
receives all 14 layers at once and the attention mechanism learns which layers
carry the trust signal — no manual layer selection. If CLAP's
leave-one-document-out (LODO) AUROC matches or beats the best hand-picked
single-layer LODO AUROC from Stage 5, that is evidence the signal can be
extracted **without** the layer-selection step, which is a cleaner and more
defensible method.

**Why DeepSeek-R1 specifically.** It is a distilled reasoning model: its
hidden states during JSON generation already encode a completed chain of
thought. Cross-layer probing of a reasoning model for extraction trust has not
been studied. A useful by-product is interpretability — the CLAP attention
weights can show whether the trust signal lives in the early/mid "reasoning"
layers or the late "answer-writing" layers. That is a new question the
per-layer setup cannot ask as directly.

---

## 3. Implementation

New code on this branch:

- `src/probe_extraction/probes/clap.py` — the CLAP model, training loop
  (AdamW + cosine schedule + class-balanced `pos_weight` + early stopping),
  5-fold CV, and a single-fold fit used by the LODO stage.
- `scripts/06_train_clap.py` — Stage 6: trains CLAP with 5-fold CV and an
  80/20 held-out split. Writes `probes/clap_probe.pt` and
  `results/clap_summary.json`.
- `scripts/07_lodo_clap.py` — Stage 7: the honest, document-level metric.
  Trains CLAP once per held-out document and writes `results/lodo_clap.json`,
  directly comparable to Stage 5's per-layer `lodo_cv.json`.
- `run_clap.sh` — SLURM job runner for Stages 6–7 (CPU-only, A40devel).
- `configs/exp_deepseek_r1_7b_pooled.yaml` — the experiment config.
- `run_deepseek_extract.sh` / `run_deepseek_analysis.sh` — the DeepSeek
  Stage 1 (GPU) and Stages 2–5 (CPU) runners.

CLAP reuses the **same** `.npz` activation files and `labels/` JSON produced by
Stages 1–2 — no re-extraction is required to switch probe types.

**Layer mapping.** The partner's Qwen probe captures 14 layers from a 32-layer
model: `[1,4,8,12,14,16,18,20,22,24,26,28,30,32]`. DeepSeek-R1-7B has 28
layers, so each index was scaled by 28/32 and rounded to give equivalent
coverage density: `[1,4,7,10,12,14,16,18,19,21,23,24,26,28]`. This keeps the
cross-model comparison fair rather than capturing more or fewer layers.

---

## 4. Status

**The pipeline is implemented and validated end-to-end on a single document.**
A one-document GPU run of Stage 1 confirmed:

- The model loads correctly on the A40 GPU (`device=cuda:0`), bf16, 28 layers,
  hidden dim 3584.
- It generates and stops cleanly (1,433 tokens, `finish=stop`) and the JSON
  parses into 11 fields with no parse error.
- Activations are captured at all 14 configured layers and saved.

**The full 28-document extraction run is in progress on the Bender cluster.**
Once it completes, Stages 2–5 (labelling + per-layer probes + per-layer LODO)
and Stages 6–7 (CLAP + CLAP LODO) will run, and the headline comparison —
**CLAP LODO AUROC vs. the best single-layer LODO AUROC** — will be filled in.

### Results — pending the current run

| Method | LODO AUROC | Status |
|---|---|---|
| Baseline — token logprob (min/mean) | — | pending |
| Per-layer logistic regression (best layer) | — | pending |
| **CLAP (all layers, attention)** | — | pending |

*(Table to be completed once the extraction + analysis + CLAP jobs finish.)*

---

## 5. What's next

1. **Complete the run** and report the CLAP-vs-per-layer LODO comparison
   (Section 4 table).
2. **Inspect CLAP's attention weights** to see which layers it relies on —
   reasoning layers vs. answer-writing layers — and compare against the
   per-layer LODO peak (layer 18 on Qwen in `Updates_3.md`).
3. **Cross-model read-across:** with both Qwen3.5-4B (per-layer) and
   DeepSeek-R1-7B (per-layer + CLAP) results in hand, assess whether the probe
   trust signal and its preferred network depth replicate across a standard
   instruction model and a reasoning model.

---

## 6. How to reproduce

```bash
# Stage 1 — extraction (GPU, A40medium). --exclude avoids the faulted node.
sbatch --exclude=node-02 run_deepseek_extract.sh

# Stages 2–5 — labelling + per-layer probes + per-layer LODO (CPU).
sbatch run_deepseek_analysis.sh

# Stages 6–7 — CLAP train + CLAP LODO (CPU).
sbatch run_clap.sh
```

Outputs land in `artifacts/deepseek_r1_7b_pooled/results/`:
`comparison.json` and `lodo_cv.json` (per-layer), and `clap_summary.json` and
`lodo_clap.json` (CLAP).
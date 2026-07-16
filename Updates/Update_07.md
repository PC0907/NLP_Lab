# Probe‑Based Trust Signals for Structured Information Extraction
## Weekly Update 07 — DeepSeek‑R1 / Reasoning‑Trace Track

**Author:** Adnan
**Date:** 2026‑07‑14
**Scope:** This week's delta. The headline experiment from Update 06's "Future
Work" — **field‑localized reasoning attribution** — is now implemented, verified,
run, and returns a **statistically significant** improvement. This completes the
three‑act arc of the reasoning‑trace track.

---

## 1. What we set out to do this week

Update 06 ended with a clean **negative** result: fusing the *document‑level*
reasoning vector with the answer‑token probe did **not** help (Δ ≈ +0.006 to
+0.013, not significant), because a document‑level vector is constant across a
document's fields and therefore cannot re‑rank fields *within* a document.

The planned fix: **localize the reasoning to each field** — for every extracted
value, find where that value is mentioned inside the `<think>…</think>` trace and
pool the hidden states of exactly those tokens. That produces a reasoning feature
that **varies per field**, so it *can* legitimately change within‑document ranking.

---

## 2. What we did

1. **Verified the implementation** (built in the previous cycle): 31/31 unit tests
   pass — attribution core (8), reasoning‑trace boundary/pooling (6),
   reasoning‑fusion harness (3), structure‑aware labeling (9), SOB loader (5).
2. **Re‑extraction with per‑token reasoning capture.** The first SOB run only
   stored *pooled* reasoning vectors. We re‑ran DeepSeek‑R1 extraction on the same
   300 SOB documents with per‑token reasoning states captured for 4 analysis
   layers (16, 19, 23, 26), saved as `__reasoning_tokens__layerN` arrays plus a
   `{doc_id}.rtokens.json` token‑string sidecar. Greedy decoding → the JSON and
   labels reproduce the first run exactly. New experiment name
   `deepseek_r1_7b_sob_attr` (the first run is preserved).
3. **Ran Stage 07** — field‑localized attribution LODO with a **paired Wilcoxon
   significance test** vs the answer‑only probe, over 5 feature variants.

---

## 3. New results — field‑localized attribution WORKS

**Setup:** 295 docs, 1,557 fields, 32.9% error rate, LODO, 199 valid folds.
**86.8% of extracted values are mentioned in the reasoning trace.**

Core comparison at the peak layer (19), paired test vs `answer`:

| Variant                         | per‑doc AUROC | Δ vs answer | paired p | pooled‑OOF |
|---------------------------------|--------------:|------------:|---------:|-----------:|
| answer (baseline)               | 0.8118        | —           | —        | 0.8422     |
| **fused_attr** (answer + attribution vec) | **0.8325** | **+0.0207** | 0.059 | 0.8561 |
| **fused_both** (answer + attribution + scalars) | **0.8345** | **+0.0227** | **0.044 ✅** | 0.8553 |
| fused_scalars (answer + scalars only) | 0.8108   | −0.001      | 0.59     | 0.8418     |
| scalars_only                    | 0.7338        | −0.078      | 0.002    | 0.7167     |

**Robustness — the effect is consistent at every layer** (per‑doc AUROC, Δ vs answer):

| Layer | answer | fused_attr Δ | fused_both Δ |
|------:|-------:|-------------:|-------------:|
| 16    | 0.785  | +0.030       | +0.031       |
| 19    | 0.812  | +0.021       | +0.023       |
| 23    | 0.799  | +0.023       | +0.020       |
| 26    | 0.778  | +0.012       | +0.018       |

All 8 deltas positive (+0.012 to +0.031); pooled‑OOF is the same story
(+0.009 to +0.015 at every layer).

---

## 4. What the results mean

1. **The mechanism is now legitimate.** Field‑localized reasoning varies across a
   document's fields, so unlike document‑level pooling it *can* — and does —
   improve within‑document error detection.
2. **The attribution *vector* is the driver, not the hand‑made scalars.**
   `fused_scalars` ≈ answer (adds nothing) and `scalars_only` is weak (0.73). The
   lift comes from the hidden states of the value‑mention reasoning tokens.
3. **Significant and consistent, though modest.** `fused_both` reaches p = 0.044;
   `fused_attr` is borderline (p = 0.059). The effect is small (~+0.02 AUROC) but
   positive at all four layers and both metrics — corroboration beyond one p‑value.
4. **Honest caveat:** with 4 variants, a strict multiple‑comparison correction
   would soften the single p‑value. The remedy is statistical power — more
   documents — which is the immediate next step.

**The three‑act story is now complete:**
1. Answer‑token probe detects errors (0.85 CV / 0.81–0.84 LODO), beats log‑prob
   baselines (+0.08). *(strong main result)*
2. Document‑level reasoning fusion = null, with a mechanistic reason. *(honest negative)*
3. Field‑localized reasoning attribution = significant, consistent gain where
   pooling failed. *(the novel positive the null motivated)*

---

## 5. Next steps

1. **Scale to ~1,000 SOB documents** — the single highest‑value move: tightens the
   confidence intervals and should push `fused_attr` and `fused_both` clearly under
   0.05 (ideally 0.01) and survive multiple‑comparison correction.
2. **Selective regeneration** — feed the improved probe into the regeneration
   framework (the project's original cost‑quality goal, and Ali's ExtractBench
   result) and measure the practical payoff.
3. **Cross‑dataset replication** — reproduce the attribution effect on ExtractBench
   with DeepSeek‑R1 for independent confirmation.

---

## 6. Result artifacts (this week)
- `artifacts/deepseek_r1_7b_sob_attr/labels/_definition_comparison.json` — labeling
  error rates (identical to the first run: strict 44.4% / structure_aware 41.4%).
- `artifacts/deepseek_r1_7b_sob_attr/results/reasoning_attribution_lodo.json` — the
  Stage 07 attribution LODO + paired significance (the tables above).
- Per‑token reasoning states: `activations/{doc_id}.npz`
  (`__reasoning_tokens__layer{16,19,23,26}`) + `{doc_id}.rtokens.json`.

## 7. Reproduce
```bash
sbatch run_sob_attr_extract_a100.sh      # GPU re-extraction w/ per-token reasoning capture
sbatch run_sob_attr_analysis_a100.sh     # 02_label -> 07_reasoning_attribution_lodo
```

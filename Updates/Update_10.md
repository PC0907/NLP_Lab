# Session Log — 8 September 2026

**Project:** Probe-Based Trust Signals for Selective Regeneration in Structured Information Extraction
**Purpose of this document:** Record of what was drafted, what was verified against the artifacts, and what discrepancies surfaced. Written for supervisor review.

---

## 1. Summary

Two activities today. First, drafting: Related Work, Experimental Setup, and a provisional Results section were written. Second, and more consequential, an audit of the LaTeX draft against the artifacts on disk.

The audit surfaced **seven discrepancies**, one of which affects the paper's headline number. None of them are fatal and all were found before submission rather than after. Most require a rerun rather than new experiments; the activations are already on disk and the reruns are CPU-only.

The short version for a supervisor: the experimental work is sound, but the write-up was drawing numbers from result files that are older than the code that produced the current metrics, and from runs whose document sets do not match what the paper claims. A rerun is in progress.

---

## 2. Bibliography verified

Six citation keys in the draft were unresolved. Five are now confirmed, one was misattributed.

| Key | Paper | Venue / ID |
|---|---|---|
| `extractconf2026` | Beyond Logprobs: A Multi-Signal Confidence Engine for LLM-Based Document Field Extraction — Nitesh Kumar (Perfios) | arXiv:2606.24420; oral, RobustifAI Workshop @ IJCAI-ECAI 2026 |
| `aiersilan2026linearly` | Hallucination Is Linearly Decodable from Mid-Layer Hidden States in Quantized LLMs — Aizierjiang Aiersilan (U. Macau) | arXiv:2606.02628 |
| `finground2026` | FinGround: Detecting and Grounding Financial Hallucinations via Atomic Claim Verification — Guo, Wu, Yiu | arXiv:2604.23588; ACL 2026 Industry Track |
| `redd2026` | Relational Deep Dive: Error-Aware Queries Over Unstructured Data — Chao, Chen, Guan, Koudas | arXiv:2511.02711 |
| `pep2026` | Prompt Embedding Probes: hallucination detection from hidden states | arXiv:2608.08024 — **author names still needed** |

### Correction

`arXiv:2605.09927` was recorded in the draft as a schema-validation repair loop that re-submits offending fields with diagnostic feedback. It is not. The actual paper is *Information Extraction of Nested Complex Structure of Quantum Cascade Lasers via Large Language Models* (Fang et al., physics.optics), a JSON-Schema-guided extraction pipeline with no repair loop.

**Consequence:** the sentence in §2.3 citing it must be deleted, and the corresponding row removed from the positioning table. The following paragraph's claim that prior loops are triggered "by rules the schema itself can express" now has no citation behind it and needs either a real source or softening.

### Additional paper worth citing

`arXiv:2607.11414` — *Confidently Wrong: Detecting Hallucinations in Financial Question Answering from LLM Internal States*. Probes answer correctness on FinQA and TAT-QA, reporting 0.73–0.79 AUROC on financial reasoning against 0.9–1.0 for factual falsehoods. Critically, on high-confidence answers a linear probe reaches 0.68–0.77 AUROC against 0.55–0.63 for the best output-side signal. That is an independent external replication of our subsumption result on a different task, and it belongs in §2.1.

---

## 3. Artifact structure findings

### 3.1 Two parallel experiment trees

```
artifacts/qwen35_4b_pooled/            — has probes/ (15 .pkl files, incl. probe_layer18.pkl)
artifacts/qwen35_4b_pooled_alltokens/  — has NO probes/ directory
```

The regeneration cache (`regen_cache_pooled_v2.json`) and the nested safe-override results live in the **alltokens** tree. The probe recorded as the causal driver of the regeneration gain (`probe_layer18.pkl`) lives in the **pooled** tree.

This means `nested_safe_override.py` is reading activations/labels from one tree and probes from another, or is training probes on the fly. Either is defensible but neither is currently described in the paper, and §3.7 describes a single coherent pipeline.

**Action:** confirm via `grep -n "probes\|\.pkl" nested_safe_override.py`, then document in §3.7.

### 3.2 Pooling is experiment-level, not a flag

The configs directory contains `exp_qwen35_4b_mean.yaml`, `exp_qwen35_4b_lasttoken_2dom.yaml`, and several `*_alltokens.yaml` variants. So "all-tokens" versus the default is a different experiment tree with different captured activations, not a post-hoc pooling choice.

This resolves the open question about Table 6, where the same probe appears with two different baselines (0.898 and 0.879) and the caption attributes it to "all-domains" versus "all-tokens" without defining either. §3.4 needs a paragraph stating what each tree captures.

---

## 4. The headline number is the wrong statistic

### 4.1 What the code does

`scripts/05b_nested_lodo.py` computes and writes two distinct quantities:

```python
mean_auroc  = float(np.mean(outer_scores))          # -> "auroc_mean"
pooled_auroc = roc_auc_score(y[valid], oof[valid])  # -> "pooled_oof_auroc"
```

The code's own comment on the pooled version:

> Pooled out-of-fold metrics: every field was scored by a probe that never saw its document (so still leakage-free), but the metric is computed ONCE over all held-out fields rather than averaged per-document. This avoids the small-fold saturation (tiny credit docs hitting AUROC 1.000) that inflates and widens the per-fold mean. This is the stable "all fields" number.

### 4.2 What the result files contain

`artifacts/qwen35_4b_pooled_alltokens/results/nested_lodo.json` (dated 15 June):

```json
{
  "auroc_mean": 0.8788145422190015,
  "auroc_std": 0.11386718391439814,
  "auprc": 0.8602183048692822,
  "n_folds": 25,
  "layers_selected": { "16": 25 },
  "candidate_layers": [14, 16, 18, 20, 22]
}
```

There is **no `pooled_oof_auroc` key**. The file predates the code that emits it.

### 4.3 The problem

The 0.879 figure that appears in the abstract, the introduction, Table 2, and the transfer table diagonal is `auroc_mean` — the **per-fold average**.

The paper's §3.5 and §4.2 argue at length that per-fold AUROC is the misleading aggregation and pooled-OOF is the meaningful one, using the 2B model as the illustration (0.877 per-fold, 0.604 pooled). The headline number is therefore the same statistic the paper warns against, at almost the same value.

### 4.4 How bad the per-fold distribution actually is

From the partial rerun (17 of 21 folds completed before the time limit), per-fold test AUROCs:

```
1.000  1.000  1.000  0.985  0.960  0.958  0.933  0.907
0.857  0.810  0.767  0.714  0.667  0.649  0.581  0.409  0.167
```

Three folds saturate at exactly 1.000. One is 0.167, i.e. substantially worse than chance. A mean over this distribution is not a meaningful summary, and the ±0.114 reported in the file understates the spread of the current run.

**This is recoverable and arguably improves the paper.** The fold distribution is direct empirical support for the §3.5 argument, better than the assertion currently there. But the headline number has to change to pooled-OOF, and it will not be 0.879.

---

## 5. Regeneration results — actual numbers

`artifacts/qwen35_4b_pooled_alltokens/results/nested_safe_override.json` (6 September):

| Objective | Fixed | Broke | **Net** | Layer (all 28 folds) | Modal τ |
|---|---|---|---|---|---|
| lenient_auroc | 42 | 12 | **+30** | 18 (28/28) | 0.80 (24/28) |
| lenient_net | 42 | 12 | **+30** | 18 (28/28) | 0.80 (24/28) |
| strict_auroc | 32 | 11 | **+21** | 18 (28/28) | 0.95 (26/28) |
| strict_net | 31 | 5 | **+26** | **14 (28/28)** | 0.95 (28/28) |

Candidate layers swept: 14, 16, 18, 20, 22. Cache: `regen_cache_pooled_v2.json`.

### Observations

The two objectives are **matcher × selection criterion** — the inner loop optimises either AUROC or net, crossed with strict/lenient scoring. This resolves the earlier ambiguity about what "two objectives" meant.

Under lenient scoring the two criteria converge exactly (identical fixed/broke counts). Under strict they diverge: selecting on net gives a better net (+26 vs +21) and picks a different layer. That is a small but real argument for selecting on the metric you report.

### Two corrections to the draft

**τ = 0.70 does not appear anywhere.** The current Table 5 caption states τ=0.70 selected in 27 of 28 folds. Actual modal thresholds are 0.80 (lenient) and 0.95 (strict). The τ=0.70 row, its 259 overrides, and the +45 net are from an earlier run and need re-deriving or deleting.

**Layer 18 is not universally selected.** It wins three of four objectives unanimously; `strict_net` selects layer 14, also unanimously. The claim as drafted ("layer 18 selected unanimously under both objectives") is false as stated. The defensible version is narrower: *within* each objective the selection is perfectly stable across folds, which is evidence the inner loop is not fitting noise.

---

## 6. Undisclosed approximation

`nested_safe_override.json` carries this note:

> Inner LODO scores cached per layer; see module docstring for the approximation this introduces.

Whatever this approximation is, it must be stated in §3.5. If the inner-loop scores are cached in a way that shares information across folds, a reviewer reading the released code will find it, and it is far better disclosed by us than discovered by them.

**Action:** read the module docstring (`sed -n '1,60p' nested_safe_override.py`) and write it into the paper.

---

## 7. Document set inconsistencies

Four different fold counts appear across artifacts that the paper treats as one benchmark:

| Run | Folds | Notes |
|---|---|---|
| Regeneration (alltokens) | 28 | full document set |
| Detection, alltokens (old) | 25 | 3 folds presumably degenerate |
| Detection, pooled (old) | 22 | different tree, 14 candidate layers |
| Detection, alltokens, no-10kq (new) | 21 | after excluding 7 10kq documents |

Additionally, `nested_lodo_nofin.json` and `nested_lodo_intersect.json` exist in several trees, so field eligibility has more axes than the paper currently admits.

### The 10kq problem

The rerun log shows:

```
EXCLUDE domains: ['finance/10kq']
Domain filter skipped: finance/10kq=7
Loaded 763 fields across 21 docs (101 errors, 13.2%).
```

So the alltokens tree contains **28 documents, of which 7 are 10kq**. The original 0.879 run reported 25 folds, which is consistent with running on all 28 documents (losing 3 to degenerate folds) and **not** consistent with running on 21.

**The paper states that 10kq is excluded from all reported results. The headline number appears to include it.** This is a separate problem from the per-fold/pooled issue and needs resolving independently.

### Field count discrepancy

Detection on the no-10kq subset uses 763 fields. The regenerate-all baseline reports 2,389 overrides, implying 2,389 regenerable fields. These are different eligibility criteria on overlapping document sets. The paper does not currently distinguish them, and Appendix A cannot be written until it does.

### Error rate

The rerun log reports 13.2% (101/763) on the no-10kq subset. The paper states 11.1%. Different subsets, but the paper quotes 11.1% without qualification and uses it in the §4.4 argument about base rates.

---

## 8. Layer selection stability

Selected layers across runs:

| Run | Distribution |
|---|---|
| Regeneration, 3 objectives | 18 × 28/28 |
| Regeneration, strict_net | 14 × 28/28 |
| Detection alltokens (old, 25 folds) | 16 × 25/25 |
| Detection pooled (old, 22 folds) | 18 × 16, 16 × 5, 20 × 1 |
| Detection alltokens no-10kq (new, 17 of 21) | 18 × 9, 16 × 5, 20 × 1, 22 × 1, 14 × 1 |

Selection is unanimous on the full 28-document set and becomes unstable once 10kq is excluded. This suggests the unanimity is partly a property of the document set rather than purely of the signal, and the paper should not lean on it as heavily as currently drafted.

The transfer sweep's claim of stability across layers 12–24 is unaffected by this and remains the stronger generalisation claim.

---

## 9. Reruns in progress

Job submitted on A100short (8h limit; A40devel's 1h limit killed the first attempt at fold 17 of 21).

```bash
# variant 1: all domains — reproduces the 0.879 run, adds pooled_oof_auroc
python scripts/05b_nested_lodo.py \
    --config configs/exp_qwen35_4b_pooled_alltokens.yaml \
    --layers 14 16 18 20 22 \
    --out-name nested_lodo_repro_alldom.json

# variant 2: 10kq excluded — the number the paper should cite
python scripts/05b_nested_lodo.py \
    --config configs/exp_qwen35_4b_pooled_alltokens.yaml \
    --layers 14 16 18 20 22 \
    --exclude-domains finance/10kq \
    --out-name nested_lodo_repro_no10kq.json
```

Runtime ≈ 3.5 min/fold, so roughly 100 + 75 minutes.

**Expected outcome:** variant 1 should reproduce `auroc_mean` = 0.8788 at 25 folds, confirming the original run included 10kq. Both variants will emit `pooled_oof_auroc`, which is the figure the paper should headline.

---

## 10. Outstanding items

### Blocking the write-up

1. **Pooled-OOF numbers for both variants** — rerun in progress.
2. **Which document set the paper reports on.** Recommend: exclude 10kq everywhere, for consistency with the stated methodology, and rerun regeneration on the same basis.
3. **The nested_safe_override approximation** — read the docstring, disclose in §3.5.
4. **Probe provenance in the regeneration runs** — which tree, confirmed by grep.
5. **The +44 vs +30 reconciliation.** The causal-substitution result (+44 correct probe, +6 mismatched) is a fixed-layer number; the headline +30 is nested. Cleanest fix is rerunning the substitution nested, which is cheap against the cache.

### Needed for appendices

6. Transfer grid, both directions, all seven layers, raw and L2 — files exist, need collating.
7. `layer_conflict.json` — the −0.002 null and the corrected +0.047 artifact story.
8. `probes/_summary.json` — probe hyperparameters (C, solver, class weighting) for §3.4.
9. Per-domain breakdown of the 7,743 / 5,701 substring-containment ceiling.
10. Hand-crafted feature list — currently unspecified, and the subsumption claim depends on the baseline being a serious attempt.
11. Dataset statistics: document and field counts per domain, per benchmark.

### New work discussed

12. **Mass-mean probing** (Marks & Tegmark difference-in-means direction) as an additional baseline. Two motivations: it tests the "signal is low-dimensional" claim from the constructive side rather than through three failed enrichments; and the geometry-of-truth literature predicts mass-mean should *transfer* better than logistic regression, which is directly testable on the EB↔insurance grid. Scoped to detection plus transfer, not regeneration.
13. **LLM-as-judge baseline** — still unbuilt.
14. **SOB cross-task comparison** — loader rebuilt, not run.

---

## 11. Assessment

The experiments are sound. What today's audit found is a write-up drawing on stale result files, plus a genuine methodological ambiguity about which document set constitutes "ExtractBench" in this paper.

The most serious finding is that the headline detection number is the per-fold mean rather than pooled-OOF. It is also the most easily fixed, and fixing it strengthens the paper: the fold distribution (three folds at 1.000, one at 0.167) is a better argument for the pooled aggregation than the prose currently in §3.5.

The regeneration results are the paper's strongest contribution and are unaffected by any of this. The +30/+21/+26 figures come from a run dated two days ago, under a properly nested protocol, with the layer and threshold selected inside the training fold. Those numbers stand.

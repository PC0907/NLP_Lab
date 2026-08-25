#!/bin/bash
#SBATCH --partition=A100medium
#SBATCH --export=NONE
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --job-name=sob_1k_final
#SBATCH --output=logs/sob_1k_final-%j.out
#SBATCH --error=logs/sob_1k_final-%j.err

# FINAL NUMBERS, with fused_decomposed as the method.
#
# The decomposition test made fused_decomposed the strongest detector on the
# 974-document corpus: per-doc 0.8276, +0.0364 over answer-only, p=1.7e-05
# (Holm 1.2e-04) -- half again the effect of fused_attr. Everything downstream
# was computed with the weaker fused_both, so the paper's headline cost-quality
# numbers understate the method.
#
# Two things here:
#   Stage 07  re-run including fused_decomposed, so the attribution table and
#             its paired tests quote the method the paper actually proposes,
#             at both pre-committed layers.
#   Stage 09  selective-regeneration curves rescored with fused_decomposed as
#             probe_fused -- these are the practical numbers for the abstract.

module load Python/3.12.3
module load CUDA/12.4.0
source ~/nlp_lab_a100/bin/activate
export PYTHONPATH=$HOME/NLP_Lab/src:$PYTHONPATH
export HF_DATASETS_OFFLINE=1
export JOBLIB_TEMP_FOLDER=$TMPDIR

set -euo pipefail
cd ~/NLP_Lab

CFG="configs/exp_deepseek_r1_7b_sob_attr_1k.yaml"
RES="artifacts/deepseek_r1_7b_sob_attr/results"

echo "=== ENVIRONMENT ==="
hostname; nproc

echo ""
echo "=== STAGE 07 (incl. fused_decomposed), layers 19 + 23 ==="
# Written to a new file so the existing reasoning_attribution_lodo.json, which
# the current Update_07 numbers came from, stays intact for comparison.
python scripts/07_reasoning_attribution_lodo.py --config "$CFG" \
    --layers 19 23 --jobs -1 \
    --out-name reasoning_attribution_lodo_final.json

echo ""
echo "=== STAGE 09 rescored with fused_decomposed ==="
python scripts/09_selective_regeneration_sob.py --config "$CFG" \
    --layer 19 --fused-variant fused_decomposed --jobs -1 \
    --out-name selective_regeneration_final.json

echo ""
echo "=== FINAL NUMBERS FOR THE PAPER ==="
python - <<'PY'
import json, pathlib
R = pathlib.Path("artifacts/deepseek_r1_7b_sob_attr/results")

a = json.load((R / "reasoning_attribution_lodo_final.json").open())
print(f"corpus: {a['n_docs']} docs, {a['n_fields']} fields, "
      f"{a['n_errors']} errors ({a['n_errors']/a['n_fields']:.1%}), "
      f"{a['pct_value_mentioned']:.1%} of values mentioned")
print("\nper-doc AUROC by layer:")
for v, per in a["per_layer"].items():
    row = "  ".join(f"L{L} {r['per_doc_auroc_mean']:.4f}"
                    for L, r in sorted(per.items(), key=lambda kv: int(kv[0])))
    print(f"  {v:<18} {row}")
print("\npaired vs answer (per-doc, layer 19):")
for v, s in a["significance_vs_answer"].items():
    if s["p_value"] is not None:
        print(f"  {v:<18} d={s['mean_delta']:+.4f}  p={s['p_value']:.4g}  n={s['n_pairs']}")

r = json.load((R / "selective_regeneration_final.json").open())
print(f"\npooled AUROC: " + " | ".join(
    f"{k} {v:.4f}" for k, v in r["pooled_auroc"].items() if v is not None))
print("\nselective regeneration @20% budget:")
for regime, h in r["headline"].items():
    print(f"  {regime:<8} fused {h['probe_fused_recall']:.1%} | "
          f"answer {h['probe_answer_recall']:.1%} | "
          f"{h['best_logprob_baseline']} {h['baseline_recall']:.1%} | "
          f"break-even repair {h['break_even_repair_rate']:.2f}")
base = r["error_rate"]
for regime in ("per_doc", "global"):
    rows = r["curves"][regime]["probe_fused"]["rows"]
    row = next((x for x in rows if abs(x["budget"] - 0.20) < 1e-9), None)
    if row:
        fe = row["final_error_rate"]
        k = "0.7" if "0.7" in fe else sorted(fe)[0]
        print(f"  {regime:<8} error rate {base:.1%} -> {fe[k]:.1%} at 20% budget "
              f"(repair rate {k}); AURC {r['curves'][regime]['probe_fused']['aurc']:.4f}")
PY
echo ""
echo "Next: python scripts/10_make_figures.py --config $CFG --controls-name decomposition_test.json"
echo "Done."

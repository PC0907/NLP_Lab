#!/bin/bash
#SBATCH --partition=A100medium
#SBATCH --export=NONE
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --job-name=sob_st07_fix
#SBATCH --output=logs/sob_st07_fix-%j.out
#SBATCH --error=logs/sob_st07_fix-%j.err

# Repairs the one thing the previous job missed.
#
# run_sob_1k_final_a100.sh re-ran Stage 07 expecting fused_decomposed in the
# table. It was not there: fused_decomposed had been added to build_features()
# but not to the VARIANTS tuple the main loop iterates, so Stage 07 swept the
# original five variants and reproduced numbers we already had. Stage 09 was
# unaffected -- it takes the variant by name via --fused-variant -- so the
# selective-regeneration results from that job are correct and final.
#
# Only the missing variant is computed here, at both pre-committed layers, so
# this costs ~50 min rather than a full 2-hour sweep. Merge with the existing
# 5-variant table: the document set, folds and `answer` baseline are identical,
# so the rows are directly comparable.

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
echo "=== STAGE 07: answer + fused_decomposed, layers 19 + 23 ==="
python scripts/07_reasoning_attribution_lodo.py --config "$CFG" \
    --layers 19 23 --jobs -1 \
    --variants fused_decomposed \
    --out-name reasoning_attribution_decomposed.json

echo ""
echo "=== MERGED ATTRIBUTION TABLE (the paper's main result) ==="
python - <<'PY'
import json, pathlib
R = pathlib.Path("artifacts/deepseek_r1_7b_sob_attr/results")
a = json.load((R / "reasoning_attribution_lodo_final.json").open())   # 5 variants
b = json.load((R / "reasoning_attribution_decomposed.json").open())   # + decomposed

# Sanity: the two runs must describe the same corpus, or the rows are not
# comparable and the merged table would be quietly wrong.
for k in ("n_docs", "n_fields", "n_errors"):
    if a[k] != b[k]:
        raise SystemExit(f"MISMATCH on {k}: {a[k]} vs {b[k]} -- do not merge.")
print(f"corpus: {a['n_docs']} docs, {a['n_fields']} fields, "
      f"{a['n_errors']} errors ({a['n_errors']/a['n_fields']:.1%})")

per = dict(a["per_layer"])
per["fused_decomposed"] = b["per_layer"]["fused_decomposed"]
sig = dict(a["significance_vs_answer"])
sig["fused_decomposed"] = b["significance_vs_answer"]["fused_decomposed"]

order = ["answer", "fused_attr", "fused_decomposed", "fused_both",
         "fused_scalars", "scalars_only"]
print("\nper-doc AUROC (pooled in brackets):")
for v in order:
    if v not in per:
        continue
    cells = "  ".join(
        f"L{L} {r['per_doc_auroc_mean']:.4f} [{r['pooled_oof_auroc']:.4f}]"
        for L, r in sorted(per[v].items(), key=lambda kv: int(kv[0])))
    print(f"  {v:<18} {cells}")

print("\npaired vs answer (per-doc, layer 19):")
for v in order:
    s = sig.get(v)
    if s and s.get("p_value") is not None:
        print(f"  {v:<18} d={s['mean_delta']:+.4f}  p={s['p_value']:.4g}  n={s['n_pairs']}")

merged = dict(a)
merged["per_layer"], merged["significance_vs_answer"] = per, sig
merged["merged_from"] = ["reasoning_attribution_lodo_final.json",
                         "reasoning_attribution_decomposed.json"]
(R / "reasoning_attribution_merged.json").write_text(json.dumps(merged, indent=2))
print("\nSaved -> results/reasoning_attribution_merged.json")
PY
echo ""
echo "Next: python scripts/10_make_figures.py --config configs/exp_deepseek_r1_7b_sob_attr_1k.yaml --controls-name decomposition_test.json"
echo "Done."

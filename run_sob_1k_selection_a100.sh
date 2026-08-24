#!/bin/bash
#SBATCH --partition=A100medium
#SBATCH --export=NONE
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --job-name=sob_1k_selection
#SBATCH --output=logs/sob_1k_selection-%j.out
#SBATCH --error=logs/sob_1k_selection-%j.err

# THE SELECTION TEST -- resolves what the 1,000-document controls turned up.
#
# The scaled run showed the reasoning gain is REAL (+0.023 over answer-only,
# Holm p=0.022) and NOT a dimensionality artifact (shuffled and random controls
# both null). But `ctrl_docmean` -- the SAME attribution vectors collapsed to one
# vector per document -- matched field-localized attribution exactly
# (fused_attr_vs_docmean = -0.004, p = 0.71). So the effect is document-level,
# not field-localized: per-field assignment adds nothing over the document mean.
#
# That raises the question this job answers. Stage 6 already showed that pooling
# the WHOLE reasoning trace per document is null. ctrl_docmean pools only the
# VALUE-MENTIONING reasoning tokens, per document, and is NOT null. Both are one
# vector per document of identical dimensionality, so the only difference is
# WHICH TOKENS GET POOLED.
#
# Predicted, if token selection is what matters:
#   tracemean_vs_answer    ~ null      (reproduces Stage 6)
#   docmean_vs_answer      ~ +0.026    (already measured)
#   docmean_vs_tracemean   ~ positive  <-- the claim
#
# Four variants at one layer, so this is much cheaper than the full control run.

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

# --------------------------------------------------------------------------
# Stages 03/04 were never run for THIS experiment -- the per-layer CV probe and
# the probe-vs-log-prob comparison only exist under deepseek_r1_7b_sob, the
# original 300-document run. Every other number in the paper is now 974
# documents, so leaving the headline "probe beats baselines" claim on the old
# corpus would mean mixing corpus sizes across tables. These two stages put C1
# on the same footing as everything else. 5-fold CV over 14 layers is ~70 probe
# fits -- cheap next to LODO's 974 per variant.
# --------------------------------------------------------------------------
echo ""
echo "=== STAGE 03: per-layer CV probe (14 layers, 974 docs) ==="
python scripts/03_train_probe.py --config "$CFG"

echo ""
echo "=== STAGE 04: probe vs token-log-prob baselines ==="
python scripts/04_evaluate.py --config "$CFG"

echo ""
echo "=== SELECTION TEST: which reasoning tokens carry the signal? ==="
python scripts/08_attribution_controls.py --config "$CFG" \
    --layers 19 --jobs -1 --bootstrap 2000 \
    --variants answer fused_attr ctrl_docmean ctrl_tracemean \
    --out-name selection_test.json

echo ""
echo "=== C1 ON THE 974-DOC CORPUS (per-layer CV probe vs baselines) ==="
python - <<'PY'
import json, pathlib
A = pathlib.Path("artifacts/deepseek_r1_7b_sob_attr")
sp, cp = A / "probes" / "_summary.json", A / "results" / "comparison.json"
if sp.exists():
    d = json.load(sp.open())
    rows = d if isinstance(d, list) else d.get("layers", d)
    print("  probe AUROC by layer (5-fold CV):")
    try:
        items = rows.items() if isinstance(rows, dict) else [(r.get("layer"), r) for r in rows]
        for L, r in items:
            a = r.get("cv_auroc_mean", r.get("auroc")) if isinstance(r, dict) else r
            print(f"    layer {L}: {a:.4f}" if isinstance(a, float) else f"    layer {L}: {a}")
    except Exception as e:
        print("    (unexpected shape, dumping raw)", str(e)[:80]); print("   ", str(d)[:600])
else:
    print("  probes/_summary.json MISSING -- Stage 03 did not produce output.")
if cp.exists():
    print("  probe vs baselines:", json.dumps(json.load(cp.open()))[:800])
else:
    print("  results/comparison.json MISSING -- Stage 04 did not produce output.")
PY

echo ""
echo "=== VERDICT ==="
python - <<'PY'
import json, pathlib
d = json.load((pathlib.Path("artifacts/deepseek_r1_7b_sob_attr/results")
               / "selection_test.json").open())
sig = d["significance"]["19"]
per = d["per_layer"]["19"]

print("per-doc AUROC:")
for v in d["variants"]:
    print(f"  {v:<16} {per[v]['per_doc_auroc_mean']:.4f}  "
          f"(pooled {per[v]['pooled_oof_auroc']:.4f})")

print("\npaired tests (Holm-corrected):")
for k, s in sig.items():
    b = s.get("bootstrap") or {}
    ci = f"[{b['ci_low']:+.4f},{b['ci_high']:+.4f}]" if b else "n/a"
    print(f"  {k:<24} d={s['mean_delta']:+.4f} CI {ci} "
          f"p={s['p_value']:.4g} holm={s['p_holm']:.4g}")

dm = sig.get("docmean_vs_tracemean", {})
tm = sig.get("tracemean_vs_answer", {})
print("\nREAD:")
if dm.get("p_holm") is not None and dm["p_holm"] < 0.05 and dm["mean_delta"] > 0:
    print("  Value-mention tokens beat the whole trace at the SAME granularity.")
    print("  -> The claim is about WHICH TOKENS you pool, not which field you")
    print("     attach them to. That is the paper's contribution.")
else:
    print("  No significant gap between value-mention pooling and whole-trace")
    print("  pooling. -> The gain is a generic document-level reasoning summary;")
    print("     report it as such and lead with the mention statistic instead.")
if tm.get("p_value") is not None:
    print(f"  (whole-trace vs answer: p={tm['p_value']:.4g} -- Stage 6 "
          f"{'reproduced' if tm['p_value'] > 0.05 else 'NOT reproduced'})")
PY
echo "Done."

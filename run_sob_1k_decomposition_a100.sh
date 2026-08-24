#!/bin/bash
#SBATCH --partition=A100medium
#SBATCH --export=NONE
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --job-name=sob_1k_decomp
#SBATCH --output=logs/sob_1k_decomp-%j.out
#SBATCH --error=logs/sob_1k_decomp-%j.err

# THE FAIR TEST OF FIELD-LOCALIZED ATTRIBUTION.
#
# Everything so far said localization adds nothing: fused_attr matched
# ctrl_docmean (-0.004, p=0.71) and even the whole-trace mean matched both. But
# that comparison had a flaw. Write
#
#     attr_i = m_d + r_i
#
# where m_d is the document mean (large, shared, high variance across documents)
# and r_i is the field-specific residual (small). `fused_attr` hands the probe
# the SUM. With L2 over 7,168 correlated dimensions the optimizer rides the
# high-variance shared direction and the residual is effectively regularized
# away -- so fused_attr ~ ctrl_docmean is the expected outcome whether or not
# r_i carries signal. We never actually tested localization.
#
# This job tests it properly:
#   ctrl_centered      answer + r_i           does field-specific info help ALONE?
#   fused_decomposed   answer + m_d + r_i     does it help ON TOP of doc-level?
#                                             (separate blocks -> independent weights)
#   ctrl_docmean_pad   answer + m_d + noise   width-matched null for the above
#
# It also computes the ATTRIBUTION GEOMETRY, which says whether any of this can
# work: if a document's per-field attribution vectors have within-doc cosine
# ~1.0, they are the same vector wearing different labels and localization is
# null by construction rather than for want of signal in the model.
#
# Outcomes:
#   decomposed > docmean  -> localization DOES work; the earlier null was a
#                            measurement artifact, and the original thesis stands.
#   decomposed = docmean  -> localization genuinely adds nothing, and the
#                            geometry numbers explain exactly why.

module load Python/3.12.3
module load CUDA/12.4.0
source ~/nlp_lab_a100/bin/activate
export PYTHONPATH=$HOME/NLP_Lab/src:$PYTHONPATH
export HF_DATASETS_OFFLINE=1
export JOBLIB_TEMP_FOLDER=$TMPDIR

set -euo pipefail
cd ~/NLP_Lab

CFG="configs/exp_deepseek_r1_7b_sob_attr_1k.yaml"

echo "=== ENVIRONMENT ==="
hostname; nproc

echo ""
echo "=== DECOMPOSITION TEST (layer 19) ==="
python scripts/08_attribution_controls.py --config "$CFG" \
    --layers 19 --jobs -1 --bootstrap 2000 \
    --variants answer fused_attr ctrl_docmean ctrl_centered \
               fused_decomposed ctrl_docmean_pad \
    --out-name decomposition_test.json

echo ""
echo "=== VERDICT ==="
python - <<'PY'
import json, pathlib
d = json.load((pathlib.Path("artifacts/deepseek_r1_7b_sob_attr/results")
               / "decomposition_test.json").open())
g = d["attribution_geometry"]["19"]
per, sig = d["per_layer"]["19"], d["significance"]["19"]

print("GEOMETRY -- is there field-specific structure to find?")
print(f"  within-document cosine between fields' attr vectors: "
      f"{g['mean_within_doc_cosine']:.4f} (median {g['median_within_doc_cosine']:.4f})")
print(f"  residual/mean norm ratio: {g['residual_to_mean_norm']:.4f}")
print(f"  fields with an all-zero attr vector (value not in trace): "
      f"{g['frac_zero_attr']:.1%}")

print("\nper-doc AUROC:")
for v in d["variants"]:
    print(f"  {v:<20} {per[v]['per_doc_auroc_mean']:.4f}  "
          f"(pooled {per[v]['pooled_oof_auroc']:.4f})")

print("\npaired tests (Holm-corrected):")
for k, s in sig.items():
    b = s.get("bootstrap") or {}
    ci = f"[{b['ci_low']:+.4f},{b['ci_high']:+.4f}]" if b else "n/a"
    print(f"  {k:<24} d={s['mean_delta']:+.4f} CI {ci} "
          f"p={s['p_value']:.4g} holm={s['p_holm']:.4g}")

dd = sig.get("decomposed_vs_docmean", {})
cc = sig.get("centered_vs_answer", {})
pad = sig.get("decomposed_vs_pad", {})
print("\nREAD:")
rescued = (dd.get("p_holm") is not None and dd["p_holm"] < 0.05
           and dd["mean_delta"] > 0
           and (pad.get("p_holm") is None or pad["p_holm"] < 0.05))
if rescued:
    print("  Field-specific residual ADDS signal on top of the document-level")
    print("  component, and beats its width-matched noise control.")
    print("  -> LOCALIZATION WORKS. The earlier null was a measurement artifact")
    print("     (L2 crushing a small residual inside a large shared component).")
    print("     The original thesis stands; report fused_decomposed as the method.")
else:
    print("  The field-specific residual adds nothing beyond the document mean.")
    if g["mean_within_doc_cosine"] and g["mean_within_doc_cosine"] > 0.95:
        print(f"  Within-doc cosine is {g['mean_within_doc_cosine']:.3f}: a document's")
        print("  fields share essentially ONE vector, so there is no field-specific")
        print("  content to recover. Localization is null BY CONSTRUCTION -- that is")
        print("  a mechanism, not a shrug, and it is the paper's explanation.")
    else:
        print(f"  Within-doc cosine is {g['mean_within_doc_cosine']:.3f}, so the vectors")
        print("  DO differ -- the differences simply are not error-relevant.")
        print("  Next lever: pool a WINDOW around each mention rather than the")
        print("  mention itself (the reasoning, not the restatement).")
if cc.get("p_value") is not None:
    print(f"  (residual alone vs answer: d={cc['mean_delta']:+.4f}, p={cc['p_value']:.4g})")
PY
echo "Done."

#!/bin/bash
#SBATCH --partition=A100medium
#SBATCH --export=NONE
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --job-name=sob_1k_analysis
#SBATCH --output=logs/sob_1k_analysis-%j.out
#SBATCH --error=logs/sob_1k_analysis-%j.err

# CPU-only full analysis on the scaled corpus: label -> attribution LODO ->
# controls -> selective regeneration. Prereq: run_sob_1k_extract_a100.sh done.
#
# Sizing note: LODO cost grows with the SQUARE of the corpus (one probe fit per
# held-out document, each fit over all the others). 300 docs took ~25 min at 16
# cores for Stage 6; 1,000 docs is roughly an order of magnitude more work, so
# this asks for 32 cores and 8 hours. Stage 8's controls are restricted to the
# peak layer to keep that in budget -- the 4-layer robustness sweep already
# exists from the 300-doc run.

module load Python/3.12.3
module load CUDA/12.4.0
source ~/nlp_lab_a100/bin/activate
export PYTHONPATH=$HOME/NLP_Lab/src:$PYTHONPATH
export HF_DATASETS_OFFLINE=1
export JOBLIB_TEMP_FOLDER=$TMPDIR

set -euo pipefail
cd ~/NLP_Lab

CFG="configs/exp_deepseek_r1_7b_sob_attr_1k.yaml"
ART="artifacts/deepseek_r1_7b_sob_attr"

echo "=== ENVIRONMENT ==="
hostname; nproc; free -g | head -2
N_RT=$(ls ${ART}/activations/*.rtokens.json 2>/dev/null | wc -l)
echo "docs with per-token reasoning states: ${N_RT}"
if [ "${N_RT}" -lt 400 ]; then
  echo "FATAL: only ${N_RT} documents have reasoning tokens -- extraction is incomplete."
  echo "       Resubmit: sbatch run_sob_1k_extract_a100.sh"
  exit 1
fi

echo ""
echo "=== STAGE 02: Label (structure_aware) ==="
python scripts/02_label.py --config "$CFG"

echo ""
# Layer 19 was pre-committed from the 300-doc run, so restricting the scaled
# analysis to it (plus 23 as a confirmation layer) is both cheaper and more
# rigorous -- it removes the post-hoc layer-selection criticism. The 4-layer
# robustness sweep already exists at 300 docs and stays in the paper.
echo "=== STAGE 07: Field-localized attribution LODO (pre-committed layers) ==="
python scripts/07_reasoning_attribution_lodo.py --config "$CFG" \
    --layers 19 23 --jobs -1

echo ""
echo "=== STAGE 08: Controls (localization, Holm, bootstrap, mentions) ==="
python scripts/08_attribution_controls.py --config "$CFG" \
    --layers 19 --jobs -1 --shuffle-reps 3 --bootstrap 2000

echo ""
echo "=== STAGE 09: Selective regeneration cost-quality ==="
python scripts/09_selective_regeneration_sob.py --config "$CFG" \
    --layer 19 --fused-variant fused_both --jobs -1

echo ""
echo "=== KEY NUMBERS (scaled corpus) ==="
python - <<'PY'
import json, pathlib
res = pathlib.Path("artifacts/deepseek_r1_7b_sob_attr/results")

a = json.load((res / "reasoning_attribution_lodo.json").open())
print(f"corpus: {a['n_docs']} docs, {a['n_fields']} fields, "
      f"{a['n_errors']} errors ({a['n_errors']/a['n_fields']:.1%}), "
      f"{a['pct_value_mentioned']:.1%} of values mentioned in the trace")
print("\nATTRIBUTION vs answer (per-doc AUROC, paired Wilcoxon):")
for v, s in a["significance_vs_answer"].items():
    p = s["p_value"]
    print(f"  {v:<14} d={s['mean_delta']:+.4f}  p={p:.4g}  n={s['n_pairs']}"
          if p is not None else f"  {v:<14} n/a")

c = json.load((res / "attribution_controls.json").open())
m = c["mention_analysis"]
g_no, g_yes = m["groups"]["not_mentioned"], m["groups"]["mentioned_any"]
if g_no["error_rate"] is not None:
    print(f"\nMENTION: unmentioned values are wrong {g_no['error_rate']:.1%} of the time "
          f"(n={g_no['n']}) vs {g_yes['error_rate']:.1%} for mentioned (n={g_yes['n']})"
          + (f", risk ratio {m['risk_ratio']:.2f}, p={m['p_value']:.4g}"
             if m.get("risk_ratio") and m.get("p_value") is not None else ""))

print("\nCONTROLS (layer 19) -- is the gain about LOCALIZATION?")
for k, s in c["significance"].get("19", {}).items():
    b = s.get("bootstrap") or {}
    ci = f"[{b['ci_low']:+.4f},{b['ci_high']:+.4f}]" if b else "n/a"
    print(f"  {k:<26} d={s['mean_delta']:+.4f} CI {ci} "
          f"p={s['p_value']:.4g} holm={s['p_holm']:.4g}")

r = json.load((res / "selective_regeneration.json").open())
print("\nSELECTIVE REGENERATION @20% budget (errors caught):")
for regime, h in r["headline"].items():
    print(f"  {regime:<8} fused {h['probe_fused_recall']:.1%} | "
          f"answer {h['probe_answer_recall']:.1%} | "
          f"{h['best_logprob_baseline']} {h['baseline_recall']:.1%} | "
          f"break-even repair rate {h['break_even_repair_rate']:.2f}")
PY
echo "Done."

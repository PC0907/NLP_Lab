#!/bin/bash
#SBATCH --partition=A100medium
#SBATCH --export=NONE
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --job-name=sob_attr_controls
#SBATCH --output=logs/sob_attr_controls-%j.out
#SBATCH --error=logs/sob_attr_controls-%j.err

# CPU-only. Runs on the EXISTING deepseek_r1_7b_sob_attr artifacts -- no GPU,
# no re-extraction. Two things Update 07 left open:
#
#   Stage 08  the controls that turn "+0.02 AUROC" into a defensible claim:
#             is the gain about LOCALIZATION, or would any extra 3584 dims do?
#             Also Holm-corrects the family of p-values and bootstraps the CI,
#             and reports the probe-free mention statistic.
#   Stage 09  what the signal is WORTH: selective-regeneration cost-quality
#             curves vs the log-prob baselines, in both budget regimes.

module load Python/3.12.3
module load CUDA/12.4.0
source ~/nlp_lab_a100/bin/activate
export PYTHONPATH=$HOME/NLP_Lab/src:$PYTHONPATH
export HF_DATASETS_OFFLINE=1
export JOBLIB_TEMP_FOLDER=$TMPDIR

set -euo pipefail
cd ~/NLP_Lab

CFG="configs/exp_deepseek_r1_7b_sob_attr.yaml"
RES="artifacts/deepseek_r1_7b_sob_attr/results"

echo "=== ENVIRONMENT ==="
hostname; nproc; free -g | head -2

# Fail loudly and immediately if the per-token reasoning states are gone,
# rather than after a queue wait.
test -n "$(ls artifacts/deepseek_r1_7b_sob_attr/activations/*.rtokens.json 2>/dev/null | head -1)" || {
  echo "FATAL: no .rtokens.json sidecars found -- the attribution artifacts are missing."
  echo "       Re-run: sbatch run_sob_attr_extract_a100.sh"
  exit 1
}
echo "rtokens sidecars: $(ls artifacts/deepseek_r1_7b_sob_attr/activations/*.rtokens.json | wc -l)"

echo ""
echo "=== STAGE 08: attribution controls (localization, Holm, bootstrap, mentions) ==="
# Controls at all four layers so the robustness claim covers them too.
python scripts/08_attribution_controls.py --config "$CFG" \
    --layers 16 19 23 26 --jobs -1 --shuffle-reps 3 --bootstrap 2000

echo ""
echo "=== STAGE 09: selective regeneration cost-quality ==="
python scripts/09_selective_regeneration_sob.py --config "$CFG" \
    --layer 19 --fused-variant fused_both --jobs -1

echo ""
echo "=== KEY NUMBERS ==="
python - <<'PY'
import json, pathlib
res = pathlib.Path("artifacts/deepseek_r1_7b_sob_attr/results")

c = json.load((res / "attribution_controls.json").open())
m = c["mention_analysis"]
print("MENTION (probe-free signal):")
for k in ("mentioned_any", "not_mentioned"):
    g = m["groups"][k]
    er = g["error_rate"]
    print(f"  {k:<16} n={g['n']:<6} error rate {er:.1%}" if er is not None else f"  {k}: n/a")
print(f"  risk ratio {m['risk_ratio']:.2f}  p={m['p_value']:.4g}"
      if m.get("risk_ratio") and m.get("p_value") is not None else "  risk ratio: n/a")

print("\nCONTROLS (layer 19) -- does localization explain the gain?")
sig = c["significance"].get("19", {})
for k, s in sig.items():
    b = s.get("bootstrap") or {}
    ci = f"[{b['ci_low']:+.4f},{b['ci_high']:+.4f}]" if b else "n/a"
    print(f"  {k:<26} d={s['mean_delta']:+.4f} CI {ci} "
          f"p={s['p_value']:.4g} holm={s['p_holm']:.4g}")

r = json.load((res / "selective_regeneration.json").open())
print("\nSELECTIVE REGENERATION @20% budget:")
for regime, h in r["headline"].items():
    print(f"  {regime:<8} fused {h['probe_fused_recall']:.1%} | "
          f"answer {h['probe_answer_recall']:.1%} | "
          f"{h['best_logprob_baseline']} {h['baseline_recall']:.1%} | "
          f"break-even repair {h['break_even_repair_rate']:.2f}")
PY
echo "Done."

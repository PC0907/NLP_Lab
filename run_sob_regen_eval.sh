#!/bin/bash
#SBATCH --partition=A100medium
#SBATCH --export=NONE
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --job-name=sob_regen_eval
#SBATCH --output=logs/sob_regen_eval-%j.out
#SBATCH --error=logs/sob_regen_eval-%j.err

# STAGES 9 + 12 -- what real selective regeneration is worth (CPU, no GPU).
#
# Stage 9 runs first because Stage 12 ranks fields by Stage 9's out-of-fold
# probe scores, and Stage 9 now writes them to results/oof_field_scores.json.
# Running both in one job means the scores Stage 12 ranks by are the scores
# Stage 9 just reported -- they cannot drift apart between runs.
#
#   Stage 9  ~25 min  (LODO sweep, 32 cores)
#   Stage 12 ~minutes (labeling only)
#
# Stage 12 refuses to run if any scored document is missing its regeneration,
# or if re-labeling the ORIGINAL extractions fails to reproduce the labels
# Stage 2 stored. Both refusals are deliberate: the first stops a number from
# being computed on a subset, the second stops before/after being compared
# under different rules. If either fires, fix the cause -- do not work around it.
#
# Usage:
#   sbatch run_sob_regen_eval.sh        (after Stage 11 has finished)

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
RES="${ART}/results"

echo "=== ENVIRONMENT ==="
hostname; nproc
echo "regenerated documents: $(ls ${ART}/regen/*.json 2>/dev/null | wc -l)"

echo ""
echo "=== STAGE 09: simulated curves + per-field OOF score dump ==="
python scripts/09_selective_regeneration_sob.py --config "$CFG" --layer 19 --jobs -1

echo ""
echo "=== STAGE 12: MEASURED repair and damage ==="
python scripts/12_regen_evaluate.py --config "$CFG" --jobs -1

echo ""
echo "=== THE NUMBER THAT REPLACES THE ASSUMPTION ==="
python - <<'PY'
import json, pathlib
RES = pathlib.Path("artifacts/deepseek_r1_7b_sob_attr/results")
mea = json.loads((RES / "regen_evaluation.json").read_text())
sim_path = RES / "selective_regeneration.json"
sim = json.loads(sim_path.read_text()) if sim_path.exists() else {}

base = mea["baseline_error_rate"]
print(f"corpus: {mea['n_fields']} fields / {mea['n_docs']} documents, "
      f"baseline error rate {100*base:.1f}%")
u = mea["resample_usability"]
print(f"resamples: {u['usable']}/{u['total']} usable "
      f"({u['truncated']} truncated, {u['parse_error']} unparseable)")

for name, st in mea["strategies"].items():
    m = st["measured_rates"]
    print(f"\n--- strategy: {name} ---")
    print(f"  MEASURED repair rate {m['repair_rate']:.3f}  "
          f"(Stage 9 assumed 0.700)")
    print(f"  MEASURED damage rate {m['damage_rate']:.3f}  "
          f"(Stage 9 assumed 0.050)")
    print(f"  outcomes over all scored fields: {st['outcome_counts']}")
    print(f"  budget spent on fields where the model repeated itself: "
          f"{st['status_counts']['identical']} of {mea['n_fields']}")
    ctrl = st["control_full_resample"]["error_rate"]
    print(f"  CONTROL, resample the whole record (no probe): "
          f"{100*ctrl:.1f}% vs {100*base:.1f}% original")
    for regime, h in st["headline"].items():
        print(f"  {regime:<8} @20%: {100*h['baseline_error_rate']:.1f}% -> "
              f"{100*h['probe_fused_error_rate']:.1f}% (probe), "
              f"{100*h['logprob_error_rate']:.1f}% ({h['best_logprob_baseline']}), "
              f"{100*(h['probe_answer_error_rate'] or 0):.1f}% (answer-only)")
    worst = max((abs(v["difference_rate_points"]) for v in st["additivity_check"]),
                default=0.0)
    print(f"  additivity check: worst drift {100*worst:.2f} percentage points")

# The simulated curve is kept alongside so the paper can say exactly how far
# the invented parameters were off. Printing it is a convenience, not a result,
# so a shape change here must not fail a job whose output is already on disk.
try:
    h = sim.get("headline", {}).get("per_doc", {})
    row = next(r for r in sim["curves"]["per_doc"]["probe_fused"]["rows"]
               if abs(r["budget"] - 0.20) < 1e-9)
    print(f"\nfor reference, Stage 9's SIMULATED per-doc @20%: probe caught "
          f"{100*(h.get('probe_fused_recall') or 0):.1f}% of errors; assuming "
          f"repair 0.7 it predicted an error rate of "
          f"{100*row['final_error_rate']['0.7']:.1f}%")
except Exception as e:
    print(f"\n(could not summarize the simulated curve: {type(e).__name__}: {e})")
PY

echo ""
echo "Results:"
echo "  ${RES}/regen_evaluation.json   (measured)"
echo "  ${RES}/selective_regeneration.json (simulated, for comparison)"
echo "  ${RES}/oof_field_scores.json   (per-field OOF scores)"
echo "Done."

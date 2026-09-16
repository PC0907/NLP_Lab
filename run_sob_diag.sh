#!/bin/bash
#SBATCH --partition=A40devel
#SBATCH --time=0:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=sob_diag
#SBATCH --output=logs/sob_diag_%j.out
#SBATCH --error=logs/sob_diag_%j.err
source ~/NLP_Lab/setup_env.sh
set -euo pipefail
cd ~/NLP_Lab
python - <<'EOF'
import json, numpy as np
from collections import Counter
from pathlib import Path
A = Path("artifacts/qwen35_4b_sob_1k")
L = [14, 16, 18, 20, 22]

s = json.load(open(A / "extractions/_summary.json"))
ids = Counter(d["doc_id"] for d in s["per_document"])
dups = {k: v for k, v in ids.items() if v > 1}
print("== duplicates in extraction summary:", len(dups), "ids,", sum(v - 1 for v in dups.values()), "extra rows")

miss, ex, fields_by_field = Counter(), [], Counter()
for lp in sorted((A / "labels").glob("*.json")):
    if lp.name.startswith("_"):
        continue
    d = json.load(lp.open())
    with np.load(A / "activations" / f"{lp.stem}.npz") as z:
        keys = set(z.files)
    for f in d["labels"]:
        if not f.get("extracted_present", True):
            continue
        ps = f["path_str"]
        have = [x for x in L if f"{ps}__layer{x}" in keys]
        if len(have) == len(L):
            continue
        ev = f.get("extracted_value")
        vt = type(ev).__name__
        miss[(f["comparison_strategy"], vt, "err" if f["is_error"] else "ok",
              "some layers" if have else "no layers")] += 1
        if len(ex) < 8:
            near = sorted(k for k in keys if k.startswith(ps))[:3]
            ex.append((lp.stem[:20], ps, vt, str(ev)[:40], near))

print("== present fields missing activations:", sum(miss.values()))
for k, v in miss.most_common():
    print("  ", v, k)
print("== examples (doc, path, type, value, npz keys with same prefix):")
for e in ex:
    print("  ", e)
EOF
echo "== rest of error dump"
sed -n '21,110p' "$(ls -t logs/sob_probe_253344.out)"
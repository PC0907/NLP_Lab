#!/usr/bin/env python3
"""Recompute steering error rates over a FIXED document set.

THE PROBLEM THIS FIXES
----------------------
steering_experiment.py reports each condition's error rate over whatever
documents parsed in that condition. When steering breaks some records, the
denominator shrinks and the rate is no longer comparable to the baseline. In
the layer-18 probe run:

    probe@+0.40   648 fields   10.80%   (-3.48pp)   15% parse-fail
    probe@-0.50   223 fields   17.04%   (+2.75pp)   40% parse-fail

Both look like effects. Neither is interpretable: they are computed over
different, smaller document sets than the baseline's 770 fields. If the
documents that broke were error-heavy, dropping them lowers the rate
mechanically. That is survivorship, not intervention.

WHAT IT DOES
------------
Restricts every condition to the documents that parsed in ALL conditions, and
recomputes rates on that fixed set. Also reports, per document, whether it
survived each condition, so the pattern of breakage is visible rather than
summarised away.

Reads the per_doc records the experiment already saved. No GPU, no model.

Usage:
  python steering_intersection.py \
      artifacts/fresh_qwen35_4b_pooled_alltokens/results/steering_probe.json
  # several files (e.g. probe + random runs) can be passed together
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", help="steering_*.json result files")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    conditions = {}       # key -> {coeff, kind, per_doc}
    for f in args.files:
        data = json.load(open(f))
        for key, c in data["conditions"].items():
            if key in conditions:
                key = f"{key}[{Path(f).stem}]"
            conditions[key] = c

    if not conditions:
        print("No conditions found.")
        return 1

    # ---- documents that parsed everywhere -------------------------------
    survived = {}
    for key, c in conditions.items():
        survived[key] = {
            d for d, r in c["per_doc"].items() if not r.get("parse_failed")
        }
    all_docs = sorted(set().union(*survived.values()))
    common = sorted(set.intersection(*survived.values()))

    print("=" * 78)
    print("DOCUMENT SURVIVAL")
    print("=" * 78)
    print(f"documents appearing in any condition : {len(all_docs)}")
    print(f"documents parsing in EVERY condition : {len(common)}")
    print()

    keys = sorted(conditions, key=lambda k: (conditions[k]["kind"],
                                             conditions[k]["coeff"]))
    hdr = f"{'document':46}" + "".join(f"{conditions[k]['coeff']:+6.2f}" for k in keys)
    print(hdr)
    for d in all_docs:
        row = f"{d[:46]:46}"
        for k in keys:
            row += f"{'  ok  ' if d in survived[k] else '  --  '}"
        print(row)
    print()
    if len(common) < 5:
        print("WARNING: fewer than 5 documents survive everywhere. The fixed-set")
        print("comparison below rests on very little data; consider dropping the")
        print("most destructive conditions and recomputing over the rest.")
        print()

    # ---- rates on the fixed set -----------------------------------------
    print("=" * 78)
    print(f"ERROR RATES ON THE FIXED SET ({len(common)} documents)")
    print("=" * 78)

    rows = {}
    for k in keys:
        c = conditions[k]
        nf = ne = 0
        for d in common:
            r = c["per_doc"].get(d, {})
            nf += r.get("n_fields", 0)
            ne += r.get("n_errors", 0)
        rows[k] = (nf, ne, ne / nf if nf else float("nan"))

    base_key = next((k for k in keys
                     if conditions[k]["kind"] == "probe"
                     and abs(conditions[k]["coeff"]) < 1e-9), None)
    base_rate = rows[base_key][2] if base_key else None

    print(f"{'condition':16} {'fields':>8} {'errors':>8} {'err rate':>10} "
          f"{'vs baseline':>13} {'orig rate':>11} {'orig fields':>12}")
    for k in keys:
        nf, ne, rate = rows[k]
        c = conditions[k]
        delta = (rate - base_rate) if base_rate is not None else float("nan")
        print(f"{k:16} {nf:8} {ne:8} {100*rate:9.2f}% {100*delta:+12.2f}pp "
              f"{100*c['error_rate']:10.2f}% {c['n_fields']:12}")

    # ---- reading ---------------------------------------------------------
    print()
    print("=" * 78)
    print("READING")
    print("=" * 78)
    print("  The 'err rate' column is the comparable one: same documents, same")
    print("  fields, only the intervention differs. 'orig rate' is what the")
    print("  experiment printed, over whatever survived that condition.")
    print()
    print("  Where the two diverge, the original was a survivorship artefact.")
    print()
    if base_rate is not None:
        moved = [(k, rows[k][2] - base_rate) for k in keys
                 if conditions[k]["kind"] == "probe"
                 and abs(conditions[k]["coeff"]) > 1e-9]
        if moved:
            biggest = max(moved, key=lambda kv: abs(kv[1]))
            print(f"  Largest probe-direction shift on the fixed set: "
                  f"{biggest[0]} at {100*biggest[1]:+.2f}pp")
            nf, ne, _ = rows[base_key]
            print(f"  Baseline has {ne} errors in {nf} fields. As a rough guide,")
            print(f"  a shift of a few tenths of a percentage point here is a")
            print(f"  handful of fields and is not distinguishable from noise.")
            print()
    rand = [k for k in keys if conditions[k]["kind"] != "probe"]
    if rand:
        print("  Compare each probe condition against the RANDOM condition at the")
        print("  same coefficient. If they move together, the effect is")
        print("  perturbation. If the probe moves and random does not, that is")
        print("  the result.")
    else:
        print("  No random-direction conditions present. Without them, a shift")
        print("  in the probe conditions cannot be attributed to this direction")
        print("  rather than to perturbation in general -- the control is what")
        print("  makes the comparison an argument.")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "n_common_documents": len(common),
            "common_documents": common,
            "fixed_set_rates": {k: {"n_fields": rows[k][0],
                                    "n_errors": rows[k][1],
                                    "error_rate": rows[k][2],
                                    "kind": conditions[k]["kind"],
                                    "coeff": conditions[k]["coeff"]}
                                for k in keys},
        }, indent=2))
        print()
        print(f"Saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
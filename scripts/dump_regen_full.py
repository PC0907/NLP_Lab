#!/usr/bin/env python3
"""Print full regeneration logs: gold / original / regenerated + outcome, grouped
by outcome so you can scan all BREAKS together, all FIXES together, etc., and see
exactly what the model is doing wrong.

Uses the same matchers as rescore_regen (strict + lenient) so outcomes match.

Usage:
  python scripts/dump_regen_full.py \
      --domain qwen35_4b_pooled_alltokens \
      --cache artifacts/qwen35_4b_pooled_alltokens/results/regen_cache_pooled.json \
      [--mode lenient] [--only BROKE] [--max 50]
"""
from __future__ import annotations

import argparse
import json
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
_rs = import_module("rescore_regen")
strict_ok = _rs.strict_ok
lenient_ok = _rs.lenient_ok
is_scorable_lenient = _rs.is_scorable_lenient


def load_labels(domain):
    labels = {}
    for f in glob.glob(f"artifacts/{domain}/labels/*.json"):
        if "_summary" in f:
            continue
        d = json.load(open(f))
        for l in d["labels"]:
            labels[(d["doc_id"], l["path_str"])] = l
    return labels


def outcome(was_err, now_ok):
    if was_err and now_ok:        return "FIXED"
    if was_err and not now_ok:    return "still_wrong"
    if (not was_err) and not now_ok: return "BROKE"
    return "kept_ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--mode", choices=["strict", "lenient"], default="lenient")
    ap.add_argument("--only", choices=["FIXED", "BROKE", "still_wrong", "kept_ok", "all"],
                    default="all")
    ap.add_argument("--max", type=int, default=1000, help="max rows to print per group")
    args = ap.parse_args()

    cache = json.load(open(args.cache))
    labels = load_labels(args.domain)
    ok = strict_ok if args.mode == "strict" else lenient_ok

    groups = {"FIXED": [], "BROKE": [], "still_wrong": [], "kept_ok": []}
    for key, entry in cache.items():
        doc_id, path = key.split("::", 1)
        lab = labels.get((doc_id, path))
        if lab is None:
            continue
        gold = lab.get("gold_value")
        if args.mode == "lenient" and not is_scorable_lenient(path, gold):
            continue
        orig = lab.get("extracted_value")
        new = entry["new_value"]
        was_err = int(lab.get("is_error", 0)) == 1
        out = outcome(was_err, ok(gold, new))
        groups[out].append({
            "doc": doc_id.split("__")[-1][:30], "path": path,
            "gold": gold, "orig": orig, "new": new,
            "raw": entry.get("raw", ""),
        })

    print(f"\nMODE={args.mode}   cache={Path(args.cache).name}")
    print(f"counts: " + "  ".join(f"{k}={len(v)}" for k, v in groups.items()))

    show = ["FIXED", "BROKE", "still_wrong", "kept_ok"] if args.only == "all" else [args.only]
    for grp in show:
        rows = groups[grp]
        if not rows:
            continue
        print("\n" + "=" * 74)
        print(f"{grp}  ({len(rows)} rows" + (f", showing {min(len(rows), args.max)}" if len(rows) > args.max else "") + ")")
        print("=" * 74)
        for r in rows[:args.max]:
            print(f"\n  {r['path']}   [{r['doc']}]")
            print(f"    gold     : {str(r['gold'])[:100]!r}")
            print(f"    original : {str(r['orig'])[:100]!r}")
            print(f"    regen    : {str(r['new'])[:100]!r}")
            # raw model output (before JSON parse) often reveals the failure mode
            if r["raw"] and str(r["raw"])[:100] != str(r["new"])[:100]:
                print(f"    raw_out  : {str(r['raw'])[:100]!r}")


if __name__ == "__main__":
    main()
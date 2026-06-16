#!/usr/bin/env python3
"""Dump every regeneration candidate: gold, original extracted value, and the
regenerated value, with the outcome under strict and lenient scoring.

Lets you eyeball what the model actually produced vs what it had before, and
which outcomes are real fixes/breaks vs matcher artefacts.

Usage:
  python scripts/dump_regen.py \
      --domain qwen35_4b_pymupdf \
      --cache artifacts/qwen35_4b_pymupdf/results/regen_cache_fixed.json \
      [--out artifacts/qwen35_4b_pymupdf/results/regen_dump.tsv] \
      [--only fixed|broke|changed|all]
"""
from __future__ import annotations
import argparse, json, glob, sys
from pathlib import Path

sys.path.insert(0, "src")
# reuse the SAME scoring logic as rescore_regen so outcomes match exactly
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
    if was_err and now_ok:   return "FIXED"
    if was_err and not now_ok: return "still_wrong"
    if (not was_err) and not now_ok: return "BROKE"
    return "kept_ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default="all",
                    choices=["all", "fixed", "broke", "changed"])
    args = ap.parse_args()

    cache = json.load(open(args.cache))
    labels = load_labels(args.domain)

    rows = []
    for key, entry in cache.items():
        doc_id, path = key.split("::", 1)
        lab = labels.get((doc_id, path))
        if lab is None:
            continue
        gold = lab.get("gold_value")
        orig = lab.get("extracted_value")
        new = entry.get("new_value")
        was_err = int(lab.get("is_error", 0)) == 1
        s_ok = strict_ok(gold, new)
        l_scorable = is_scorable_lenient(path, gold)
        l_ok = lenient_ok(gold, new) if l_scorable else None
        s_out = outcome(was_err, s_ok)
        l_out = outcome(was_err, l_ok) if l_ok is not None else "unscored"

        rows.append({
            "doc": doc_id.split("__")[-1][:24],
            "path": path,
            "was_error": was_err,
            "gold": gold,
            "orig": orig,
            "new": new,
            "strict": s_out,
            "lenient": l_out,
            "changed": str(orig) != str(new),
        })

    # filter
    def keep(r):
        if args.only == "all": return True
        if args.only == "fixed": return r["strict"] == "FIXED" or r["lenient"] == "FIXED"
        if args.only == "broke": return r["strict"] == "BROKE" or r["lenient"] == "BROKE"
        if args.only == "changed": return r["changed"]
        return True
    rows = [r for r in rows if keep(r)]

    # console: readable blocks
    for r in rows:
        print(f"[{r['strict']:>11s} | {r['lenient']:>11s}]  {r['path']}")
        print(f"      gold: {str(r['gold'])[:80]!r}")
        print(f"      orig: {str(r['orig'])[:80]!r}")
        print(f"      new : {str(r['new'])[:80]!r}")
    print(f"\n{len(rows)} rows ({args.only}).")

    # optional TSV for spreadsheet inspection
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w") as f:
            f.write("doc\tpath\twas_error\tstrict\tlenient\tchanged\tgold\torig\tnew\n")
            for r in rows:
                f.write("\t".join(str(x).replace("\t", " ").replace("\n", " ")
                        for x in [r["doc"], r["path"], r["was_error"], r["strict"],
                                  r["lenient"], r["changed"], r["gold"], r["orig"], r["new"]]) + "\n")
        print(f"Wrote TSV: {outp}")


if __name__ == "__main__":
    main()
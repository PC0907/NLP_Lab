#!/usr/bin/env python3
"""Benchmark statistics for Section 3.1 and Appendix A.

Reads the per-document label files in a labels directory (skips
_summary.json) and prints: a filtering funnel, per-domain statistics,
and the error composition including substring containment.
Reads JSON only; no model, no GPU.
"""
import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

NUM_RE = re.compile(r"^[\s$€£]*-?[\d,]*\.?\d+\s*%?$")


def value_type(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        s = v.strip()
        if s.lower() in {"true", "false", "yes", "no"}:
            return "bool"
        if NUM_RE.match(s):
            return "number"
        return "string"
    return "other"


def is_substring_pair(gold, extracted):
    if gold is None or extracted is None:
        return False
    g, e = str(gold).strip().lower(), str(extracted).strip().lower()
    return g != e and (g in e or e in g)


def new_domain():
    return {"docs": 0, "labels": 0, "emitted": 0, "emitted_err": 0,
            "err_types": Counter(), "vtypes": Counter(), "gold_len": [],
            "max_depth": 0, "value_mismatch": 0, "substring": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--exclude-domains", nargs="*", default=[])
    ap.add_argument("--max-gold-chars", type=int, default=80)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    files = sorted(p for p in Path(args.labels_dir).glob("*.json")
                   if p.name != "_summary.json")
    funnel = Counter()
    doms = defaultdict(new_domain)

    for p in files:
        d = json.loads(p.read_text())
        domain = d.get("domain", "all")
        excluded = domain in args.exclude_domains
        s = doms[domain]
        s["docs"] += 1
        funnel["1_documents"] += 1
        if not excluded:
            funnel["2_documents_kept"] += 1

        for lab in d["labels"]:
            s["labels"] += 1
            s["err_types"][lab.get("error_type")] += 1
            depth = sum(1 for k in lab["path"] if isinstance(k, str))
            s["max_depth"] = max(s["max_depth"], depth)
            funnel["3_labelled_fields"] += 1

            if not lab.get("extracted_present"):
                continue  # omission: no generated tokens, nothing to probe
            gold = lab.get("gold_value")
            s["emitted"] += 1
            s["emitted_err"] += lab.get("is_error", 0)
            s["vtypes"][value_type(gold if gold is not None
                                   else lab.get("extracted_value"))] += 1
            if gold is not None:
                s["gold_len"].append(len(str(gold)))
            if lab.get("error_type") == "value_mismatch":
                s["value_mismatch"] += 1
                s["substring"] += is_substring_pair(gold,
                                                    lab.get("extracted_value"))
            funnel["4_emitted_fields"] += 1
            if excluded:
                continue
            funnel["5_emitted_kept_domains"] += 1
            if gold is None or len(str(gold)) <= args.max_gold_chars:
                funnel["6_emitted_kept_gold_le_max"] += 1

    print("=" * 72)
    print("FILTERING FUNNEL (excluded domains:", args.exclude_domains, ")")
    for k in sorted(funnel):
        print(f"  {k:32s} {funnel[k]:>8,}")

    print("=" * 72)
    print("PER DOMAIN (emitted fields only for error rate and types)")
    hdr = (f"  {'domain':22s} {'docs':>5s} {'labels':>7s} {'emitted':>8s} "
           f"{'err':>5s} {'rate':>6s} {'depth':>5s} {'medlen':>6s}")
    print(hdr)
    for dom, s in sorted(doms.items()):
        rate = s["emitted_err"] / s["emitted"] if s["emitted"] else float("nan")
        med = statistics.median(s["gold_len"]) if s["gold_len"] else float("nan")
        print(f"  {dom:22s} {s['docs']:>5d} {s['labels']:>7,} {s['emitted']:>8,} "
              f"{s['emitted_err']:>5d} {rate:>6.1%} {s['max_depth']:>5d} {med:>6.0f}")

    print("=" * 72)
    print("VALUE TYPES AND ERROR COMPOSITION")
    for dom, s in sorted(doms.items()):
        print(f"  {dom}")
        print(f"    value types  : {dict(s['vtypes'])}")
        print(f"    error types  : {dict(s['err_types'])}")
        print(f"    value_mismatch {s['value_mismatch']}, "
              f"of which substring containment {s['substring']}")

    if args.out:
        out = {"funnel": dict(funnel),
               "domains": {k: {**v, "err_types": dict(v["err_types"]),
                               "vtypes": dict(v["vtypes"]),
                               "gold_len": None}
                           for k, v in doms.items()}}
        Path(args.out).write_text(json.dumps(out, indent=2))
        print("Saved", args.out)


if __name__ == "__main__":
    main()
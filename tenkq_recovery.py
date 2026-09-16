#!/usr/bin/env python3
"""How many 10-K fields are legitimately answerable?

The 10-K domain is excluded from every headline result because its error rate
is 80-90%. The taxonomy showed why: 42% of its errors have gold absent from
the document text and 41% have neither value present, so no extractor could
have got them right. The `scale` and `value` fields carry no schema
description, so the benchmark's encoding convention is unstated.

This asks what remains once those are removed. The filters are deliberately
defined on the ANNOTATION, never on whether the model was correct:

  A. gold value present in the document text     (recoverable in principle)
  B. schema provides a description for the field  (convention is knowable)
  C. gold is non-empty                            (there is something to extract)

Filtering on correctness -- e.g. dropping fields that are often wrong -- would
remove exactly the positives the probe exists to detect, and is not done here.

Reports each criterion separately and in combination, with the resulting error
rate, so the decision rests on what the annotation supports rather than on a
target field count.

Read-only. No GPU.

Usage:  python tenkq_recovery.py [--domain fresh_qwen35_4b_pooled_alltokens]
"""
from __future__ import annotations
import argparse, json, glob, re
from pathlib import Path
from collections import defaultdict

PARSED = Path("data/parsed/pymupdf")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="fresh_qwen35_4b_pooled_alltokens")
    p.add_argument("--target-domain", default="finance/10kq")
    return p.parse_args()


def norm(s):
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r"[\u2010-\u2015\u2212]", "-", s)
    s = s.replace("\u00ad", "").replace("&amp;", "&")
    s = re.sub(r"[$,\s]+", " ", s)
    return s.strip()


def numeric_variants(v):
    """A number may be written many ways; try the plausible renderings."""
    out = set()
    try:
        f = float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return out
    for cand in (str(v), f"{f}", f"{f:.0f}" if f == int(f) else f"{f}",
                 f"{f:,.0f}" if f == int(f) else f"{f:,}"):
        out.add(norm(cand))
    return {c for c in out if c}


def schema_desc_paths(schema, prefix=""):
    """Set of dotted paths whose schema entry carries a description."""
    described = set()
    undescribed = set()
    if not isinstance(schema, dict):
        return described, undescribed
    for k, sub in schema.get("properties", {}).items():
        path = f"{prefix}.{k}" if prefix else k
        if not isinstance(sub, dict):
            continue
        (described if sub.get("description") else undescribed).add(path)
        t = sub.get("type")
        if t == "object":
            d, u = schema_desc_paths(sub, path)
            described |= d; undescribed |= u
        elif t == "array":
            items = sub.get("items", {})
            if isinstance(items, dict) and items.get("type") == "object":
                d, u = schema_desc_paths(items, path)
                described |= d; undescribed |= u
    return described, undescribed


def strip_indices(path_str):
    """authors.2.name -> authors.name, so it matches the schema path."""
    return ".".join(p for p in path_str.split(".") if not p.lstrip("-").isdigit())


def main():
    args = parse_args()
    labdir = Path(f"artifacts/{args.domain}/labels")

    stats = defaultdict(int)
    err = defaultdict(int)
    per_doc = {}
    undescribed_seen = defaultdict(int)

    for f in sorted(labdir.glob("*.json")):
        if f.name.startswith("_"):
            continue
        x = json.load(f.open())
        if x.get("domain") != args.target_domain:
            continue
        doc = x.get("doc_id", f.stem)
        txt_p = PARSED / f"{doc}.txt"
        if not txt_p.exists():
            print(f"  (no parsed text for {doc}, skipping)")
            continue
        text = norm(txt_p.read_text(encoding="utf-8", errors="replace"))
        described, undescribed = schema_desc_paths(x.get("schema", {}))

        d_tot = d_keep = d_err = 0
        for l in x.get("labels", []):
            if not l.get("extracted_present", True):
                continue
            g = l.get("gold_value")
            is_err = int(l.get("is_error", 0))
            spath = strip_indices(l["path_str"])

            gn = norm(g)
            has_gold = bool(gn)
            gold_in = False
            if has_gold:
                if len(gn) >= 3 and gn in text:
                    gold_in = True
                else:
                    gold_in = any(v in text for v in numeric_variants(g) if len(v) >= 2)
            has_desc = spath in described
            if not has_desc and spath in undescribed:
                undescribed_seen[spath] += 1

            stats["total"] += 1; err["total"] += is_err
            if has_gold:
                stats["A_nonempty"] += 1; err["A_nonempty"] += is_err
            if gold_in:
                stats["B_gold_in_text"] += 1; err["B_gold_in_text"] += is_err
            if has_desc:
                stats["C_described"] += 1; err["C_described"] += is_err
            if has_gold and gold_in:
                stats["AB"] += 1; err["AB"] += is_err
            if has_gold and gold_in and has_desc:
                stats["ABC"] += 1; err["ABC"] += is_err
                d_keep += 1; d_err += is_err
            d_tot += 1
        per_doc[doc] = (d_tot, d_keep, d_err)

    print("=" * 78)
    print(f"10-K RECOVERY: {args.domain}")
    print("=" * 78)
    labels = {
        "total":          "all extracted_present fields",
        "A_nonempty":     "A. gold non-empty",
        "B_gold_in_text": "B. gold appears in document text",
        "C_described":    "C. schema provides a description",
        "AB":             "A and B",
        "ABC":            "A and B and C  <- the defensible inclusion set",
    }
    print(f"{'criterion':46} {'fields':>7} {'errors':>7} {'err rate':>9}")
    for k, lab in labels.items():
        n, e = stats[k], err[k]
        print(f"{lab:46} {n:7} {e:7} {e/max(n,1):8.1%}")

    print()
    print("per document (all / kept by ABC / errors among kept):")
    for doc, (t, k, e) in sorted(per_doc.items()):
        print(f"  {doc[:44]:46} {t:5} / {k:5} / {e:5}  ({e/max(k,1):5.1%})")

    if undescribed_seen:
        print()
        print("fields with NO schema description (excluded by C), most frequent:")
        for p, n in sorted(undescribed_seen.items(), key=lambda kv: -kv[1])[:15]:
            print(f"  {p:56} {n:5}")

    print()
    print("=" * 78)
    clean = 738   # fresh nested-LODO field count, 10-K excluded
    add = stats["ABC"]
    print(f"clean domains (current evaluation set):        {clean:6} fields")
    print(f"recoverable 10-K fields (criteria A+B+C):      {add:6} fields")
    print(f"combined:                                      {clean + add:6} fields")
    print()
    print("Every criterion above is computed from the BENCHMARK alone -- gold")
    print("presence, schema descriptions, gold emptiness -- and never from")
    print("whether the model was correct. Filtering on the error rate would")
    print("remove the positives the probe exists to detect.")
    print()
    print("If the recovered error rate is broadly comparable to the clean")
    print("domains, these fields are answerable and belong in the evaluation.")
    print("If it is still near 90%, the remaining errors have a cause this")
    print("filter does not capture, and the domain should stay excluded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
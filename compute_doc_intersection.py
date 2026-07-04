#!/usr/bin/env python3
"""Compute the intersection of documents successfully labeled by ALL models,
so the scaling comparison (2B vs 4B vs 9B) is on an identical document set.

Each model truncates/OOMs on different documents, so their labeled-doc sets
differ. A fair comparison must restrict every model to the docs that ALL of
them produced. This script:

  1. reads each model's labels dir,
  2. optionally excludes a domain (finance/10kq),
  3. reports each model's docs and the intersection,
  4. prints the intersection composition by domain (so you see it's not
     silently collapsing to one domain),
  5. writes the intersection doc_id list to a file for --include-docs.

Usage:
    python compute_doc_intersection.py \\
        --labels artifacts/qwen35_2b_pooled/labels \\
                 artifacts/qwen35_4b_pooled_alltokens/labels \\
                 artifacts/qwen35_9b_pooled/labels \\
        --exclude-domains finance/10kq \\
        --out intersection_docs.txt
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_doc_domains(labels_dir: Path, exclude):
    """Return {doc_id: domain} for a labels dir, applying domain exclusion and
    requiring the doc to have BOTH an error and a non-error field (else LODO
    can't use it as a test fold anyway)."""
    out = {}
    for p in sorted(labels_dir.glob("*.json")):
        if p.name.startswith("_"):
            continue
        data = json.load(p.open())
        domain = data.get("domain")
        if exclude and domain in exclude:
            continue
        labels = data.get("labels", [])
        n = len(labels)
        n_err = sum(int(x.get("is_error", 0)) for x in labels)
        # keep only docs usable as a LODO test fold (both classes present)
        if n_err == 0 or n_err == n or n == 0:
            continue
        out[p.stem] = domain
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True,
                    help="Labels dirs, one per model.")
    ap.add_argument("--exclude-domains", nargs="*", default=None)
    ap.add_argument("--out", default="intersection_docs.txt")
    args = ap.parse_args()

    exclude = set(args.exclude_domains) if args.exclude_domains else set()

    per_model = []
    for ld in args.labels:
        d = load_doc_domains(Path(ld), exclude)
        per_model.append((ld, d))
        comp = Counter(d.values())
        print(f"\n{ld}")
        print(f"  usable docs (both classes, after exclusion): {len(d)}")
        for dom, c in sorted(comp.items()):
            print(f"    {dom}: {c}")

    # intersection of doc_ids across all models
    common = set(per_model[0][1].keys())
    for _, d in per_model[1:]:
        common &= set(d.keys())

    # domain composition of the intersection (use first model's domain map)
    dom_map = per_model[0][1]
    comp = Counter(dom_map[doc] for doc in common)

    print("\n" + "=" * 64)
    print(f"INTERSECTION: {len(common)} documents present & usable in ALL {len(per_model)} models")
    for dom, c in sorted(comp.items()):
        print(f"    {dom}: {c}")
    print("=" * 64)

    Path(args.out).write_text("\n".join(sorted(common)) + "\n")
    print(f"\nWrote {len(common)} doc_ids to {args.out}")
    print("Pass this to nested LODO via --include-docs-file", args.out)

    # also show what each model LOSES vs the intersection, for transparency
    print("\nDocs each model has but the intersection drops:")
    for ld, d in per_model:
        lost = sorted(set(d.keys()) - common)
        print(f"  {ld}: drops {len(lost)}")
        for doc in lost:
            print(f"     - {doc} ({d[doc]})")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Three read-only checks on the Qwen runs. No GPU, no model, file reads only.

1. REPRODUCIBILITY: fresh full-length run (grid_qwen35_4b_pymupdf) against the
   original artifacts (qwen35_4b_pooled), document by document. Same code
   path, same parsed text, temperature 0 -- field counts and error counts
   should match exactly. Any difference is worth explaining.

2. TRUNCATION EFFECT: full-length (600k) against truncated (300k) Qwen runs.
   If they match, the fields Qwen needs sit in the first 300k characters and
   truncation is immaterial for the primary result.

3. ERROR TAXONOMY: for every error, is the extracted value in the document?
   Is the gold? Does one contain the other? This decomposes the error rate
   into matcher-strictness artefacts, annotation defects, and genuine model
   errors -- and is the audit the pre-registration rule requires before any
   matcher change.

Usage:  python qwen_checks.py
"""
from __future__ import annotations
import json, glob, re, sys
from collections import defaultdict
from pathlib import Path

A = Path("artifacts")
PARSED = Path("data/parsed/pymupdf")


# ---------------------------------------------------------------------------
def load_labels(d):
    out = {}
    for f in sorted(glob.glob(f"{A}/{d}/labels/*.json")):
        if f.endswith("_summary.json"):
            continue
        x = json.load(open(f))
        doc = x.get("doc_id", Path(f).stem)
        L = [l for l in x.get("labels", []) if l.get("extracted_present", True)]
        out[doc] = {
            "domain": x.get("domain", "?"),
            "n": len(L),
            "e": sum(int(l.get("is_error", 0)) for l in L),
            "labels": L,
        }
    return out


def compare(name_a, a, name_b, b):
    print("=" * 78)
    print(f"{name_a}  vs  {name_b}")
    print("=" * 78)
    print(f"{'document':50} {name_a[:12]:>14} {name_b[:12]:>14}")
    same = diff = 0
    for k in sorted(set(a) | set(b)):
        ra = f"{a[k]['n']:4}/{a[k]['e']:4}" if k in a else "      ---"
        rb = f"{b[k]['n']:4}/{b[k]['e']:4}" if k in b else "      ---"
        eq = k in a and k in b and (a[k]["n"], a[k]["e"]) == (b[k]["n"], b[k]["e"])
        flag = "" if eq else "  <--"
        same += eq; diff += (not eq)
        print(f"{k[:50]:50} {ra:>14} {rb:>14}{flag}")
    print(f"\n{same} identical, {diff} differ (fields/errors)")
    print()


# ---------------------------------------------------------------------------
def _norm(s):
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r"[\u2010-\u2015\u2212]", "-", s)   # dashes
    s = s.replace("\u00ad", "")                     # soft hyphen
    s = re.sub(r"[$,\s]+", " ", s).strip()
    return s


def taxonomy(d):
    print("=" * 78)
    print(f"ERROR TAXONOMY: {d}")
    print("=" * 78)
    labs = load_labels(d)
    by_dom = defaultdict(lambda: defaultdict(int))
    total = defaultdict(int)
    examples = defaultdict(list)

    for doc, info in labs.items():
        txt_path = PARSED / f"{doc}.txt"
        if not txt_path.exists():
            continue
        text = _norm(txt_path.read_text(encoding="utf-8", errors="replace"))
        dom = info["domain"]
        for l in info["labels"]:
            if not int(l.get("is_error", 0)):
                continue
            g = _norm(l.get("gold_value"))
            x = _norm(l.get("extracted_value"))
            g_in = bool(g) and len(g) >= 3 and g in text
            x_in = bool(x) and len(x) >= 3 and x in text
            contains = bool(g) and bool(x) and len(g) >= 3 and len(x) >= 3 \
                and (g in x or x in g)

            if not g:
                cat = "gold_empty"
            elif contains:
                cat = "form_mismatch"          # one contains the other: matcher strictness
            elif g_in and not x_in:
                cat = "hallucination"          # gold present, extraction invented
            elif g_in and x_in:
                cat = "wrong_value"            # both present, model picked wrong one
            elif not g_in and x_in:
                cat = "gold_unreachable"       # gold not in text; annotation defect
            else:
                cat = "both_absent"
            by_dom[dom][cat] += 1
            total[cat] += 1
            if len(examples[cat]) < 3 and len(str(l.get("gold_value"))) < 40 \
                    and len(str(l.get("extracted_value"))) < 40:
                examples[cat].append((doc[:30], l["path_str"][:28],
                                      l.get("gold_value"), l.get("extracted_value")))

    cats = ["form_mismatch", "gold_unreachable", "both_absent", "wrong_value",
            "hallucination", "gold_empty"]
    n_all = sum(total.values()) or 1
    print(f"{'category':18} {'count':>7} {'share':>7}   meaning")
    meaning = {
        "form_mismatch":    "matcher strictness: extraction contains gold or vice versa",
        "gold_unreachable": "annotation defect: gold not in document text",
        "both_absent":      "annotation defect: neither value in text",
        "wrong_value":      "genuine model error: right value present, wrong one chosen",
        "hallucination":    "genuine model error: value invented",
        "gold_empty":       "gold is empty/null",
    }
    for c in cats:
        print(f"{c:18} {total[c]:7} {total[c]/n_all:6.1%}   {meaning[c]}")
    print(f"{'TOTAL':18} {n_all:7}")

    art = total["form_mismatch"] + total["gold_unreachable"] + total["both_absent"] + total["gold_empty"]
    gen = total["wrong_value"] + total["hallucination"]
    print(f"\nartefact-class (matcher + annotation): {art} ({art/n_all:.1%})")
    print(f"genuine-model-error class:             {gen} ({gen/n_all:.1%})")

    print(f"\n{'domain':30}" + "".join(f"{c[:10]:>11}" for c in cats))
    for dom in sorted(by_dom):
        n = sum(by_dom[dom].values()) or 1
        print(f"{dom:30}" + "".join(f"{by_dom[dom][c]/n:>10.0%} " for c in cats))

    print("\nexamples:")
    for c in cats:
        for doc, p, g, x in examples[c]:
            print(f"  [{c}] {doc} :: {p}")
            print(f"      gold={g!r}  extracted={x!r}")
    print()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    new_full = load_labels("grid_qwen35_4b_pymupdf")
    old      = load_labels("qwen35_4b_pooled")
    new_trunc = load_labels("trunc_qwen35_4b_pymupdf")

    if new_full and old:
        compare("fresh full-length", new_full, "original pooled", old)
    else:
        print("reproducibility check skipped: missing", 
              "grid_qwen35_4b_pymupdf" if not new_full else "qwen35_4b_pooled")

    if new_full and new_trunc:
        compare("full-length 600k", new_full, "truncated 300k", new_trunc)

    if new_full:
        taxonomy("grid_qwen35_4b_pymupdf")
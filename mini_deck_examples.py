#!/usr/bin/env python3
"""Mine concrete examples from saved artifacts for the supervisor deck.

Read-only. Login-safe (no computation). Produces:
  - one striking wrong-value example (for the "silent failure" hook)
  - one probe-flagged high-P(error) example (for the detection story)
  - one regen fix example (for the +48 story) -- read from the cache

Usage: python mine_deck_examples.py
"""
from __future__ import annotations
import json, glob, pickle
from pathlib import Path

REPO = Path.home() / "NLP_Lab"
LABELS_DIR = REPO / "artifacts/qwen35_4b_pooled_alltokens/labels"
CACHE = REPO / "artifacts/qwen35_4b_pooled_alltokens/results/regen_cache_pooled_v2.json"
FIXABILITY = REPO / "artifacts/qwen35_4b_pooled_alltokens/results/fixability.json"


def _load_labels():
    out = {}
    for f in sorted(LABELS_DIR.glob("*.json")):
        if f.name.startswith("_"):
            continue
        d = json.load(open(f))
        for fld in d.get("labels", []):
            out[(f.stem, fld["path_str"])] = fld
    return out


print("=" * 70)
print("EXAMPLE 1: SILENT-FAILURE HOOK (for slide 2)")
print("=" * 70)
print("A real ExtractBench field where the model produced a plausible-looking")
print("but wrong value. Prefer short-gold errors (numbers, dates, names).\n")

labels = _load_labels()
short_errs = []
for (doc, path), fld in labels.items():
    if int(fld.get("is_error", 0)) != 1:
        continue
    if not fld.get("extracted_present", True):
        continue
    gold = fld.get("gold_value")
    ext = fld.get("extracted_value")
    if gold is None or ext is None:
        continue
    gs, es = str(gold), str(ext)
    # short values -- numbers, dates, names -- more striking than long text
    if len(gs) > 30 or len(es) > 30:
        continue
    # skip identical-modulo-form cases (not silent failures, just annotation strictness)
    if gs.strip().lower() in es.strip().lower() or es.strip().lower() in gs.strip().lower():
        continue
    short_errs.append((doc, path, gold, ext))

print(f"Found {len(short_errs)} candidate short-value silent failures.\n")
for doc, path, gold, ext in short_errs[:6]:
    print(f"  doc  : {doc[:60]}")
    print(f"  path : {path}")
    print(f"  gold : {gold!r}")
    print(f"  model: {ext!r}   <-- silently wrong")
    print()


print("=" * 70)
print("EXAMPLE 2: PROBE-FLAGGED REGENERATION FIX (for slide 7/8)")
print("=" * 70)
print("A concrete before/after: model extracted wrong value, probe flagged it,")
print("regeneration recovered the correct value.\n")

if not CACHE.exists():
    print(f"Regen cache not found at {CACHE}")
else:
    cache = json.load(open(CACHE))
    fixed = []
    for key, entry in cache.items():
        try:
            doc, path = key.split("::", 1)
        except ValueError:
            continue
        lab = labels.get((doc, path))
        if lab is None:
            continue
        if int(lab.get("is_error", 0)) != 1:
            continue  # want an ORIGINAL error
        gold = lab.get("gold_value")
        ext = lab.get("extracted_value")
        new = entry.get("new_value")
        if None in (gold, ext, new):
            continue
        gs, es, ns = str(gold), str(ext), str(new)
        if len(gs) > 40 or len(es) > 40 or len(ns) > 40:
            continue
        # naive "fix": regen matches gold better than extraction did
        if ns.strip().lower() == gs.strip().lower() and es.strip().lower() != gs.strip().lower():
            fixed.append((doc, path, gold, ext, new))

    print(f"Found {len(fixed)} candidate 'error -> fixed by regen' cases.\n")
    for doc, path, gold, ext, new in fixed[:6]:
        print(f"  doc      : {doc[:60]}")
        print(f"  path     : {path}")
        print(f"  gold     : {gold!r}")
        print(f"  original : {ext!r}   <-- probe flagged (P(error) high)")
        print(f"  regen    : {new!r}   <-- FIXED")
        print()


print("=" * 70)
print("EXAMPLE 3: FIXABILITY BREAKDOWN (for slide 11 / annotation caveat)")
print("=" * 70)
print("Explains why the +48 ceiling is bounded by annotation strictness.\n")

if FIXABILITY.exists():
    fix = json.load(open(FIXABILITY))
    print(f"Total errors            : {fix.get('n_errors')}")
    print(f"Fixable (gold in text)  : {fix.get('n_fixable')}  ({fix['n_fixable']/fix['n_errors']:.0%})")
    print(f"Unfixable               : {fix.get('n_unfixable')}")
    print(f"\nBy method (how gold was found in text):")
    for m, n in fix.get("by_method", {}).items():
        print(f"  {m:24} {n}")
    print("\nNote: normalized_substr = extraction was a SUPERSET of gold")
    print("(model was arguably right; gold's exact form is the mismatch).")

    # a striking annotation-strictness example
    for row in fix.get("per_field", [])[:20]:
        if row.get("method") == "normalized_substr":
            print("\nExample of annotation-form mismatch (not a real error):")
            print(f"  doc  : {row['doc_id'][:60]}")
            print(f"  path : {row['path_str']}")
            print(f"  gold : {row['gold_value']!r}")
            print(f"  model: {row['extracted_value']!r}   <-- SUPERSET of gold")
            break

print("\n" + "=" * 70)
print("Done. Copy any of the above examples into the deck.")
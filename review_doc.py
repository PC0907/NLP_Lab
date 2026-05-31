#!/usr/bin/env python3
"""Show value_mismatch + hallucination errors for one document.

Usage:
    python review_doc.py <substring of doc name>
    python review_doc.py --list          # list all docs
    python review_doc.py <name> --all     # show ALL labels, not just errors
"""
import json
import glob
import sys

LABELS_DIR = "artifacts/qwen35_4b_pooled/labels"


def list_docs():
    print("Available documents:")
    for f in sorted(glob.glob(f"{LABELS_DIR}/*.json")):
        if "_summary" in f:
            continue
        print("   ", f.split("/")[-1].replace(".json", ""))


def show_doc(target: str, show_all: bool = False):
    matches = [
        f for f in glob.glob(f"{LABELS_DIR}/*.json")
        if "_summary" not in f and target.lower() in f.lower()
    ]
    if not matches:
        print(f"No label file matching: {target!r}\n")
        list_docs()
        return
    if len(matches) > 1:
        print(f"Multiple matches for {target!r} — be more specific:")
        for m in matches:
            print("   ", m.split("/")[-1].replace(".json", ""))
        return

    f = matches[0]
    doc = f.split("/")[-1].replace(".json", "")
    labels = json.load(open(f))["labels"]

    if show_all:
        rows = labels
        title = "ALL fields"
    else:
        rows = [l for l in labels if l["error_type"] in ("value_mismatch", "hallucination")]
        title = "errors (value_mismatch + hallucination)"

    print(f"=== {doc} ===")
    print(f"{len(rows)} {title}  (of {len(labels)} total fields)\n")

    for l in rows:
        marker = "" if l["is_error"] == 0 else "  *** ERROR ***"
        print(f"[{l['comparison_strategy']:14s}] [{l['error_type']:14s}] {l['path_str']}{marker}")
        print(f"      gold = {l['gold_value']!r}")
        print(f"      ext  = {l['extracted_value']!r}")
        print()


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] in ("--list", "-l"):
        list_docs()
    else:
        show_all = "--all" in args
        target = [a for a in args if not a.startswith("-")][0]
        show_doc(target, show_all=show_all)
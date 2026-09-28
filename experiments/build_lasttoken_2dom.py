"""Build a last_token dataset restricted to exactly the 8 documents that the
mean-position run succeeded on, so the position ablation is apples-to-apples.
"""
import json, shutil, sys, glob
from pathlib import Path

ARTIFACTS = Path("artifacts")
POOLED = ARTIFACTS / "qwen35_4b_lasttoken_2dom"
# last_token source experiments holding these domains' activations
SOURCES = ["qwen35_4b_pymupdf", "qwen35_4b_credit"]

def main() -> int:
    # The allowlist: doc_ids the mean run actually produced labels for.
    mean_labels = ARTIFACTS / "qwen35_4b_mean" / "labels"
    allow = set()
    for f in glob.glob(str(mean_labels / "*.json")):
        if Path(f).name.startswith("_"):
            continue
        allow.add(json.load(open(f))["doc_id"])
    print(f"Allowlist: {len(allow)} docs from the mean run")
    for d in sorted(allow):
        print(f"  {d}")

    for sub in ("labels", "activations", "extractions"):
        (POOLED / sub).mkdir(parents=True, exist_ok=True)

    copied = 0
    for exp in SOURCES:
        src = ARTIFACTS / exp
        for lab in glob.glob(str(src / "labels" / "*.json")):
            if Path(lab).name.startswith("_"):
                continue
            doc_id = json.load(open(lab))["doc_id"]
            if doc_id not in allow:
                continue
            stem = Path(lab).stem
            shutil.copy2(lab, POOLED / "labels" / Path(lab).name)
            for sub, ext in (("activations", ".npz"), ("extractions", ".json")):
                p = src / sub / f"{stem}{ext}"
                if p.exists():
                    shutil.copy2(p, POOLED / sub / p.name)
            copied += 1

    print(f"\nCopied {copied} docs into {POOLED}")
    if copied != len(allow):
        print(f"WARNING: {len(allow)-copied} allowlisted docs not found in "
              f"last_token sources — check domain coverage.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""Download the SOB (Structured Output Benchmark) text dataset to local disk.

Run this ONCE on the Bender LOGIN NODE (compute nodes have no internet). It
caches interfaze-ai/sob and saves it with datasets.save_to_disk so the SOB
loader can read it fully offline via load_from_disk.

Usage:
    python scripts/00_download_sob.py --out data/sob

Then in the experiment config:
    data:
      benchmark: "sob"
      benchmark_path: "data/sob"
      split: "test"

Dataset: interfaze-ai/sob (arXiv:2604.25359). Text subset derived from HotpotQA
(CC-BY-SA-4.0); benchmark code MIT. ~5,000 test records (~919 tokens context).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = "interfaze-ai/sob"
CONFIG = "default"  # the text / multi-hop subset


def main() -> int:
    ap = argparse.ArgumentParser(description="Cache SOB text dataset to disk.")
    ap.add_argument("--out", default="data/sob", help="Directory to save_to_disk.")
    ap.add_argument("--config", default=CONFIG, help="HF dataset config (default: text).")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` not installed. `pip install datasets` in your venv.")
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {REPO} (config={args.config}) ...")
    ds = load_dataset(REPO, args.config)   # DatasetDict: train/validation/test
    print("Splits + sizes:", {k: len(v) for k, v in ds.items()})

    ds.save_to_disk(str(out))
    print(f"Saved to {out.resolve()}")
    print("Set data.benchmark='sob', data.benchmark_path='%s', data.split='test'." % args.out)

    # Sanity peek at one record's shape.
    test = ds["test"] if "test" in ds else next(iter(ds.values()))
    ex = test[0]
    print("\nExample record keys:", list(ex.keys()))
    print("  question:", str(ex.get("question"))[:120])
    print("  json_schema keys:", list((ex.get("json_schema") or {}).get("properties", {}).keys()))
    print("  ground_truth:", str(ex.get("ground_truth"))[:160])
    return 0


if __name__ == "__main__":
    sys.exit(main())

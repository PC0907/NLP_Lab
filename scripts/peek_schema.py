#!/usr/bin/env python3
"""Print the schema node for a few fields, to see what context is available to
inject into the correction prompt (description? format? convention notes?).

Run from the repo root:
  python scripts/peek_schema.py --config configs/exp_qwen35_4b_10kq.yaml \
      --paths balance_sheet.goodwill.0.unit balance_sheet.goodwill.0.scale balance_sheet.goodwill.0.value
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
from probe_extraction.config import load_config

load_benchmark = import_module("01_extract").load_benchmark
schema_for_path = import_module("07_regen_single").schema_for_path

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--paths", nargs="+", required=True)
args = ap.parse_args()

cfg = load_config(args.config)
bench = load_benchmark(cfg)
doc = next(iter(bench))   # first doc; schema is shared per domain
print("doc:", doc.doc_id)
print("=" * 60)
# also print the top-level schema keys so we see overall structure
print("Top-level schema keys:", list(doc.schema.keys()) if hasattr(doc.schema, "keys") else type(doc.schema))
print("=" * 60)
for p in args.paths:
    try:
        node = schema_for_path(doc.schema, p)
        print(f"\n### {p}")
        print(json.dumps(node, indent=2, ensure_ascii=False)[:800])
    except Exception as e:
        print(f"\n### {p}  -> ERROR: {e}")
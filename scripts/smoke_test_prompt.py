#!/usr/bin/env python3
"""Smoke-test the enriched correction prompt WITHOUT any GPU/generation.

Builds build_correction_prompt() for a few real fields and prints the prompt so
you can verify:
  - the schema description block appears (and is right for `unit` vs absent for `scale`)
  - list-element siblings are populated (identity anchor), target field blanked
  - the position hint shows for list elements
  - document text is truncated, not crashing

Run from repo root:
  python scripts/smoke_test_prompt.py --config configs/exp_qwen35_4b_10kq.yaml \
      --paths balance_sheet.goodwill.0.unit balance_sheet.goodwill.0.scale
  python scripts/smoke_test_prompt.py --config configs/exp_qwen35_4b_pymupdf.yaml \
      --paths authors.2.name
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

from probe_extraction.config import load_config
load_benchmark = import_module("01_extract").load_benchmark
_sweep = import_module("09_regen_sweep")
build_correction_prompt = _sweep.build_correction_prompt
schema_for_path = import_module("07_regen_single").schema_for_path

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--paths", nargs="+", required=True)
ap.add_argument("--doc", default=None, help="doc_id substring to pick a specific doc")
ap.add_argument("--truncate-print", type=int, default=1200,
                help="chars of the prompt to print (document body is long)")
args = ap.parse_args()

cfg = load_config(args.config)
bench = load_benchmark(cfg)
docs = list(bench)
doc = docs[0]
if args.doc:
    doc = next((d for d in docs if args.doc in d.doc_id), docs[0])

# load this doc's parsed extraction for the sibling context
ext_path = cfg.artifacts_path / "extractions" / f"{doc.doc_id}.json"
parsed = {}
if ext_path.exists():
    ext = json.load(open(ext_path))
    parsed = ext.get("parsed_json") or {}
else:
    print(f"[warn] no extraction at {ext_path}; sibling context will be empty")

print(f"doc: {doc.doc_id}")
print(f"doc.text length: {len(doc.text)} chars (MAX_DOC_CHARS={_sweep.MAX_DOC_CHARS})")
print("=" * 72)

for p in args.paths:
    field_node = schema_for_path(doc.schema, p)
    prompt = build_correction_prompt(doc, parsed, p, field_node)
    # the document body is huge; show head (through the schema/instructions) and
    # confirm the doc got truncated by length, not by crash
    print(f"\n##### PATH: {p}")
    print(f"schema_node: {json.dumps(field_node, ensure_ascii=False)[:200]}")
    print("-" * 72)
    # print everything UP TO the document body, plus the instruction tail
    doc_marker = "# Document"
    i = prompt.find(doc_marker)
    head = prompt[:i] if i > 0 else prompt[:args.truncate_print]
    print(head)
    # show the instructions tail too (after the document)
    instr = prompt.find("# Instructions")
    if instr > 0:
        print("...[document body omitted]...")
        print(prompt[instr:instr+500])
    print("=" * 72)
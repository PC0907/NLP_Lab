#!/usr/bin/env python3
"""Gemma schema-key smoke test (the Llama-risk gate).

GOAL (narrow): does Gemma emit the ExtractBench schema's LITERAL key structure,
or does it paraphrase/re-nest keys (the failure that killed the Llama cross-model
run)? We are NOT measuring extraction accuracy -- only comparing the KEY TREE of
the model's output against the SCHEMA's expected keys.

GREEN -> Gemma is viable for cross-model generalization; proceed to full run.
RED   -> same schema-key wall as Llama; needs a normalized/value-based matcher
         before any cross-family model can be used.

Runs ONE document. Single forward pass. Needs GPU.

Usage:
  python smoke_test_gemma.py --config configs/exp_gemma3_12b_pooled.yaml --doc-index 0
"""
from __future__ import annotations
import argparse, json, sys

from probe_extraction.config import load_config, get_hf_token
from probe_extraction.extraction import Extractor
from probe_extraction.models import HuggingFaceLLM


def key_tree(obj, prefix=""):
    """Dotted key paths in nested dict/list; values ignored, list indices
    collapsed ([0].x and [1].x -> '[].x'). Structure, not content."""
    keys = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            keys.add(path)
            keys |= key_tree(v, path)
    elif isinstance(obj, list):
        for item in obj:
            keys |= key_tree(item, f"{prefix}[]")
    return keys


def schema_key_tree(schema, prefix=""):
    """Expected key paths implied by a JSON Schema (properties / items)."""
    keys = set()
    if not isinstance(schema, dict):
        return keys
    props = schema.get("properties", {})
    for k, sub in props.items():
        path = f"{prefix}.{k}" if prefix else k
        keys.add(path)
        t = sub.get("type")
        if t == "object":
            keys |= schema_key_tree(sub, path)
        elif t == "array":
            items = sub.get("items", {})
            if isinstance(items, dict) and items.get("type") == "object":
                keys |= schema_key_tree(items, f"{path}[]")
    return keys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="A Gemma config (model.name = the Gemma checkpoint).")
    ap.add_argument("--doc-index", type=int, default=0,
                    help="Which benchmark doc to test (0 = first).")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Load the benchmark the SAME way 01_extract.py does, then take one doc.
    # (Import here so the dispatcher logic stays in one place.)
    sys.path.insert(0, "scripts")
    from importlib import import_module
    ex = import_module("01_extract")
    bench = ex.load_benchmark(cfg)
    docs = list(bench)
    if not docs:
        print("No documents loaded -- check config domains/benchmark_path.")
        return 2
    doc = docs[min(args.doc_index, len(docs) - 1)]

    print("=" * 70)
    print(f"GEMMA SCHEMA-KEY SMOKE TEST")
    print(f"model : {cfg.model.name}")
    print(f"doc   : {doc.doc_id}  ({len(doc.text)} chars, domain={doc.domain})")
    print("=" * 70)

    hf_token = get_hf_token()
    llm = HuggingFaceLLM(
        model_name=cfg.model.name,
        dtype=cfg.model.dtype,
        quantization=cfg.model.quantization,
        device_map=cfg.model.device_map,
        trust_remote_code=cfg.model.trust_remote_code,
        hf_token=hf_token,
    )
    extractor = Extractor(
        llm=llm,
        layers=[min(8, llm.num_layers)],   # any valid layer; activations unused
        position="last_token",
        max_new_tokens=cfg.model.max_new_tokens,
        temperature=cfg.model.temperature,
        top_p=cfg.model.top_p,
        include_schema=cfg.extraction.include_schema,
        max_input_chars=cfg.extraction.max_input_chars,
    )

    result = extractor.extract(doc)

    print("\n--- generation ---")
    print("finish_reason:", result.finish_reason,
          "| generated_tokens:", result.generated_token_count)

    if result.parsed_json is None:
        print("\nPARSE FAILED:", result.parse_error)
        print("Raw (first 1500 chars):\n", result.raw_generated_text[:1500])
        print("\nVERDICT: RED -- Gemma did not produce parseable JSON on this schema.")
        return 2

    expected = schema_key_tree(doc.schema)
    got = key_tree(result.parsed_json)

    matched = sorted(expected & got)
    missing = sorted(expected - got)   # schema wanted, Gemma didn't emit (or renamed)
    extra   = sorted(got - expected)   # Gemma emitted, schema didn't ask for

    print("\n--- KEY STRUCTURE vs SCHEMA (structure only, values ignored) ---")
    print(f"schema keys : {len(expected)}")
    print(f"gemma keys  : {len(got)}")
    print(f"matched     : {len(matched)}")
    print(f"\n[MATCHED] emitted literally ({len(matched)}):")
    for k in matched: print("   ", k)
    print(f"\n[MISSING] in schema, NOT emitted ({len(missing)}):")
    for k in missing: print("   ", k)
    print(f"\n[EXTRA] emitted, NOT in schema ({len(extra)}):")
    for k in extra: print("   ", k)

    frac = len(matched) / max(len(expected), 1)
    print("\n" + "=" * 70)
    print(f"KEY OVERLAP: {frac:.0%} of schema keys emitted literally.")
    if frac >= 0.9 and not extra:
        print("VERDICT: GREEN -- keys align. Gemma is viable for cross-model work.")
    elif frac >= 0.9:
        print("VERDICT: AMBER -- keys mostly align but Gemma added keys. Inspect")
        print("[EXTRA]: extra NESTING is the Llama failure mode; stray scalars are benign.")
    else:
        print("VERDICT: RED -- keys diverge (the Llama wall). Cross-model needs a")
        print("normalized / value-based matcher before Gemma can be used.")
        print("Check [MISSING]/[EXTRA]: case/spacing renames are fixable by")
        print("normalization; extra nesting levels are the hard case.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""Smoke test for DeepSeek-R1-Distill-Qwen-7B on one ExtractBench document.

This model is REASONING-TUNED, which raises three questions the normal
schema-key check does not cover:

  1. Does it emit a chain-of-thought before the JSON? If so, how long?
  2. Does the JSON complete within the token budget, or does reasoning
     consume it first?
  3. Does the parser recover the JSON from a response that has a reasoning
     preamble in front of it?

Plus the usual gate:
  4. Does it reproduce the schema's literal key structure?

Runs ONE document. Needs GPU.

Usage:
  python smoke_test_r1.py --config configs/exp_r1qwen7b_pooled.yaml --doc-index 0
"""
from __future__ import annotations
import argparse, json, re, sys

from probe_extraction.config import load_config, get_hf_token
from probe_extraction.extraction import Extractor
from probe_extraction.models import HuggingFaceLLM


def key_tree(obj, prefix=""):
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
    keys = set()
    if not isinstance(schema, dict):
        return keys
    for k, sub in schema.get("properties", {}).items():
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
    ap.add_argument("--config", required=True)
    ap.add_argument("--doc-index", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)

    sys.path.insert(0, "scripts")
    from importlib import import_module
    ex = import_module("01_extract")
    docs = list(ex.load_benchmark(cfg))
    if not docs:
        print("No documents loaded -- check config domains / benchmark_path.")
        return 2
    doc = docs[min(args.doc_index, len(docs) - 1)]

    print("=" * 72)
    print("R1-DISTILL SMOKE TEST")
    print(f"model : {cfg.model.name}")
    print(f"doc   : {doc.doc_id}  ({len(doc.text)} chars, domain={doc.domain})")
    print("=" * 72)

    llm = HuggingFaceLLM(
        model_name=cfg.model.name,
        dtype=cfg.model.dtype,
        quantization=cfg.model.quantization,
        device_map=cfg.model.device_map,
        trust_remote_code=cfg.model.trust_remote_code,
        hf_token=get_hf_token(),
    )
    extractor = Extractor(
        llm=llm,
        layers=[min(8, llm.num_layers)],
        position="last_token",
        max_new_tokens=cfg.model.max_new_tokens,
        temperature=cfg.model.temperature,
        top_p=cfg.model.top_p,
        include_schema=cfg.extraction.include_schema,
        max_input_chars=cfg.extraction.max_input_chars,
    )

    result = extractor.extract(doc)
    raw = result.raw_generated_text or ""

    print("\n--- generation ---")
    print("finish_reason :", result.finish_reason)
    print("raw chars     :", len(raw))

    # --- Q1/Q2: reasoning preamble? ---
    print("\n--- reasoning-trace check ---")
    think_open = raw.find("<think>")
    think_close = raw.find("</think>")
    if think_open != -1:
        if think_close != -1:
            print(f"<think> block present: chars {think_open}..{think_close+8} "
                  f"({think_close - think_open} chars of reasoning)")
            print("Reasoning COMPLETED -- JSON should follow it.")
        else:
            print(f"<think> OPENED at char {think_open} but never closed.")
            print("Reasoning consumed the whole budget; no JSON produced.")
    else:
        print("No <think> markers found.")

    # where does the JSON actually start?
    brace = raw.find("{")
    fence = raw.find("```")
    print(f"first '{{' at char : {brace}")
    print(f"first '```' at char: {fence}")
    if brace > 500:
        print(f"NOTE: {brace} chars precede the JSON -- a substantial preamble.")

    print("\n--- first 600 chars of raw output ---")
    print(raw[:600])
    print("\n--- last 400 chars of raw output ---")
    print(raw[-400:])

    # --- Q3: did the parser recover JSON? ---
    print("\n--- parse ---")
    if result.parsed_json is None:
        print("PARSE FAILED:", result.parse_error)
        print("\nVERDICT: RED -- parser did not recover JSON from the response.")
        print("If a <think> block is present above, the parser likely needs to")
        print("strip the reasoning preamble before parsing.")
        return 2
    print("parsed OK")

    # --- Q4: schema keys ---
    expected = schema_key_tree(doc.schema)
    got = key_tree(result.parsed_json)
    matched = sorted(expected & got)
    missing = sorted(expected - got)
    extra = sorted(got - expected)

    print("\n--- KEY STRUCTURE vs SCHEMA ---")
    print(f"schema keys {len(expected)} | emitted {len(got)} | matched {len(matched)}")
    if missing:
        print(f"[MISSING] ({len(missing)}):")
        for k in missing:
            print("   ", k)
    if extra:
        print(f"[EXTRA] ({len(extra)}):")
        for k in extra:
            print("   ", k)

    frac = len(matched) / max(len(expected), 1)
    truncated = (result.finish_reason == "length")
    print("\n" + "=" * 72)
    print(f"KEY OVERLAP: {frac:.0%}")
    if truncated and missing and not extra:
        print("VERDICT: TRUNCATED / INCONCLUSIVE -- hit max_new_tokens.")
        print("Missing keys are likely truncation, not divergence (no extras).")
        print("Raise max_new_tokens or test a shorter document.")
    elif frac >= 0.9 and not extra:
        print("VERDICT: GREEN -- keys align; model is viable for a full run.")
    elif frac >= 0.9:
        print("VERDICT: AMBER -- keys align but extras emitted; inspect [EXTRA].")
    else:
        print("VERDICT: RED -- keys diverge; matcher work needed before use.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
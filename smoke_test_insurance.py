#!/usr/bin/env python3
"""Insurance-claims schema-key smoke test (Llama-risk check).

GOAL (narrow): does Qwen3.5 emit the CONSTRUCT insurance schema's LITERAL nested
key structure, or does it paraphrase/re-nest keys (the failure that killed the
Llama cross-model run)? We are NOT measuring extraction accuracy or training a
probe -- only comparing the KEY TREE of the model's output against gold's.

GREEN keys align  -> insurance viable; build the ExtractBench-format converter.
RED   keys diverge-> same schema-key problem as Llama; insurance -> future work.

Runs ONE document. Loads the model (needs a GPU), single forward pass.

Usage:
    python smoke_test_insurance.py --config configs/exp_qwen35_4b_pooled.yaml
"""
from __future__ import annotations

import argparse
import json
import sys

from datasets import load_dataset

# NOTE: matches how this repo is imported when run from ~/NLP_Lab with src/ on
# the path. If you get an import error, switch to `from probe_extraction...`
# (as scripts/01_extract.py does) and run with the editable install active.
from src.probe_extraction.config import load_config, get_hf_token
from src.probe_extraction.data import Document
from src.probe_extraction.extraction import Extractor
from src.probe_extraction.models import HuggingFaceLLM


# Insurance schema derived from CONSTRUCT's gold structure. This IS the key
# structure under test: we show the model these keys and check whether it echoes
# them literally.
INSURANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "header": {
            "type": "object",
            "properties": {
                "claim_id": {"type": "string"},
                "report_date": {"type": "string", "description": "YYYY-MM-DD"},
                "incident_date": {"type": "string", "description": "YYYY-MM-DD"},
                "reported_by": {"type": "string"},
                "channel": {"type": "string"},
            },
        },
        "policy_details": {
            "type": "object",
            "properties": {
                "policy_number": {"type": "string"},
                "policyholder_name": {"type": "string"},
                "coverage_type": {"type": "string"},
                "effective_date": {"type": "string"},
                "expiration_date": {"type": "string"},
            },
        },
        "insured_objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "object_id": {"type": "string"},
                    "object_type": {"type": "string"},
                    "make_model": {"type": "string"},
                    "year": {"type": "integer"},
                    "location_address": {"type": "string"},
                    "estimated_value": {"type": "number"},
                },
            },
        },
        "incident_description": {
            "type": "object",
            "properties": {
                "incident_type": {"type": "string"},
                "location_type": {"type": "string"},
                "estimated_damage_amount": {"type": "number"},
                "police_report_number": {"type": "string"},
            },
        },
    },
}


def key_tree(obj, prefix=""):
    """Set of dotted key paths in nested dict/list; values ignored, list indices
    collapsed ([0].x and [1].x both -> '[].x'). Structure, not content."""
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)

    # --- load one insurance claim + gold from CONSTRUCT (HF) ------------------
    ds = load_dataset("Cleanlab/insurance-claims-extraction")["train"]
    row = ds[args.index]
    claim_text = row["claim_text"]
    gold = row["ground_truth"]
    if isinstance(gold, str):
        try:
            gold = json.loads(gold)
        except json.JSONDecodeError:
            import ast
            gold = ast.literal_eval(gold)

    print("=" * 70)
    print(f"Insurance claim #{args.index}  (text={len(claim_text)} chars)")
    print("=" * 70)

    doc = Document(
        doc_id="insurance__smoke__0",
        domain="insurance/claims",
        text=claim_text,
        schema=INSURANCE_SCHEMA,
        gold=gold,
        source_path=None,  # extract() never reads this
    )

    # --- load the model EXACTLY as scripts/01_extract.py does -----------------
    hf_token = get_hf_token()  # NO argument
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
        layers=[min(8, llm.num_layers)],  # any valid layer; activations unused here
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
        print("Raw (first 1500 chars):")
        print(result.raw_generated_text[:1500])
        print("\n=> Cannot assess keys: model did not produce parseable JSON.")
        return 2

    gold_keys = key_tree(gold)
    pred_keys = key_tree(result.parsed_json)
    matched = sorted(gold_keys & pred_keys)
    missing = sorted(gold_keys - pred_keys)
    extra   = sorted(pred_keys - gold_keys)

    print("\n--- KEY STRUCTURE COMPARISON (structure only, values ignored) ---")
    print(f"gold keys : {len(gold_keys)}")
    print(f"pred keys : {len(pred_keys)}")
    print(f"matched   : {len(matched)}")
    print(f"\n[MATCHED] model emitted literally ({len(matched)}):")
    for k in matched: print("   ", k)
    print(f"\n[MISSING] in gold, NOT in model output ({len(missing)}):")
    for k in missing: print("   ", k)
    print(f"\n[EXTRA] in model output, NOT in gold ({len(extra)}):")
    for k in extra: print("   ", k)

    print("\n" + "=" * 70)
    frac = len(matched) / max(len(gold_keys), 1)
    print(f"KEY OVERLAP: {frac:.0%} of gold keys emitted literally.")
    if frac >= 0.9 and not extra:
        print("VERDICT: GREEN. Keys align -> insurance viable. Build the converter.")
    elif frac >= 0.9 and extra:
        print("VERDICT: AMBER. Keys mostly align but model added keys -> inspect "
              "[EXTRA]; may be extra nesting (the Llama failure mode).")
    else:
        print("VERDICT: RED. Keys diverge -> same schema-key problem as Llama. "
              "Insurance -> future work (needs normalized/value-based matcher).")
        print("Renamed keys (case/spacing) are fixable by normalization; extra "
              "NESTING levels are the hard case.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
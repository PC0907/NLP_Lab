#!/usr/bin/env python3
"""Minimal single-field selective regeneration test.

The atomic operation behind selective regeneration:
  1. Pick a field the probe flags as high error-probability that was in fact
     an error (label=1) -- the ideal regeneration candidate.
  2. Build a TARGETED prompt asking the model to re-extract ONLY that field.
  3. (dry-run) show prompt + original value + gold value, no GPU.
  4. (--generate) call the model at temperature>0, parse the single value,
     compare to gold via the AUTO matcher: fixed / still-wrong / unchanged.

Usage:
    # dry-run (no GPU): pick a field, show the targeted prompt
    python scripts/07_regen_single.py --config configs/exp_qwen35_4b_pooled.yaml

    # actually regenerate (needs GPU)
    python scripts/07_regen_single.py --config configs/exp_qwen35_4b_pooled.yaml --generate
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

from probe_extraction.config import load_config
from probe_extraction.utils.logging import setup_logging
from probe_extraction.labeling.value_compare import compare_values, ComparisonStrategy

logger = logging.getLogger(__name__)

BEST_LAYER = 18  # precommit / current best; revisit with layer-selection fix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-field regeneration test")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--generate", action="store_true",
                   help="Actually call the model (needs GPU). Default: dry-run.")
    p.add_argument("--layer", type=int, default=BEST_LAYER)
    p.add_argument("--temperature", type=float, default=0.7)
    return p.parse_args()


def schema_for_path(schema: dict, path_str: str) -> dict:
    """Walk a JSON Schema to the node for a data-path like
    'age_groups.2.results.0.time'. Numeric segments consume an 'items' level.
    Returns the field's schema node (with description/type) or {} if not found.
    """
    node = schema
    for seg in path_str.split("."):
        if seg.isdigit():
            # array index in the data -> 'items' in the schema
            node = node.get("items", {})
            continue
        # object property
        props = node.get("properties", {})
        if seg in props:
            node = props[seg]
        else:
            # maybe we're at an array whose items are objects
            node = node.get("items", {}).get("properties", {}).get(seg, {})
    # resolve a leading anyOf to the non-null variant if present
    if "anyOf" in node:
        variants = [v for v in node["anyOf"] if v.get("type") != "null"]
        if variants:
            node = variants[0]
    return node


TARGETED_SYSTEM = (
    "You are a careful data extraction assistant. Given a document and a "
    "description of ONE field, you extract just that field's value from the "
    "document. You are precise and conservative and do not invent information. "
    "If the field is not present, you return null. Your output is ONLY the "
    "JSON value for that one field, with no surrounding text or markdown."
)

TARGETED_USER = """Extract ONE specific field from the following document.

# Field to extract
Path: {path_str}
Description: {description}
Type: {type_str}

# Document

{document_text}

# Instructions
- Output ONLY the JSON value for this single field (e.g. "200m", 1947, null).
- Do not output the whole object, keys, markdown fences, or any other text.
- Use null if the field is not present in the document.
- Do not invent information.

# Output

"""


def build_targeted_prompt(field_node: dict, path_str: str, document_text: str) -> str:
    return TARGETED_USER.format(
        path_str=path_str,
        description=field_node.get("description", "(no description)"),
        type_str=field_node.get("type", "(unspecified)"),
        document_text=document_text,
    )


def pick_candidate(labels_dir: Path, activations_dir: Path, probe, layer: int):
    """Find one field: probe-flagged high-risk AND actually an error (label=1).

    Returns (doc_id, path_str, gold_value, extracted_value, probe_score) or None.
    """
    best = None  # (probe_score, doc_id, path_str, gold, extracted)
    for labels_path in sorted(labels_dir.glob("*.json")):
        if labels_path.name.startswith("_"):
            continue
        doc_id = labels_path.stem
        data = json.load(labels_path.open())
        act_path = activations_dir / f"{doc_id}.npz"
        if not act_path.exists():
            continue
        with np.load(act_path) as acts:
            for fld in data.get("labels", []):
                if int(fld["is_error"]) != 1:
                    continue  # we want a real error to try to fix
                if not fld.get("extracted_present", True):
                    continue  # skip omissions (nothing to re-extract from)
                key = f"{fld['path_str']}__layer{layer}"
                if key not in acts:
                    continue
                score = float(probe.score(acts[key][None, :].astype(np.float32))[0])
                cand = (score, doc_id, fld["path_str"],
                        fld.get("gold_value"), fld.get("extracted_value"))
                if best is None or score > best[0]:
                    best = cand
    return best


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_dir=cfg.logging.log_dir,
                  log_name="07_regen_single", log_to_file=cfg.logging.log_to_file)

    artifacts = cfg.artifacts_path
    labels_dir = artifacts / "labels"
    activations_dir = artifacts / "activations"
    probes_dir = artifacts / "probes"

    probe_path = probes_dir / f"probe_layer{args.layer}.pkl"
    if not probe_path.exists():
        logger.error("No probe at %s", probe_path)
        return 1
    probe = pickle.load(probe_path.open("rb"))

    cand = pick_candidate(labels_dir, activations_dir, probe, args.layer)
    if cand is None:
        logger.error("No suitable candidate field found.")
        return 1
    score, doc_id, path_str, gold, extracted = cand

    logger.info("=" * 70)
    logger.info("Candidate field (highest probe error-score among true errors):")
    logger.info("  doc_id     : %s", doc_id)
    logger.info("  path_str   : %s", path_str)
    logger.info("  probe score: %.4f  (P(error))", score)
    logger.info("  gold value : %r", gold)
    logger.info("  original   : %r  (labelled WRONG)", extracted)

    # Load the document text + schema for the targeted prompt.
    ext_path = artifacts / "extractions" / f"{doc_id}.json"
    ext = json.load(ext_path.open())
    document_text = ext.get("source_text") or ext.get("document_text") or ""
    schema = ext.get("schema") or {}
    if not document_text:
        logger.warning("No document text found in extraction file; "
                       "targeted prompt will be incomplete. Keys: %s",
                       list(ext.keys()))

    field_node = schema_for_path(schema, path_str) if schema else {}
    prompt = build_targeted_prompt(field_node, path_str, document_text)

    logger.info("-" * 70)
    logger.info("Field schema node: %s", json.dumps(field_node)[:300])
    logger.info("-" * 70)
    logger.info("TARGETED PROMPT (first 1500 chars):\n%s", prompt[:1500])
    logger.info("-" * 70)

    if not args.generate:
        logger.info("DRY RUN — not calling the model. Re-run with --generate to "
                    "actually re-extract this field.")
        return 0

    # --- Actual regeneration (GPU) ---
    from probe_extraction.models.hf_model import HFModel  # adjust import to your wrapper
    llm = HFModel(  # NOTE: confirm constructor args match your codebase
        model_name=cfg.model.name,
        dtype=cfg.model.dtype,
    )
    full_prompt = TARGETED_SYSTEM + "\n\n" + prompt
    out = llm.generate(
        prompt=full_prompt,
        max_new_tokens=256,
        temperature=args.temperature,
        top_p=cfg.model.top_p,
        return_logprobs=False,
    )
    raw = out.text if hasattr(out, "text") else str(out)
    logger.info("Regenerated raw output: %r", raw[:300])

    # Parse the single JSON value.
    new_value = None
    try:
        new_value = json.loads(raw.strip())
    except Exception:
        # fall back: strip fences / take first token-ish
        cleaned = raw.strip().strip("`").strip()
        try:
            new_value = json.loads(cleaned)
        except Exception:
            new_value = cleaned  # treat as raw string

    logger.info("Regenerated value: %r", new_value)

    # Compare to gold with the AUTO matcher.
    now_correct = compare_values(
        gold, new_value, strategy=ComparisonStrategy.AUTO, number_tolerance=0.0,
    )
    logger.info("=" * 70)
    logger.info("RESULT: original was WRONG; after regeneration -> %s",
                "FIXED (now matches gold)" if now_correct else "STILL WRONG")
    logger.info("  gold     : %r", gold)
    logger.info("  original : %r", extracted)
    logger.info("  new      : %r", new_value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
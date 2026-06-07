#!/usr/bin/env python3
"""Minimal single-field selective regeneration test.

Atomic operation behind selective regeneration:
  1. Pick a probe-flagged true error with a SHORT gold value (a fixable
     case -- a number/date/name, not a 600-char legal clause).
  2. Build a TARGETED prompt asking the model to re-extract ONLY that field.
  3. (dry-run) show prompt + original + gold, no GPU.
  4. (--generate) re-extract at temperature>0, compare to gold via AUTO matcher.

Usage:
    python scripts/07_regen_single.py --config configs/exp_qwen35_4b_pooled.yaml
    python scripts/07_regen_single.py --config configs/exp_qwen35_4b_credit.yaml --generate
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

from probe_extraction.config import load_config, get_hf_token
from probe_extraction.utils.logging import setup_logging
from probe_extraction.labeling.value_compare import compare_values, ComparisonStrategy

# reuse extraction's benchmark loader
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
_extract = import_module("01_extract")
load_benchmark = _extract.load_benchmark

logger = logging.getLogger(__name__)
BEST_LAYER = 18


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-field regeneration test")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--generate", action="store_true",
                   help="Actually re-extract via the model (needs GPU). Default: dry-run.")
    p.add_argument("--layer", type=int, default=BEST_LAYER)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--probe-path", type=str, default=None,
                   help="Explicit path to the probe .pkl (defaults to "
                        "<config artifacts>/probes/probe_layer<L>.pkl). "
                        "Use the POOLED probe here while running a per-domain config.")
    p.add_argument("--max-gold-len", type=int, default=60,
                   help="Only consider fields whose gold value string is at most "
                        "this many chars (focus on fixable, short fields).")
    return p.parse_args()


def schema_for_path(schema: dict, path_str: str) -> dict:
    """Walk a JSON Schema to the node for a data path like
    'age_groups.2.results.0.time'. Numeric segments consume an 'items' level."""
    node = schema
    for seg in path_str.split("."):
        if seg.isdigit():
            node = node.get("items", {})
            continue
        props = node.get("properties", {})
        if seg in props:
            node = props[seg]
        else:
            node = node.get("items", {}).get("properties", {}).get(seg, {})
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


def pick_candidate(labels_dir, activations_dir, probe, layer, max_gold_len):
    """Highest-probe-score true error with a SHORT gold value (fixable case)."""
    best = None
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
                    continue
                if not fld.get("extracted_present", True):
                    continue  # omissions: nothing extracted to compare/refine
                gold = fld.get("gold_value")
                # focus on SHORT fixable fields
                if gold is None or len(str(gold)) > max_gold_len:
                    continue
                key = f"{fld['path_str']}__layer{layer}"
                if key not in acts:
                    continue
                score = float(probe.score(acts[key][None, :].astype(np.float32))[0])
                cand = (score, doc_id, fld["path_str"], gold, fld.get("extracted_value"))
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

    if args.probe_path:
        probe_path = Path(args.probe_path)
    else:
        probe_path = probes_dir / f"probe_layer{args.layer}.pkl"
    if not probe_path.exists():
        logger.error("No probe at %s", probe_path)
        return 1
    probe = pickle.load(probe_path.open("rb"))

    cand = pick_candidate(labels_dir, activations_dir, probe, args.layer, args.max_gold_len)
    if cand is None:
        logger.error("No suitable SHORT-gold candidate found (try raising --max-gold-len).")
        return 1
    score, doc_id, path_str, gold, extracted = cand

    logger.info("=" * 70)
    logger.info("Candidate (highest probe error-score among SHORT true errors):")
    logger.info("  doc_id     : %s", doc_id)
    logger.info("  path_str   : %s", path_str)
    logger.info("  probe score: %.4f  (P(error))", score)
    logger.info("  gold value : %r", gold)
    logger.info("  original   : %r  (labelled WRONG)", extracted)

    # Load the benchmark and find this document (gives us text + schema).
    benchmark = load_benchmark(cfg)
    doc = next((d for d in benchmark if d.doc_id == doc_id), None)
    if doc is None:
        logger.error("Could not find doc %s in benchmark. Check config domains.", doc_id)
        return 1
    if doc.extraction_error is not None:
        logger.error("Doc %s has no text: %s", doc_id, doc.extraction_error)
        return 1

    field_node = schema_for_path(doc.schema, path_str)
    user_msg = TARGETED_USER.format(
        path_str=path_str,
        description=field_node.get("description", "(no description)"),
        type_str=field_node.get("type", "(unspecified)"),
        document_text=doc.text,
    )

    logger.info("-" * 70)
    logger.info("Field schema node: %s", json.dumps(field_node)[:300])
    logger.info("-" * 70)
    logger.info("TARGETED PROMPT user message (first 1200 chars):\n%s", user_msg[:1200])
    logger.info("  [document is %d chars total]", len(doc.text))
    logger.info("-" * 70)

    if not args.generate:
        logger.info("DRY RUN — not calling the model. Re-run with --generate.")
        return 0

    # --- Actual regeneration (GPU) ---
    from probe_extraction.models import HuggingFaceLLM
    llm = HuggingFaceLLM(
        model_name=cfg.model.name,
        dtype=cfg.model.dtype,
        quantization=cfg.model.quantization,
        device_map=cfg.model.device_map,
        trust_remote_code=cfg.model.trust_remote_code,
        hf_token=get_hf_token(),
    )
    prompt = llm.format_chat(TARGETED_SYSTEM, user_msg)
    out = llm.generate(
        prompt=prompt,
        max_new_tokens=256,
        temperature=args.temperature,
        top_p=cfg.model.top_p,
        return_logprobs=False,
    )
    raw = out.text or ""
    logger.info("Regenerated raw: %r", raw[:300])

    # Parse the single JSON value (tolerant).
    new_value = None
    for attempt in (raw.strip(), raw.strip().strip("`").strip()):
        try:
            new_value = json.loads(attempt)
            break
        except Exception:
            continue
    if new_value is None:
        new_value = raw.strip()  # treat as raw string

    now_correct = compare_values(
        gold, new_value, strategy=ComparisonStrategy.AUTO, number_tolerance=0.0,
    )
    logger.info("=" * 70)
    logger.info("RESULT: original WRONG -> after regen: %s",
                "FIXED" if now_correct else "STILL WRONG")
    logger.info("  gold     : %r", gold)
    logger.info("  original : %r", extracted)
    logger.info("  new      : %r", new_value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
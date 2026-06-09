#!/usr/bin/env python3
"""Selective regeneration: correction-with-context + cost-quality sweep.

Pipeline:
  1. Load the fixable error set (from 08_fixability_filter.py output) -- only
     errors whose gold value is present in the document text are regeneration
     candidates. Optionally also include CORRECT fields so we can measure
     collateral damage (regenerating a correct field can break it).
  2. For each candidate field, build a CORRECTION-WITH-CONTEXT prompt: show the
     model the surrounding object (siblings) but NOT its own (possibly wrong)
     value, and ask it to re-derive just the target field. (Blind single-field
     re-extraction fails on nested fields -- it returns null for lack of
     locating context. Siblings provide that context; hiding the original
     avoids anchoring the model to its mistake.)
  3. Regenerate (GPU) and CACHE the new values to disk. This is the only GPU
     step; everything after is CPU and re-runnable.
  4. Sweep a probe-score threshold tau. At each tau, "regenerate" every field
     with probe_score >= tau (using cached results) and measure:
        cost    = number of regenerations
        quality = net fields corrected (errors fixed minus correct broken)
     Produce the cost-quality curve for probe-guided vs logprob-guided vs
     random vs oracle ordering.

Two phases, controlled by flags:
  --regenerate : do the GPU regeneration pass, cache results. (sbatch / GPU)
  (default)    : load cached results, run the sweep + curve. (CPU, login node)

Usage:
  # phase 1 (GPU, sbatch): regenerate candidate fields for a domain
  python scripts/09_regen_sweep.py --config configs/exp_qwen35_4b_credit.yaml \
      --probe-path artifacts/qwen35_4b_pooled/probes/probe_layer18.pkl \
      --regenerate

  # phase 2 (CPU): build the cost-quality curve from cached results
  python scripts/09_regen_sweep.py --config configs/exp_qwen35_4b_credit.yaml \
      --probe-path artifacts/qwen35_4b_pooled/probes/probe_layer18.pkl
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

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
load_benchmark = import_module("01_extract").load_benchmark
# reuse the schema walker from the single-field script
_regen_single = import_module("07_regen_single")
schema_for_path = _regen_single.schema_for_path

logger = logging.getLogger(__name__)
BEST_LAYER = 18


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Selective regeneration sweep")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--probe-path", type=str, required=True,
                   help="Path to the trained probe .pkl (use the POOLED probe).")
    p.add_argument("--layer", type=int, default=BEST_LAYER)
    p.add_argument("--regenerate", action="store_true",
                   help="GPU phase: re-extract candidate fields and cache. "
                        "Default (no flag): CPU phase, build the curve from cache.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--include-correct", action="store_true", default=True,
                   help="Also regenerate a sample of CORRECT fields to measure "
                        "collateral damage (default on).")
    p.add_argument("--cache", type=str, default=None,
                   help="Path to the regeneration cache JSON (default: "
                        "<artifacts>/results/regen_cache.json).")
    return p.parse_args()


CORRECTION_SYSTEM = (
    "You are a careful data extraction assistant. You are given a document and "
    "a partially-extracted record with ONE field left blank. Fill in ONLY the "
    "blank field, using the document. Be precise and conservative; do not "
    "invent information. If the field is genuinely not in the document, return "
    "null. Output ONLY the JSON value for the blank field."
)

CORRECTION_USER = """A record was extracted from the document below, but one field needs to be re-derived.

# The record so far (one field blanked out as <FILL_THIS>)
{context_obj}

# Field to fill
Key: {field_key}
Description: {description}
Type: {type_str}

# Document

{document_text}

# Instructions
- Determine ONLY the value of "{field_key}" from the document.
- Use the surrounding record fields above to locate the right place in the document.
- Output ONLY the JSON value (e.g. "200m", 1947, null). No keys, no markdown, no other text.

# Output

"""


def _parent_and_key(parsed_json, path_str):
    """Navigate parsed_json to the PARENT object of path_str, return
    (parent_obj, leaf_key). Handles dotted paths with numeric indices."""
    segs = path_str.split(".")
    node = parsed_json
    for seg in segs[:-1]:
        if isinstance(node, list):
            node = node[int(seg)]
        elif isinstance(node, dict):
            node = node.get(seg, {})
        else:
            return None, segs[-1]
    return node, segs[-1]


def build_correction_prompt(doc, parsed_json, path_str, field_node):
    """Build the correction-with-context prompt for one field."""
    parent, key = _parent_and_key(parsed_json, path_str)
    # Build a context object: the parent dict with the target field blanked.
    if isinstance(parent, dict):
        ctx = dict(parent)
        ctx[key] = "<FILL_THIS>"
    else:
        ctx = {key: "<FILL_THIS>"}
    context_obj = json.dumps(ctx, ensure_ascii=False, indent=2)[:1500]
    return CORRECTION_USER.format(
        context_obj=context_obj,
        field_key=key,
        description=field_node.get("description", "(no description)"),
        type_str=field_node.get("type", "(unspecified)"),
        document_text=doc.text,
    )


def load_candidates(cfg, probe, layer, include_correct):
    """Build the candidate list: fixable errors (+ optional correct sample),
    each with probe score, logprob, gold, original value, doc_id, path."""
    artifacts = cfg.artifacts_path
    labels_dir = artifacts / "labels"
    activations_dir = artifacts / "activations"
    extractions_dir = artifacts / "extractions"

    # fixable set from the filter
    fix_path = artifacts / "results" / "fixability.json"
    fixable = set()
    if fix_path.exists():
        fixdata = json.load(fix_path.open())
        for f in fixdata["per_field"]:
            if f["gold_in_text"]:
                fixable.add((f["doc_id"], f["path_str"]))
    else:
        logger.warning("No fixability.json; treating ALL errors as candidates.")

    cands = []
    for labels_path in sorted(labels_dir.glob("*.json")):
        if labels_path.name.startswith("_"):
            continue
        doc_id = labels_path.stem
        data = json.load(labels_path.open())
        act_path = activations_dir / f"{doc_id}.npz"
        if not act_path.exists():
            continue
        ext_path = extractions_dir / f"{doc_id}.json"
        ext = json.load(ext_path.open()) if ext_path.exists() else {}
        token_lps = ext.get("token_logprobs")
        spans = {f["path_str"]: f.get("token_span") for f in ext.get("fields", [])}

        with np.load(act_path) as acts:
            for fld in data.get("labels", []):
                ps = fld["path_str"]
                is_err = int(fld.get("is_error", 0))
                if not fld.get("extracted_present", True):
                    continue
                # candidates: fixable errors, plus (optionally) correct fields
                if is_err == 1:
                    if fixable and (doc_id, ps) not in fixable:
                        continue  # error but not fixable -> skip
                elif not include_correct:
                    continue
                key = f"{ps}__layer{layer}"
                if key not in acts:
                    continue
                score = float(probe.score(acts[key][None, :].astype(np.float32))[0])
                # logprob baseline score (min logprob over the field span)
                min_lp = float("nan")
                span = spans.get(ps)
                if token_lps and span and len(span) == 2:
                    a, b = max(0, span[0]), min(span[1], len(token_lps))
                    if b > a:
                        min_lp = float(np.min(token_lps[a:b]))
                cands.append({
                    "doc_id": doc_id, "path_str": ps,
                    "is_error": is_err,
                    "gold_value": fld.get("gold_value"),
                    "extracted_value": fld.get("extracted_value"),
                    "probe_score": score,
                    "min_logprob": min_lp,
                })
    return cands


# ---------------------------------------------------------------------------
# Phase 1: regenerate (GPU)
# ---------------------------------------------------------------------------

def phase_regenerate(cfg, cands, temperature, cache_path):
    from probe_extraction.models import HuggingFaceLLM
    llm = HuggingFaceLLM(
        model_name=cfg.model.name, dtype=cfg.model.dtype,
        quantization=cfg.model.quantization, device_map=cfg.model.device_map,
        trust_remote_code=cfg.model.trust_remote_code, hf_token=get_hf_token(),
    )
    # load benchmark for document text + schema
    benchmark = load_benchmark(cfg)
    docs = {d.doc_id: d for d in benchmark}
    # load extractions for parsed_json (sibling context)
    ext_dir = cfg.artifacts_path / "extractions"

    cache = {}
    if cache_path.exists():
        cache = json.load(cache_path.open())

    for i, c in enumerate(cands):
        key = f"{c['doc_id']}::{c['path_str']}"
        if key in cache:
            continue  # resume
        doc = docs.get(c["doc_id"])
        if doc is None or getattr(doc, "extraction_error", None) is not None:
            continue
        ext = json.load((ext_dir / f"{c['doc_id']}.json").open())
        parsed = ext.get("parsed_json") or {}
        field_node = schema_for_path(doc.schema, c["path_str"])
        prompt_user = build_correction_prompt(doc, parsed, c["path_str"], field_node)
        prompt = llm.format_chat(CORRECTION_SYSTEM, prompt_user)
        out = llm.generate(prompt=prompt, max_new_tokens=128,
                           temperature=temperature, top_p=cfg.model.top_p,
                           return_logprobs=False)
        raw = (out.text or "").strip()
        new_val = None
        for attempt in (raw, raw.strip("`").strip()):
            try:
                new_val = json.loads(attempt)
                break
            except Exception:
                continue
        if new_val is None:
            new_val = raw
        cache[key] = {"new_value": new_val, "raw": raw[:200]}
        if (i + 1) % 10 == 0:
            logger.info("Regenerated %d/%d; checkpointing cache.", i + 1, len(cands))
            cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    logger.info("Regeneration cache written: %s (%d entries)", cache_path, len(cache))


# ---------------------------------------------------------------------------
# Phase 2: sweep + curve (CPU)
# ---------------------------------------------------------------------------

def _outcome(gold, new_value):
    """Is the regenerated value correct vs gold (AUTO matcher)?"""
    return compare_values(gold, new_value, strategy=ComparisonStrategy.AUTO,
                          number_tolerance=0.0)


def phase_sweep(cfg, cands, cache_path):
    if not cache_path.exists():
        logger.error("No regeneration cache at %s. Run --regenerate first.", cache_path)
        return 1
    cache = json.load(cache_path.open())

    # annotate each candidate with its regeneration outcome
    enriched = []
    for c in cands:
        key = f"{c['doc_id']}::{c['path_str']}"
        if key not in cache:
            continue
        new_val = cache[key]["new_value"]
        was_error = c["is_error"] == 1
        now_correct = _outcome(c["gold_value"], new_val)
        # net effect IF we regenerate this field:
        #   error -> correct : +1 (fixed)
        #   error -> still wrong : 0
        #   correct -> wrong : -1 (broken)
        #   correct -> still correct : 0
        if was_error and now_correct:
            delta = +1
        elif (not was_error) and (not now_correct):
            delta = -1
        else:
            delta = 0
        c = dict(c, regen_new=new_val, regen_delta=delta)
        enriched.append(c)

    n = len(enriched)
    n_err = sum(c["is_error"] for c in enriched)
    logger.info("Sweep over %d regenerated candidates (%d errors, %d correct).",
                n, n_err, n - n_err)

    # For each ordering strategy, sweep the number of fields regenerated (cost)
    # from 0..n, picking fields in that strategy's priority order, and track
    # cumulative net corrections (quality).
    def curve(order_key, reverse=True):
        ordered = sorted(enriched, key=lambda c: c[order_key], reverse=reverse)
        cost, net = [0], [0]
        running = 0
        for k, c in enumerate(ordered, 1):
            running += c["regen_delta"]
            cost.append(k)
            net.append(running)
        return cost, net

    import random
    rng = random.Random(42)

    curves = {}
    # probe-guided: regenerate highest probe-score first
    curves["probe"] = curve("probe_score", reverse=True)
    # logprob-guided: regenerate lowest min_logprob first (most uncertain)
    #   lower logprob = more likely error, so ascending
    curves["logprob"] = curve("min_logprob", reverse=False)
    # random ordering
    shuffled = enriched[:]
    rng.shuffle(shuffled)
    cost_r, net_r, run = [0], [0], 0
    for k, c in enumerate(shuffled, 1):
        run += c["regen_delta"]; cost_r.append(k); net_r.append(run)
    curves["random"] = (cost_r, net_r)
    # oracle: regenerate actual errors first (using gold knowledge)
    curves["oracle"] = curve("is_error", reverse=True)

    # report: net corrections at a few cost budgets
    logger.info("=" * 70)
    logger.info("COST-QUALITY (net fields corrected vs # regenerations):")
    budgets = [int(n * f) for f in (0.1, 0.25, 0.5, 1.0)]
    header = "  budget   " + "  ".join(f"{name:>8s}" for name in curves)
    logger.info(header)
    for b in budgets:
        row = f"  {b:5d}    "
        for name, (cost, net) in curves.items():
            # net corrections at cost <= b
            idx = min(b, len(net) - 1)
            row += f"  {net[idx]:8d}"
        logger.info(row)
    logger.info("=" * 70)
    logger.info("Max net corrections possible (oracle, all errors fixed-able): "
                "fixed=%d, broken-if-all=%d",
                sum(1 for c in enriched if c["is_error"] and c["regen_delta"] == +1),
                sum(1 for c in enriched if not c["is_error"] and c["regen_delta"] == -1))

    # save full curves
    out = cfg.artifacts_path / "results" / "regen_curves.json"
    out.write_text(json.dumps({k: {"cost": v[0], "net": v[1]} for k, v in curves.items()},
                              indent=2))
    logger.info("Saved curves to %s", out)
    return 0


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_dir=cfg.logging.log_dir,
                  log_name="09_regen_sweep", log_to_file=cfg.logging.log_to_file)

    probe = pickle.load(open(args.probe_path, "rb"))
    cands = load_candidates(cfg, probe, args.layer, args.include_correct)
    logger.info("Loaded %d regeneration candidates (errors + correct sample).", len(cands))

    cache_path = Path(args.cache) if args.cache else (cfg.artifacts_path / "results" / "regen_cache.json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if args.regenerate:
        phase_regenerate(cfg, cands, args.temperature, cache_path)
        return 0
    else:
        return phase_sweep(cfg, cands, cache_path)


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""Multi-sample regeneration with PROBE-BASED selection + safe-override.

The novel step beyond 09_regen_sweep.py: instead of regenerating a flagged field
ONCE and blindly overwriting, we:
  1. Regenerate the field N times (temperature > 0 for diversity).
  2. Score EACH candidate (and the original) with the probe -- by running the
     candidate value back through the model in its record context and reading the
     layer-L activation, then probe -> P(error). (PromptPort ablation: the judge
     needs both context and the candidate value; we get this by re-embedding the
     candidate in the document+record context.)
  3. SAFE-OVERRIDE: keep the original unless the best candidate's P(error) is
     LOWER than the original's by at least a margin delta. Among candidates that
     clear the margin, take the lowest P(error). Else keep original (do no harm).

This directly attacks the "regeneration breaks correct fields" problem: a correct
field whose original P(error) is already low will not be overridden, because no
candidate can beat it by the margin. The probe is the selector (your thesis),
replacing PromptPort's DistilBERT verifier and self-reported confidence (which the
literature shows is overconfident).

Caches everything so the downstream selection/threshold sweep is CPU-only.

Two phases:
  --regenerate : GPU. For each flagged candidate, sample N regenerations, embed +
                 probe-score each, cache {candidates:[{value,p_err}], orig_p_err}.
  (default)    : CPU. Apply safe-override at a sweep of margins, report net /
                 override precision (like safe_override_rescore.py but using the
                 per-candidate probe scores, not just the original's).

Usage (phase 1, GPU/sbatch):
  python scripts/10_regen_select.py --config configs/exp_qwen35_4b_pooled_regen.yaml \
      --probe-path artifacts/qwen35_4b_pooled/probes/probe_layer18.pkl \
      --regenerate --n-samples 5 --temperature 0.7 --flag-threshold 0.5

Usage (phase 2, CPU):
  python scripts/10_regen_select.py --config configs/exp_qwen35_4b_pooled_regen.yaml \
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
_sweep = import_module("09_regen_sweep")
build_correction_prompt = _sweep.build_correction_prompt
load_candidates = _sweep.load_candidates
CORRECTION_SYSTEM = _sweep.CORRECTION_SYSTEM
_regen_single = import_module("07_regen_single")
schema_for_path = _regen_single.schema_for_path

logger = logging.getLogger(__name__)
BEST_LAYER = 18


def parse_args():
    p = argparse.ArgumentParser(description="Multi-sample regeneration + probe selection")
    p.add_argument("--config", required=True)
    p.add_argument("--probe-path", required=True)
    p.add_argument("--layer", type=int, default=BEST_LAYER)
    p.add_argument("--regenerate", action="store_true")
    p.add_argument("--n-samples", type=int, default=5,
                   help="Number of regenerations per flagged field.")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature for diversity (must be > 0).")
    p.add_argument("--flag-threshold", type=float, default=0.5,
                   help="Only regenerate fields whose ORIGINAL probe P(error) >= this.")
    p.add_argument("--cache", default=None)
    return p.parse_args()


def _extract_activation_for_value(llm, probe_layer, doc, parsed_json, path_str,
                                  field_node, candidate_value):
    """Embed candidate_value into the record context, run the model, and return
    the layer-`probe_layer` activation at the field's last token. This mirrors how
    the original extraction activations were captured, so the probe sees a
    comparable representation.

    NOTE: depends on the model wrapper exposing a way to capture hidden states for
    a given (prompt, value) -- we reuse the extraction path. If the wrapper lacks a
    direct hook, this falls back to scoring the generation prompt's last token.
    """
    # Build the correction prompt (document + record context), then append the
    # candidate value as the "answer", and capture the hidden state at its span.
    user = build_correction_prompt(doc, parsed_json, path_str, field_node)
    prompt = llm.format_chat(CORRECTION_SYSTEM, user)
    # Append candidate so the activation is computed OVER the candidate tokens.
    full = prompt + json.dumps(candidate_value, ensure_ascii=False)
    # llm.capture_hidden_states should return {layer: (seq_len, hidden)} for `full`.
    hs = llm.capture_hidden_states(full, layers=[probe_layer])
    vec = hs[probe_layer]
    if vec.ndim > 1:
        vec = vec[-1]          # last token = the candidate's final token
    return vec.astype(np.float32)


def phase_regenerate(cfg, args, cands, probe, cache_path):
    from probe_extraction.models import HuggingFaceLLM
    llm = HuggingFaceLLM(
        model_name=cfg.model.name, dtype=cfg.model.dtype,
        quantization=cfg.model.quantization, device_map=cfg.model.device_map,
        trust_remote_code=cfg.model.trust_remote_code, hf_token=get_hf_token(),
    )
    benchmark = load_benchmark(cfg)
    docs = {d.doc_id: d for d in benchmark}
    ext_dir = cfg.artifacts_path / "extractions"

    cache = {}
    if cache_path.exists():
        cache = json.load(cache_path.open())

    # Only regenerate fields the probe FLAGS (original P(error) >= flag-threshold).
    flagged = [c for c in cands if c["probe_score"] >= args.flag_threshold]
    logger.info("Flagged %d / %d candidates for multi-sample regeneration (tau_flag=%.2f).",
                len(flagged), len(cands), args.flag_threshold)

    for i, c in enumerate(flagged):
        key = f"{c['doc_id']}::{c['path_str']}"
        if key in cache:
            continue
        doc = docs.get(c["doc_id"])
        if doc is None or getattr(doc, "extraction_error", None) is not None:
            continue
        ext = json.load((ext_dir / f"{c['doc_id']}.json").open())
        parsed = ext.get("parsed_json") or {}
        field_node = schema_for_path(doc.schema, c["path_str"])
        user = build_correction_prompt(doc, parsed, c["path_str"], field_node)
        prompt = llm.format_chat(CORRECTION_SYSTEM, user)

        candidates = []
        for s in range(args.n_samples):
            out = llm.generate(prompt=prompt, max_new_tokens=128,
                               temperature=args.temperature, top_p=cfg.model.top_p,
                               return_logprobs=False)
            raw = (out.text or "").strip()
            val = None
            for attempt in (raw, raw.strip("`").strip()):
                try:
                    val = json.loads(attempt); break
                except Exception:
                    continue
            if val is None:
                val = raw
            # probe-score this candidate
            try:
                vec = _extract_activation_for_value(
                    llm, args.layer, doc, parsed, c["path_str"], field_node, val)
                p_err = float(probe.score(vec[None, :])[0])
            except Exception as e:
                logger.warning("probe-score failed for %s sample %d: %s", key, s, e)
                p_err = float("nan")
            candidates.append({"value": val, "p_err": p_err})

        cache[key] = {
            "orig_p_err": float(c["probe_score"]),
            "orig_value": c.get("extracted_value"),
            "gold_value": c.get("gold_value"),
            "is_error": int(c["is_error"]),
            "candidates": candidates,
        }
        if (i + 1) % 5 == 0:
            logger.info("Regenerated+scored %d/%d flagged; checkpoint.", i + 1, len(flagged))
            cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    logger.info("Wrote multi-sample cache: %s (%d entries)", cache_path, len(cache))


def _ok(gold, val, mode):
    _rs = import_module("rescore_regen")
    return (_rs.lenient_ok if mode == "lenient" else _rs.strict_ok)(gold, val)


def phase_select(cfg, args, cache_path):
    if not cache_path.exists():
        logger.error("No multi-sample cache at %s. Run --regenerate first.", cache_path)
        return 1
    cache = json.load(cache_path.open())

    for mode in ("strict", "lenient"):
        logger.info("=" * 72)
        logger.info("MODE: %s — safe-override with probe-selected best candidate", mode.upper())
        logger.info("  %-7s %8s %8s %7s %7s %7s   %s", "margin", "n_over", "ovr_rate",
                    "fixed", "broke", "net", "precision")
        for margin in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]:
            n_over = fixed = broke = 0
            total = len(cache)
            for key, e in cache.items():
                gold = e["gold_value"]
                was_err = e["is_error"] == 1
                orig_p = e["orig_p_err"]
                # best candidate by lowest P(error)
                cands = [c for c in e["candidates"] if c["p_err"] == c["p_err"]]  # drop nan
                if not cands:
                    continue
                best = min(cands, key=lambda c: c["p_err"])
                # safe-override: override only if best beats original by >= margin
                if best["p_err"] <= orig_p - margin:
                    n_over += 1
                    new_ok = _ok(gold, best["value"], mode)
                    if was_err and new_ok:
                        fixed += 1
                    elif (not was_err) and (not new_ok):
                        broke += 1
                # else keep original (no change)
            net = fixed - broke
            changed = fixed + broke
            prec = (fixed / changed) if changed else float("nan")
            prec_s = f"{prec:.2f}" if prec == prec else " n/a"
            logger.info("  %-7.2f %8d %8.3f %7d %7d %7d   %s",
                        margin, n_over, n_over / total if total else 0.0,
                        fixed, broke, net, prec_s)
    return 0


def main():
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_dir=cfg.logging.log_dir,
                  log_name="10_regen_select", log_to_file=cfg.logging.log_to_file)
    probe = pickle.load(open(args.probe_path, "rb"))
    cache_path = Path(args.cache) if args.cache else (
        cfg.artifacts_path / "results" / "regen_select_cache.json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if args.regenerate:
        cands = load_candidates(cfg, probe, args.layer, include_correct=True)
        logger.info("Loaded %d candidates.", len(cands))
        phase_regenerate(cfg, args, cands, probe, cache_path)
        return 0
    else:
        return phase_select(cfg, args, cache_path)


if __name__ == "__main__":
    sys.exit(main())
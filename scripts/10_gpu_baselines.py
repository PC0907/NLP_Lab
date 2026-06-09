#!/usr/bin/env python3
"""GPU black-box baselines: P(True) and self-consistency.

Both are SCALAR baselines (one score per field), evaluated full-set via
evaluate_baseline -- comparable to token_logprob. They need GPU generation
passes, so they run in two phases like the regen sweep:
  --generate : GPU pass, cache the raw signals.
  (default)  : CPU, compute scores + AUROC vs labels.

P(True) (Kadavath et al. 2022, arXiv:2207.05221):
  Ask the model whether its own extracted value is correct; use P(True).
  Low P(True) -> likely error.

Self-consistency (Wang et al. 2022, arXiv:2203.11171; Manakul et al. 2023,
SelfCheckGPT, arXiv:2303.08896):
  Resample the document extraction N times at temperature>0; for each field,
  measure agreement of the resampled values with the original. Low agreement
  -> likely error.

Usage:
  # GPU phase (sbatch)
  python scripts/10_gpu_baselines.py --config configs/exp_qwen35_4b_credit.yaml --generate
  # CPU phase
  python scripts/10_gpu_baselines.py --config configs/exp_qwen35_4b_credit.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from probe_extraction.config import load_config, get_hf_token
from probe_extraction.utils.logging import setup_logging
from probe_extraction.labeling.value_compare import compare_values, ComparisonStrategy
from probe_extraction.baselines import evaluate_baseline

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
load_benchmark = import_module("01_extract").load_benchmark
_regen = import_module("07_regen_single")
schema_for_path = _regen.schema_for_path

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="P(True) + self-consistency baselines")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--generate", action="store_true",
                   help="GPU phase: produce + cache signals. Default: CPU scoring.")
    p.add_argument("--n-samples", type=int, default=5,
                   help="Self-consistency resamples per document.")
    p.add_argument("--sc-temperature", type=float, default=0.7)
    p.add_argument("--cache", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# field iteration helper
# ---------------------------------------------------------------------------

def iter_fields(cfg):
    """Yield (doc_id, field_dict) for all labelled, extracted-present fields."""
    labels_dir = cfg.artifacts_path / "labels"
    for lp in sorted(labels_dir.glob("*.json")):
        if lp.name.startswith("_"):
            continue
        doc_id = lp.stem
        data = json.load(lp.open())
        for fld in data.get("labels", []):
            if not fld.get("extracted_present", True):
                continue
            yield doc_id, fld


def _nav(parsed, path_str):
    """Navigate parsed_json to path_str's value (None if missing)."""
    node = parsed
    for seg in path_str.split("."):
        if isinstance(node, list):
            try:
                node = node[int(seg)]
            except (ValueError, IndexError):
                return None
        elif isinstance(node, dict):
            node = node.get(seg)
        else:
            return None
        if node is None:
            return None
    return node


# ---------------------------------------------------------------------------
# P(True) prompt
# ---------------------------------------------------------------------------

PTRUE_SYSTEM = (
    "You are a careful verifier. You are shown a document, a field, and a "
    "proposed value extracted from the document. Answer with exactly one word: "
    "True if the proposed value is correct per the document, or False if it is "
    "wrong. Answer only True or False."
)
PTRUE_USER = """# Field
{path_str}: {description}

# Proposed value
{value}

# Document
{document_text}

# Is the proposed value correct? Answer True or False.
Answer:"""


# ---------------------------------------------------------------------------
# Phase 1: GPU generation
# ---------------------------------------------------------------------------

def phase_generate(cfg, n_samples, sc_temperature, cache_path):
    from probe_extraction.models import HuggingFaceLLM
    llm = HuggingFaceLLM(
        model_name=cfg.model.name, dtype=cfg.model.dtype,
        quantization=cfg.model.quantization, device_map=cfg.model.device_map,
        trust_remote_code=cfg.model.trust_remote_code, hf_token=get_hf_token(),
    )
    benchmark = load_benchmark(cfg)
    docs = {d.doc_id: d for d in benchmark}
    ext_dir = cfg.artifacts_path / "extractions"

    cache = json.load(cache_path.open()) if cache_path.exists() else {"ptrue": {}, "selfcons": {}}

    # ---- P(True): per field ----
    from probe_extraction.extraction.prompts import build_extraction_prompt
    for doc_id, fld in iter_fields(cfg):
        key = f"{doc_id}::{fld['path_str']}"
        if key in cache["ptrue"]:
            continue
        doc = docs.get(doc_id)
        if doc is None or getattr(doc, "extraction_error", None) is not None:
            continue
        node = schema_for_path(doc.schema, fld["path_str"])
        user = PTRUE_USER.format(
            path_str=fld["path_str"],
            description=node.get("description", ""),
            value=fld.get("extracted_value"),
            document_text=doc.text,
        )
        prompt = llm.format_chat(PTRUE_SYSTEM, user)
        out = llm.generate(prompt=prompt, max_new_tokens=4,
                           temperature=0.0, top_p=1.0, return_logprobs=True)
        raw = (out.text or "").strip()
        # crude P(True): 1.0 if says True, 0.0 if False; refine with logprobs if available
        said_true = raw.lower().startswith("true")
        cache["ptrue"][key] = {"raw": raw, "p_true": 1.0 if said_true else 0.0}

    cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    logger.info("P(True) cached: %d fields", len(cache["ptrue"]))

    # ---- Self-consistency: resample each document N times ----
    for doc_id, doc in docs.items():
        if getattr(doc, "extraction_error", None) is not None:
            continue
        if doc_id in cache["selfcons"]:
            continue
        sys_msg, user_msg = build_extraction_prompt(
            schema=doc.schema, document_text=doc.text, include_schema=cfg.extraction.include_schema)
        prompt = llm.format_chat(sys_msg, user_msg)
        samples = []
        for _ in range(n_samples):
            out = llm.generate(prompt=prompt, max_new_tokens=cfg.model.max_new_tokens,
                               temperature=sc_temperature, top_p=cfg.model.top_p,
                               return_logprobs=False)
            try:
                samples.append(json.loads((out.text or "").strip()))
            except Exception:
                samples.append(None)
        cache["selfcons"][doc_id] = samples
        cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False, default=str))
        logger.info("Self-consistency cached doc %s (%d samples)", doc_id, n_samples)

    logger.info("GPU phase done. Cache: %s", cache_path)


# ---------------------------------------------------------------------------
# Phase 2: CPU scoring
# ---------------------------------------------------------------------------

def phase_score(cfg, cache_path):
    if not cache_path.exists():
        logger.error("No cache at %s; run --generate first.", cache_path); return 1
    cache = json.load(cache_path.open())
    ext_dir = cfg.artifacts_path / "extractions"

    ptrue_scores, sc_scores, y = [], [], []
    for doc_id, fld in iter_fields(cfg):
        key = f"{doc_id}::{fld['path_str']}"
        label = int(fld.get("is_error", 0))

        # P(True): higher P(True) = less likely error -> score_higher_is_error=False
        pt = cache["ptrue"].get(key, {}).get("p_true", float("nan"))

        # Self-consistency: agreement of resamples with the original extracted value.
        # low agreement -> more likely error.
        samples = cache["selfcons"].get(doc_id, [])
        orig = fld.get("extracted_value")
        agree = 0; total = 0
        for s in samples:
            if not isinstance(s, dict):
                continue
            total += 1
            v = _nav(s, fld["path_str"])
            if compare_values(orig, v, strategy=ComparisonStrategy.AUTO, number_tolerance=0.0):
                agree += 1
        sc = (agree / total) if total else float("nan")  # agreement rate

        ptrue_scores.append(pt)
        sc_scores.append(sc)
        y.append(label)

    y = np.array(y)
    ptrue_scores = np.array(ptrue_scores)
    sc_scores = np.array(sc_scores)

    # P(True): low P(True) -> error. score_higher_is_error=False.
    m_ptrue = evaluate_baseline(scores=ptrue_scores, y=y,
                                name="p_true", score_higher_is_error=False)
    # self-consistency: low agreement -> error. score_higher_is_error=False.
    m_sc = evaluate_baseline(scores=sc_scores, y=y,
                             name="self_consistency", score_higher_is_error=False)

    logger.info("=" * 60)
    logger.info("P(True):           AUROC=%.3f, AUPRC=%.3f (n=%d)",
                m_ptrue.auroc, m_ptrue.auprc, m_ptrue.n_samples)
    logger.info("Self-consistency:  AUROC=%.3f, AUPRC=%.3f (n=%d)",
                m_sc.auroc, m_sc.auprc, m_sc.n_samples)

    out = cfg.artifacts_path / "results" / "gpu_baselines.json"
    out.write_text(json.dumps({
        "p_true": {"auroc": m_ptrue.auroc, "auprc": m_ptrue.auprc},
        "self_consistency": {"auroc": m_sc.auroc, "auprc": m_sc.auprc},
    }, indent=2))
    logger.info("Saved to %s", out)
    return 0


def main():
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_dir=cfg.logging.log_dir,
                  log_name="10_gpu_baselines", log_to_file=cfg.logging.log_to_file)
    cache_path = Path(args.cache) if args.cache else (cfg.artifacts_path / "results" / "gpu_baselines_cache.json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if args.generate:
        phase_generate(cfg, args.n_samples, args.sc_temperature, cache_path); return 0
    return phase_score(cfg, cache_path)


if __name__ == "__main__":
    sys.exit(main())
"""Stage 1: Run the LLM on benchmark documents, save extractions and activations.

This is the entry point for the extraction stage. Given a config, it:

  1. Loads the configured benchmark.
  2. Loads the configured model (with quantization, etc.).
  3. For each document, runs the extractor and saves results to disk.
  4. Writes a summary report at the end.

Artifacts written (under {artifacts_dir}/extractions/ and /activations/):

  artifacts/extractions/{doc_id}.json
    Includes: parsed JSON, raw text, token logprobs, per-field metadata,
    timing, finish reason, and parse errors.

  artifacts/activations/{doc_id}.npz
    Includes: per-field activations keyed by "{path_str}__layer{N}".
    Loadable via numpy.load(...) from downstream stages.

  artifacts/extractions/_summary.json
    Aggregate report: counts of successes/failures, JSON parse rates,
    average tokens, total elapsed time, etc.

Usage:
    python scripts/01_extract.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any
from probe_extraction.data.real_kie import RealKIE
import numpy as np
from tqdm import tqdm

from probe_extraction.config import Config, get_hf_token, load_config
from probe_extraction.data.extract_bench import ExtractBench
from probe_extraction.extraction import Extractor, ExtractionResult
from probe_extraction.models import HuggingFaceLLM
from probe_extraction.utils.logging import setup_logging

logger = logging.getLogger(__name__)


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM extraction on benchmark documents.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional override for max documents (debug aid).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load model and benchmark, but exit before generating.",
    )
    return parser.parse_args()


# ============================================================================
# Benchmark loading dispatcher
# ============================================================================

def load_benchmark(cfg: Config, limit_override: int | None = None):
    """Construct the benchmark loader from config.

    Currently only ExtractBench is supported. Adding new benchmarks is a
    matter of extending this dispatch.
    """
    if cfg.data.benchmark == "extract_bench":
        max_docs = limit_override if limit_override is not None else cfg.data.max_documents
        return ExtractBench(
            benchmark_path=cfg.benchmark_path,
            domains=cfg.data.domains or None,
            max_documents=max_docs,
            pdf_backend=cfg.data.pdf_extractor,
        )
    if cfg.data.benchmark == "real_kie":
        max_docs = limit_override if limit_override is not None else cfg.data.max_documents
        return RealKIE(
            benchmark_path=cfg.benchmark_path,
            domains=cfg.data.domains or None,
            max_documents=max_docs,
        )
    if cfg.data.benchmark == "sob":
        from probe_extraction.data.sob import SOB
        max_docs = limit_override if limit_override is not None else cfg.data.max_documents
        return SOB(
            benchmark_path=cfg.benchmark_path,
            split=cfg.data.split,
            domains=cfg.data.domains or None,
            max_documents=max_docs,
        )
    raise ValueError(f"Unknown benchmark: {cfg.data.benchmark!r}")


# ============================================================================
# Persistence
# ============================================================================

def _parse_reasoning_token_layers(spec: str, available: list[int]) -> list[int]:
    """Parse REASONING_TOKEN_LAYERS ("16,19,23,26") into a sorted subset of the
    layers actually being captured. Empty/invalid -> [] (capture disabled)."""
    if not spec.strip():
        return []
    want = set()
    for part in spec.split(","):
        part = part.strip()
        if part.isdigit():
            want.add(int(part))
    return sorted(want & set(available))


def save_extraction_metadata(
    result: ExtractionResult,
    extractions_dir: Path,
) -> None:
    """Save the non-activation parts of an ExtractionResult as JSON.

    The activations live in a separate .npz file so this JSON stays human-
    inspectable. Per-field activations are not duplicated here; only their
    paths and metadata.
    """
    fields_meta: list[dict[str, Any]] = []
    for f in result.fields:
        fields_meta.append(
            {
                "path": f.path,
                "path_str": f.path_str,
                "value": f.value,
                "is_empty": f.is_empty,
                "token_span": list(f.token_span),
                "activation_layers": sorted(f.activations.keys()),
            }
        )

    payload = {
        "doc_id": result.doc_id,
        "domain": result.domain,
        "prompt_token_count": result.prompt_token_count,
        "generated_token_count": result.generated_token_count,
        "finish_reason": result.finish_reason,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "raw_generated_text": result.raw_generated_text,
        "parsed_json": result.parsed_json,
        "parse_error": result.parse_error,
        "token_logprobs": result.token_logprobs,
        "captured_layers": result.captured_layers,
        "fields": fields_meta,
    }

    out_path = extractions_dir / f"{result.doc_id}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_activations(
    result: ExtractionResult,
    activations_dir: Path,
) -> None:
    """Save per-field activations to a single compressed .npz file.

    Keys follow the pattern: "{path_str}__layer{N}".
    Loadable via:
        with np.load(path) as data:
            vec = data["personalInfo.fullName__layer20"]
    """
    arrays: dict[str, np.ndarray] = {}
    for f in result.fields:
        for layer, vec in f.activations.items():
            key = f"{f.path_str}__layer{layer}"
            arrays[key] = vec

    # Reasoning-trace pooled vectors (reasoning models only). Reserved "__"
    # prefix never collides with a JSON field path_str, so existing per-field
    # loaders ignore these keys; the fused reasoning probe reads them explicitly.
    # e.g. "__reasoning_mean__layer18", "__reasoning_last__layer18".
    for pool_name, per_layer in getattr(result, "reasoning_activations", {}).items():
        for layer, vec in per_layer.items():
            arrays[f"__{pool_name}__layer{layer}"] = vec

    # Per-token reasoning-trace states for offline field-localized attribution.
    # Key "__reasoning_tokens__layer{N}" -> (n_reasoning_tokens, hidden_dim).
    # Reserved "__" prefix is ignored by existing per-field loaders. The aligned
    # token surface strings go to a small JSON sidecar (variable-length text
    # doesn't belong in an npz).
    token_states = getattr(result, "reasoning_token_states", {})
    for layer, mat in token_states.items():
        arrays[f"__reasoning_tokens__layer{layer}"] = mat

    if not arrays:
        return  # nothing to save

    out_path = activations_dir / f"{result.doc_id}.npz"
    np.savez_compressed(out_path, **arrays)

    token_strings = getattr(result, "reasoning_token_strings", [])
    if token_states and token_strings:
        side = activations_dir / f"{result.doc_id}.rtokens.json"
        with side.open("w", encoding="utf-8") as fh:
            json.dump(token_strings, fh, ensure_ascii=False)


def write_summary(
    results: list[ExtractionResult],
    summary_path: Path,
    total_elapsed: float,
) -> None:
    """Write an aggregate JSON report describing the run."""
    n = len(results)
    n_success = sum(r.is_success for r in results)
    n_parse_failed = sum(r.parse_error is not None for r in results)
    n_no_text = sum(
        r.parse_error is not None
        and "document has no text" in (r.parse_error or "")
        for r in results
    )
    n_finish_length = sum(r.finish_reason == "length" for r in results)
    total_fields = sum(len(r.fields) for r in results)
    total_empty_fields = sum(
        sum(1 for fld in r.fields if fld.is_empty) for r in results
    )
    total_generated = sum(r.generated_token_count for r in results)

    summary = {
        "total_documents": n,
        "successful": n_success,
        "parse_failed": n_parse_failed,
        "skipped_no_text": n_no_text,
        "finish_length_truncated": n_finish_length,
        "total_fields_extracted": total_fields,
        "total_empty_fields": total_empty_fields,
        "total_generated_tokens": total_generated,
        "total_elapsed_seconds": round(total_elapsed, 1),
        "per_document": [
            {
                "doc_id": r.doc_id,
                "domain": r.domain,
                "is_success": r.is_success,
                "finish_reason": r.finish_reason,
                "generated_tokens": r.generated_token_count,
                "fields_extracted": len(r.fields),
                "elapsed_seconds": round(r.elapsed_seconds, 2),
                "parse_error": r.parse_error,
            }
            for r in results
        ],
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    # Set up logging FIRST so model-loading messages are captured.
    log_path = setup_logging(
        level=cfg.logging.level,
        log_dir=cfg.logging.log_dir,
        log_name="01_extract",
        log_to_file=cfg.logging.log_to_file,
    )
    if log_path:
        logger.info("Logging to %s", log_path)

    logger.info("Experiment: %s", cfg.experiment.name)
    logger.info("Model: %s (quantization=%s)", cfg.model.name, cfg.model.quantization)
    logger.info("Benchmark: %s, domains=%s", cfg.data.benchmark, cfg.data.domains)
    logger.info("Activation layers: %s, position=%s",
                cfg.activations.layers, cfg.activations.position)

    # ------ Prepare output directories ------
    artifacts = cfg.artifacts_path
    extractions_dir = artifacts / "extractions"
    activations_dir = artifacts / "activations"
    extractions_dir.mkdir(parents=True, exist_ok=True)
    activations_dir.mkdir(parents=True, exist_ok=True)

    # ------ Load benchmark ------
    benchmark = load_benchmark(cfg, limit_override=args.limit)
    logger.info("Loaded benchmark: %d documents", len(benchmark))

    # ------ Load model ------
    hf_token = get_hf_token()
    if hf_token is None:
        logger.warning(
            "No HF_TOKEN found (Kaggle Secrets or env). Public models will "
            "still load but rate limits may apply."
        )

    llm = HuggingFaceLLM(
        model_name=cfg.model.name,
        dtype=cfg.model.dtype,
        quantization=cfg.model.quantization,
        device_map=cfg.model.device_map,
        trust_remote_code=cfg.model.trust_remote_code,
        hf_token=hf_token,
    )

    # ------ Reasoning-token capture (for offline field-localized attribution) --
    # Persist per-token reasoning states for a small layer subset. Controlled by
    # env so no config schema change is needed:
    #   REASONING_TOKEN_LAYERS="16,19,23,26"  (empty string disables capture)
    #   REASONING_TOKEN_CAP="2048"            (0 = no cap)
    rt_layers = _parse_reasoning_token_layers(
        os.environ.get("REASONING_TOKEN_LAYERS", ""), cfg.activations.layers,
    )
    rt_cap = int(os.environ.get("REASONING_TOKEN_CAP", "0") or "0")
    if rt_layers:
        logger.info("Reasoning-token capture ON for layers %s (cap=%d).", rt_layers, rt_cap)

    # ------ Build extractor ------
    extractor = Extractor(
        llm=llm,
        layers=cfg.activations.layers,
        position=cfg.activations.position,
        max_new_tokens=cfg.model.max_new_tokens,
        temperature=cfg.model.temperature,
        top_p=cfg.model.top_p,
        include_schema=cfg.extraction.include_schema,
        max_input_chars=cfg.extraction.max_input_chars,
        reasoning_token_layers=rt_layers,
        reasoning_token_cap=rt_cap,
    )

    if args.dry_run:
        logger.info("Dry run: model and benchmark loaded successfully. Exiting.")
        return 0

    # ------ Run extraction ------
    results: list[ExtractionResult] = []
    run_start = time.perf_counter()
    for doc in tqdm(benchmark, desc="Extracting", total=len(benchmark)):
        try:
            result = extractor.extract(doc)
        except Exception as e:
            # Last-resort safety net. We never want one document's failure
            # to stop the whole run. Log fully and synthesize an error result.
            logger.exception("Unhandled error on %s: %s", doc.doc_id, e)
            result = ExtractionResult(
                doc_id=doc.doc_id,
                domain=doc.domain,
                prompt_token_count=0,
                generated_token_count=0,
                finish_reason="error",
                elapsed_seconds=0.0,
                raw_generated_text="",
                parsed_json=None,
                parse_error=f"unhandled exception: {type(e).__name__}: {e}",
                token_logprobs=None,
                captured_layers=cfg.activations.layers,
            )

        # Save immediately, per document, so a crash mid-run preserves
        # everything completed up to that point.
        save_extraction_metadata(result, extractions_dir)
        save_activations(result, activations_dir)
        results.append(result)

    total_elapsed = time.perf_counter() - run_start

    # ------ Summary ------
    summary_path = extractions_dir / "_summary.json"
    write_summary(results, summary_path, total_elapsed)

    # Recompute aggregates for the final log line. Single pass, no surprises.
    n_total = len(results)
    n_success = sum(r.is_success for r in results)
    n_no_text = sum(
        1 for r in results
        if r.parse_error is not None
        and "document has no text" in r.parse_error
    )
    n_parse_failed = sum(
        1 for r in results
        if r.parse_error is not None
        and "document has no text" not in r.parse_error
    )

    logger.info("=" * 70)
    logger.info(
        "Done: %d/%d successful, %d parse-failed, %d skipped (no text). "
        "Total: %.1fs.",
        n_success, n_total, n_parse_failed, n_no_text, total_elapsed,
    )
    logger.info("Summary: %s", summary_path)
    logger.info("Extractions: %s", extractions_dir)
    logger.info("Activations: %s", activations_dir)

    return 0

if __name__ == "__main__":
    sys.exit(main())
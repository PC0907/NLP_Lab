"""Stage 2: Label extracted fields against gold annotations.

Reads the extraction artifacts from Stage 1, runs the matcher against the
benchmark's gold annotations, and writes per-document label files. Also
writes an aggregate summary describing the label distribution — we want
to know up front how many errors we have, of what kinds, before training.

Artifacts written:
  artifacts/labels/{doc_id}.json       — per-field labels
  artifacts/labels/_summary.json       — aggregate stats

Usage:
    python scripts/02_label.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
sys.setrecursionlimit(20000)
from pathlib import Path
from typing import Any

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent))
from importlib import import_module
load_benchmark = import_module("01_extract").load_benchmark

from probe_extraction.config import load_config
from probe_extraction.labeling.matcher import LabelingResult, label_extraction
from probe_extraction.labeling.value_compare import ComparisonStrategy

# Map config match_mode -> matcher (leaf_default, structure_aware) settings.
_MODE_PARAMS = {
    "strict":          dict(leaf_default=ComparisonStrategy.EXACT, structure_aware=False),
    "auto":            dict(leaf_default=ComparisonStrategy.AUTO,  structure_aware=False),
    "structure_aware": dict(leaf_default=ComparisonStrategy.AUTO,  structure_aware=True),
}

# from probe_extraction.config import load_config
# from probe_extraction.data.extract_bench import ExtractBench
# from probe_extraction.labeling.matcher import LabelingResult, label_extraction
from probe_extraction.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Label Stage 1 extractions.")
    p.add_argument("--config", required=True)
    return p.parse_args()


def load_extraction(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_labels(result: LabelingResult, labels_dir: Path) -> None:
    """Save labels as plain JSON. Lightweight; consumed by Stage 3."""
    payload = {
        "doc_id": result.doc_id,
        "domain": result.domain,
        "n_total": result.n_total,
        "n_errors": result.n_errors,
        "n_hallucinations": result.n_hallucinations,
        "n_omissions": result.n_omissions,
        "n_value_mismatches": result.n_value_mismatches,
        "n_type_mismatches": result.n_type_mismatches,
        "labels": [
            {
                "path": lab.path,
                "path_str": lab.path_str,
                "is_error": lab.is_error,
                "error_type": lab.error_type,
                "comparison_strategy": lab.comparison_strategy,
                "gold_present": lab.gold_present,
                "extracted_present": lab.extracted_present,
                "gold_value": lab.gold_value,
                "extracted_value": lab.extracted_value,
            }
            for lab in result.labels
        ],
    }
    out = labels_dir / f"{result.doc_id}.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def write_summary(results: list[LabelingResult], summary_path: Path) -> None:
    n_total = sum(r.n_total for r in results)
    n_errors = sum(r.n_errors for r in results)
    summary = {
        "n_documents": len(results),
        "n_total_fields": n_total,
        "n_errors": n_errors,
        "error_rate": n_errors / n_total if n_total else 0.0,
        "n_hallucinations": sum(r.n_hallucinations for r in results),
        "n_omissions": sum(r.n_omissions for r in results),
        "n_value_mismatches": sum(r.n_value_mismatches for r in results),
        "n_type_mismatches": sum(r.n_type_mismatches for r in results),
        "per_document": [
            {
                "doc_id": r.doc_id,
                "n_total": r.n_total,
                "n_errors": r.n_errors,
                "error_rate": round(r.error_rate, 3),
            }
            for r in results
        ],
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    setup_logging(
        level=cfg.logging.level,
        log_dir=cfg.logging.log_dir,
        log_name="02_label",
        log_to_file=cfg.logging.log_to_file,
    )

    artifacts = cfg.artifacts_path
    extractions_dir = artifacts / "extractions"
    labels_dir = artifacts / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # We need gold annotations AND per-document schemas from the benchmark.
    # Per-document (not per-domain) schema is required for benchmarks like SOB
    # where every record carries its own JSON Schema; for ExtractBench this is
    # simply the domain schema repeated, so it is correct for both.
    benchmark = load_benchmark(cfg)
    gold_by_id: dict[str, dict[str, Any]] = {}
    schema_by_id: dict[str, dict[str, Any]] = {}
    for doc in benchmark:
        gold_by_id[doc.doc_id] = doc.gold
        schema_by_id[doc.doc_id] = doc.schema

    # Resolve the primary labeling mode (what we save + train on) and prepare
    # an all-modes comparison (for the paper's error-definition table).
    primary_mode = getattr(cfg.labeling, "match_mode", "strict")
    if primary_mode not in _MODE_PARAMS:
        logger.warning("Unknown match_mode %r; falling back to 'strict'.", primary_mode)
        primary_mode = "strict"
    logger.info("Primary labeling match_mode: %s", primary_mode)
    definition_counts = {m: {"n_total": 0, "n_errors": 0, "n_docs": 0} for m in _MODE_PARAMS}

    # Iterate Stage 1 extraction files.
    results: list[LabelingResult] = []
    for ext_path in sorted(extractions_dir.glob("*.json")):
        if ext_path.name.startswith("_"):
            continue  # skip _summary.json
        ext = load_extraction(ext_path)
        doc_id = ext["doc_id"]
        domain = ext["domain"]

        if ext.get("parse_error"):
            logger.info("Skipping %s (extraction failed: %s)",
                        doc_id, ext["parse_error"])
            continue

        if doc_id not in gold_by_id:
            logger.warning("No gold for %s; skipping", doc_id)
            continue

        # Label under ALL modes (cheap, CPU). The primary mode's labels are
        # saved + used downstream; the others feed the comparison summary only.
        try:
            per_mode = {
                m: label_extraction(
                    doc_id=doc_id,
                    domain=domain,
                    schema=schema_by_id.get(doc_id, {}),
                    gold=gold_by_id[doc_id],
                    extracted=ext["parsed_json"],
                    fuzzy_threshold=cfg.labeling.fuzzy_threshold,
                    number_tolerance=cfg.labeling.number_tolerance,
                    **params,
                )
                for m, params in _MODE_PARAMS.items()
            }
        except (RecursionError, Exception) as e:
            logger.warning("Skipping %s (labeling failed: %s: %s)",
                           doc_id, type(e).__name__, e)
            continue

        result = per_mode[primary_mode]
        save_labels(result, labels_dir)
        results.append(result)
        for m, r in per_mode.items():
            definition_counts[m]["n_total"] += r.n_total
            definition_counts[m]["n_errors"] += r.n_errors
            definition_counts[m]["n_docs"] += 1
        logger.info(
            "%s: %d fields, %d errors (%.0f%%) [mode=%s]",
            doc_id, result.n_total, result.n_errors,
            100 * result.error_rate, primary_mode,
        )

    # Error-definition comparison (paper Table 1): error rate under each mode.
    for m, c in definition_counts.items():
        c["error_rate"] = c["n_errors"] / c["n_total"] if c["n_total"] else 0.0
    comp_path = labels_dir / "_definition_comparison.json"
    with comp_path.open("w", encoding="utf-8") as f:
        json.dump({"primary_mode": primary_mode, "modes": definition_counts}, f, indent=2)
    logger.info("Error-definition comparison (error rate by mode): %s",
                {m: round(definition_counts[m]["error_rate"], 3) for m in definition_counts})

    summary_path = labels_dir / "_summary.json"
    write_summary(results, summary_path)

    total_fields = sum(r.n_total for r in results)
    total_errors = sum(r.n_errors for r in results)
    logger.info("=" * 70)
    logger.info(
        "Done: %d documents labeled, %d fields total, %d errors (%.1f%%)",
        len(results), total_fields, total_errors,
        100 * total_errors / max(total_fields, 1),
    )
    logger.info("Labels: %s", labels_dir)
    logger.info("Summary: %s", summary_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
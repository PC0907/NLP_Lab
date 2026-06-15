#!/usr/bin/env python3
"""Gold-coverage / parsing-quality audit: are gold values present in the
PDF-extracted text AT ALL (across ALL fields, not just errors)?

This is the parsing-quality counterpart to 08_fixability_filter.py:
  - 08 asks, of the ERRORS, which are fixable (gold-in-text)?
  - THIS asks, of ALL gold annotations, what fraction survive PDF extraction?

Why it matters:
  "Gold value not in the extracted text" => the model CANNOT extract it; it is
  not there to extract. So this rate is a CEILING on achievable extraction
  accuracy given the PDF parser. A low rate means PDF-parsing loss is capping
  performance, motivating a better extractor (e.g. Docling). It is the metric
  to compare PyMuPDF vs Docling: re-run on each parser's text and compare the
  gold-in-text rate.

Uses the SAME matcher (value_in_text / _normalize) as 08, so numbers are
directly comparable -- the only difference is we do not filter to errors.

Usage (per-domain; document text comes from the benchmark + that domain's
pdf_extractor setting):
    python scripts/08b_gold_coverage.py --config configs/exp_qwen35_4b_credit.yaml
    python scripts/08b_gold_coverage.py --config configs/exp_qwen35_4b_credit_docling.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from probe_extraction.config import load_config
from probe_extraction.utils.logging import setup_logging

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
load_benchmark = import_module("01_extract").load_benchmark
# reuse 08's matcher verbatim so rates are comparable
_fix = import_module("08_fixability_filter")
_normalize = _fix._normalize
value_in_text = _fix.value_in_text

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gold-coverage parsing-quality audit (all fields)")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--max-value-len", type=int, default=80)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_dir=cfg.logging.log_dir,
                  log_name="08b_gold_coverage", log_to_file=cfg.logging.log_to_file)

    artifacts = cfg.artifacts_path
    labels_dir = artifacts / "labels"

    logger.info("Loading benchmark (pdf_extractor=%s)...",
                getattr(cfg.data, "pdf_extractor", "?"))
    benchmark = load_benchmark(cfg)
    doc_text = {}
    for d in benchmark:
        if getattr(d, "extraction_error", None) is None and getattr(d, "text", None):
            doc_text[d.doc_id] = _normalize(d.text)
    logger.info("Loaded text for %d documents.", len(doc_text))

    # Tally over ALL fields (errors + correct).
    n_total = 0
    n_in_text = 0
    n_no_text = 0
    by_method = {}
    # per-domain rollup (the config is usually single-domain, but a doc_id
    # encodes its domain prefix, so we group by that)
    by_domain = {}   # domain -> [total, in_text]
    missing = []     # gold values NOT found in text (the extraction losses)

    for labels_path in sorted(labels_dir.glob("*.json")):
        if labels_path.name.startswith("_"):
            continue
        doc_id = labels_path.stem
        domain = doc_id.split("__")[0] + "__" + doc_id.split("__")[1] \
                 if "__" in doc_id else doc_id
        data = json.load(labels_path.open())
        text_norm = doc_text.get(doc_id)

        for fld in data.get("labels", []):
            # ALL fields, not just errors
            n_total += 1
            by_domain.setdefault(domain, [0, 0])
            by_domain[domain][0] += 1
            gold = fld.get("gold_value")

            if text_norm is None:
                n_no_text += 1
                present, method = False, "doc_text_unavailable"
            else:
                present, method = value_in_text(gold, text_norm, args.max_value_len)

            by_method[method] = by_method.get(method, 0) + 1
            if present:
                n_in_text += 1
                by_domain[domain][1] += 1
            else:
                missing.append({
                    "doc_id": doc_id,
                    "path_str": fld.get("path_str"),
                    "gold_value": gold,
                    "is_error": int(fld.get("is_error", 0)),
                    "method": method,
                })

    # ----- Report -----
    logger.info("=" * 70)
    logger.info("GOLD COVERAGE (gold value present in PDF-extracted text?) -- ALL fields")
    logger.info("  pdf_extractor          : %s", getattr(cfg.data, "pdf_extractor", "?"))
    logger.info("  total gold fields      : %d", n_total)
    if n_total:
        logger.info("  GOLD IN TEXT           : %d (%.1f%%)   <-- parsing-quality rate",
                    n_in_text, 100 * n_in_text / n_total)
        logger.info("  GOLD MISSING           : %d (%.1f%%)",
                    n_total - n_in_text, 100 * (n_total - n_in_text) / n_total)
        logger.info("  (of which no doc text  : %d)", n_no_text)
    logger.info("-" * 70)
    logger.info("Per-domain rate:")
    logger.info("  %-28s %8s %8s %7s", "domain", "fields", "in_text", "rate")
    for dom, (tot, ins) in sorted(by_domain.items()):
        logger.info("  %-28s %8d %8d %6.1f%%", dom, tot, ins, 100 * ins / tot if tot else 0)
    logger.info("-" * 70)
    logger.info("Breakdown by match method:")
    for m, c in sorted(by_method.items(), key=lambda kv: -kv[1]):
        logger.info("  %-24s %d", m, c)
    logger.info("=" * 70)
    logger.info("Interpretation: the GOLD-IN-TEXT rate is a CEILING on extraction")
    logger.info("accuracy for this PDF parser. Compare across parsers (PyMuPDF vs")
    logger.info("Docling) by re-running this on each parser's extracted text.")

    out_path = Path(args.out) if args.out else (artifacts / "results" / "gold_coverage.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "experiment": cfg.experiment.name,
        "pdf_extractor": getattr(cfg.data, "pdf_extractor", None),
        "n_total": n_total,
        "n_in_text": n_in_text,
        "n_missing": n_total - n_in_text,
        "rate": (n_in_text / n_total) if n_total else None,
        "by_domain": {k: {"total": v[0], "in_text": v[1]} for k, v in by_domain.items()},
        "by_method": by_method,
        "missing": missing,
    }, indent=2, ensure_ascii=False))
    logger.info("Wrote gold-coverage detail to %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
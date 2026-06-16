#!/usr/bin/env python3
"""Value-in-text fixability filter.

For every field labelled an ERROR, ask the prerequisite question for
selective regeneration: *is the gold value even present in the document text
the model was given?*

  - If YES  -> the error is potentially FIXABLE: the answer is in the text,
               so a (better) re-extraction could recover it.
  - If NO   -> the error is UNFIXABLE by regeneration: the answer is not in
               the extracted text (PDF-parsing loss, or a gold value that is
               metadata / derived / not literally present). Regenerating it
               cannot help, and arguably it should not count as a model error
               at all.

Why this matters:
  1. It quantifies how much of the error set is contaminated by upstream
     PDF-extraction loss vs genuine model error (a number reviewers will want).
  2. It is the fixability gate for selective regeneration: only regenerate
     (and only fairly score the probe on) errors whose answer is recoverable.

Method:
  Search the document text for the gold value using the SAME AUTO matcher
  logic used for labelling, so "present" is judged consistently. For numeric
  / short values we also try a normalised substring search. This is a
  necessary-but-not-sufficient check (the value can be present but in an
  unextractable context); we document that limitation.

Usage (per-domain, because document text is loaded from the benchmark):
    python scripts/08_fixability_filter.py --config configs/exp_qwen35_4b_swimming.yaml
    python scripts/08_fixability_filter.py --config configs/exp_qwen35_4b_credit.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from probe_extraction.config import load_config
from probe_extraction.utils.logging import setup_logging
from probe_extraction.labeling.value_compare import compare_values, ComparisonStrategy

# reuse extraction's benchmark loader
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
load_benchmark = import_module("01_extract").load_benchmark

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Value-in-text fixability filter")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--max-value-len", type=int, default=80,
                   help="Skip the substring search for gold values longer than "
                        "this (long legal clauses are checked loosely / flagged "
                        "separately).")
    p.add_argument("--out", type=str, default=None,
                   help="Optional path to write per-field fixability JSON.")
    return p.parse_args()


# =============================================================================
# REPLACEMENT for _normalize() in scripts/08_fixability_filter.py
# (08b_gold_coverage.py imports this, so fixing it here fixes both.)
#
# Add  `import html`  near the top of 08_fixability_filter.py (re is already
# imported). Then replace the existing _normalize with the version below.
#
# Why: the old _normalize handled PyMuPDF's clean text but not Docling's
# markdown/entity artifacts, unfairly scoring Docling gold-coverage too low:
#   - HTML entities:   "AT&amp;T"          vs gold "AT&T"
#   - soft hyphens:    "hand\xad written"  (line-break hyphenation) vs "handwritten"
#   - unicode dashes:  \u2010-\u2015,\u2212 vs ascii "-"
#   - irregular spaces:"J. S.  Denker"      (double space) vs "J. S. Denker"
# Applied identically to gold and document text, so the comparison is fair
# across parsers.
# =============================================================================

# unicode dash variants -> ascii hyphen
_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"
_SOFT_HYPHEN = "\u00ad"


def _normalize(s: str) -> str:
    """Lowercase, decode HTML entities, join soft-hyphen line breaks, unify
    dashes, strip formatting punctuation, and collapse ALL whitespace -- a
    tolerant search that is fair across PyMuPDF (clean) and Docling
    (markdown/entities)."""
    import html
    import re
    s = str(s)
    s = html.unescape(s)            # &amp; -> &  (Docling markdown)
    s = s.lower()
    # soft hyphen = line-break hyphenation; remove it AND any whitespace right
    # after it so "hand\xad written" -> "handwritten"
    s = re.sub(_SOFT_HYPHEN + r"\s*", "", s)
    for d in _DASHES:               # unify unicode dashes to ascii hyphen
        s = s.replace(d, "-")
    s = s.replace(",", "").replace("$", "")   # formatting / currency
    s = re.sub(r"\s+", " ", s)      # collapse all whitespace runs
    return s.strip()

def value_in_text(gold_value, document_text_norm: str, max_value_len: int) -> tuple[bool, str]:
    """Return (present, method). Tolerant check for whether the gold value
    appears in the (normalised) document text.

    method is one of: 'exact_substr', 'normalized_substr', 'numeric',
    'too_long_unchecked', 'empty_gold'.
    """
    if gold_value is None:
        return False, "empty_gold"
    g = str(gold_value).strip()
    if g == "":
        return False, "empty_gold"

    # Long values (legal clauses etc.): a literal substring search is
    # unreliable (the model/gold may paraphrase). Flag separately rather than
    # claim present/absent.
    if len(g) > max_value_len:
        # loose check: are most of the distinctive words present?
        words = [w for w in _normalize(g).split() if len(w) > 4]
        if not words:
            return False, "too_long_unchecked"
        hits = sum(1 for w in words if w in document_text_norm)
        present = hits / len(words) >= 0.6
        return present, "too_long_wordmatch"

    gn = _normalize(g)
    if not gn:
        return False, "empty_gold"

    # Numeric: search for the number with/without formatting.
    try:
        num = float(gn)
        # try a few renderings
        candidates = {gn, str(num), str(int(num)) if num.is_integer() else str(num)}
        for c in candidates:
            if c and c in document_text_norm:
                return True, "numeric"
        return False, "numeric"
    except ValueError:
        pass

    # String: normalized substring search.
    if gn in document_text_norm:
        return True, "normalized_substr"
    return False, "normalized_substr"


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_dir=cfg.logging.log_dir,
                  log_name="08_fixability", log_to_file=cfg.logging.log_to_file)

    artifacts = cfg.artifacts_path
    labels_dir = artifacts / "labels"

    # Load benchmark once -> {doc_id: normalized document text}
    logger.info("Loading benchmark (parses documents)...")
    benchmark = load_benchmark(cfg)
    doc_text = {}
    for d in benchmark:
        if getattr(d, "extraction_error", None) is None and getattr(d, "text", None):
            doc_text[d.doc_id] = _normalize(d.text)
    logger.info("Loaded text for %d documents.", len(doc_text))

    # Walk all error fields, check fixability.
    n_errors = 0
    n_fixable = 0
    n_unfixable = 0
    n_no_text = 0
    by_method = {}
    per_field = []

    for labels_path in sorted(labels_dir.glob("*.json")):
        if labels_path.name.startswith("_"):
            continue
        doc_id = labels_path.stem
        data = json.load(labels_path.open())
        text_norm = doc_text.get(doc_id)
        for fld in data.get("labels", []):
            if int(fld.get("is_error", 0)) != 1:
                continue
            # only value_mismatch / hallucination are "did the model get a
            # present value wrong"; omissions are a different question, but we
            # include them since the gold value should still be in the text.
            n_errors += 1
            gold = fld.get("gold_value")

            if text_norm is None:
                n_no_text += 1
                method = "doc_text_unavailable"
                present = False
            else:
                present, method = value_in_text(gold, text_norm, args.max_value_len)

            by_method[method] = by_method.get(method, 0) + 1
            if present:
                n_fixable += 1
            else:
                n_unfixable += 1

            per_field.append({
                "doc_id": doc_id,
                "path_str": fld.get("path_str"),
                "error_type": fld.get("error_type"),
                "gold_value": gold,
                "extracted_value": fld.get("extracted_value"),
                "gold_in_text": present,
                "method": method,
            })

    # ----- Report -----
    logger.info("=" * 70)
    logger.info("FIXABILITY (gold value present in extracted document text?)")
    logger.info("  total errors examined : %d", n_errors)
    if n_errors:
        logger.info("  FIXABLE   (in text)   : %d (%.1f%%)", n_fixable, 100 * n_fixable / n_errors)
        logger.info("  UNFIXABLE (not in text): %d (%.1f%%)", n_unfixable, 100 * n_unfixable / n_errors)
        logger.info("  (of which no doc text : %d)", n_no_text)
    logger.info("-" * 70)
    logger.info("Breakdown by method:")
    for m, c in sorted(by_method.items(), key=lambda kv: -kv[1]):
        logger.info("  %-24s %d", m, c)
    logger.info("=" * 70)
    logger.info("Interpretation:")
    logger.info("  - UNFIXABLE errors cannot be repaired by regeneration (answer")
    logger.info("    not in the text). They are candidates for exclusion as")
    logger.info("    extraction-loss artefacts, and must NOT count against the")
    logger.info("    probe in the regeneration cost-quality analysis.")
    logger.info("  - FIXABLE errors are the fair denominator for selective")
    logger.info("    regeneration.")

    out_path = Path(args.out) if args.out else (artifacts / "results" / "fixability.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "experiment": cfg.experiment.name,
        "n_errors": n_errors,
        "n_fixable": n_fixable,
        "n_unfixable": n_unfixable,
        "n_no_text": n_no_text,
        "by_method": by_method,
        "per_field": per_field,
    }, indent=2, ensure_ascii=False))
    logger.info("Wrote per-field fixability to %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
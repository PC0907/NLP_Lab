#!/usr/bin/env python3
"""Parse PDFs with several extractors and write the text out for comparison.

PURPOSE
-------
Separate parsing from extraction. Every model run currently re-parses every
PDF, so the text an LLM sees is never inspected directly and is only assumed to
be identical across runs. This writes each parser's output to disk once, which:

  1. lets you READ what the model is actually given, rather than inferring it
     from error rates;
  2. tests DETERMINISM -- parse N times, hash, and see whether the output is
     stable (pymupdf should be; docling and MinerU run neural models, so this
     is worth checking rather than assuming);
  3. gives a durable record of what produced a given set of results.

PARSERS
-------
  pymupdf     Reads the embedded text layer. Fast, no models. Baseline.
  docling     Layout- and table-aware; runs table-structure models on GPU.
  camelot     Reads ruled tables directly from the PDF vector content. No
              model. The comparisons suggest this may beat neural parsers on
              born-digital documents with consistent ruling -- which is what
              the swimming tables and the 10-K filings are.
  mineru      Two-stage VLM pipeline; reported as the accuracy leader for
              reading order, which is exactly what failed on the swimming
              tables. Heavy: downloads weights on first run.

Parsers are called DIRECTLY here rather than through the pipeline's config
system, so nothing in src/ needs changing and unavailable parsers just skip.

OUTPUT
------
  data/parsed/<parser>/<doc_id>.txt        the text
  data/parsed/_report.json                 hashes, lengths, timings, determinism

Usage:
  python parse_corpus.py --domains sport/swimming --parsers pymupdf docling camelot mineru --repeats 3
  python parse_corpus.py --domains sport/swimming --parsers pymupdf --repeats 5   # determinism only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUT_ROOT = Path("data/parsed")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/exp_qwen35_4b_pooled_alltokens.yaml",
                   help="Only used to locate the benchmark and its PDFs.")
    p.add_argument("--domains", nargs="+", default=["sport/swimming"])
    p.add_argument("--parsers", nargs="+",
                   default=["pymupdf", "docling", "camelot", "mineru"])
    p.add_argument("--repeats", type=int, default=3,
                   help="Parse each document this many times to test determinism. "
                        "Only the first run's text is written to disk.")
    p.add_argument("--max-docs", type=int, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Parsers. Each returns text or raises. Imports are inside so a missing
# dependency skips that parser instead of killing the script.
# ---------------------------------------------------------------------------

def parse_pymupdf(pdf_path: Path) -> str:
    import fitz
    doc = fitz.open(pdf_path)
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


def parse_docling(pdf_path: Path) -> str:
    from docling.document_converter import DocumentConverter
    conv = DocumentConverter()
    result = conv.convert(str(pdf_path))
    return result.document.export_to_markdown()


def parse_camelot(pdf_path: Path) -> str:
    """Tables only -- camelot extracts ruled/stream tables, not body text.

    Tries lattice first (needs ruling lines and Ghostscript), falls back to
    stream (whitespace-based). Tables are serialised as pipe-delimited rows,
    which preserves the row grouping that flat text extraction destroys.
    """
    import camelot
    chunks = []
    for flavor in ("lattice", "stream"):
        try:
            tables = camelot.read_pdf(str(pdf_path), pages="all", flavor=flavor)
        except Exception as e:
            logger.debug("  camelot %s failed: %s", flavor, e)
            continue
        if len(tables) == 0:
            continue
        chunks.append(f"[camelot: {len(tables)} tables, flavor={flavor}]")
        for i, t in enumerate(tables):
            chunks.append(f"\n--- table {i} (page {t.page}) ---")
            df = t.df
            for _, row in df.iterrows():
                cells = [str(c).replace("\n", " ").strip() for c in row.tolist()]
                chunks.append(" | ".join(cells))
        break  # first flavor that found anything wins
    if not chunks:
        raise RuntimeError("camelot found no tables with either flavor")
    return "\n".join(chunks)


# ---------------------------------------------------------------------------
# REPLACEMENT for parse_mineru() in parse_corpus.py
#
# Signature confirmed against MinerU 3.4.5:
#   do_parse(output_dir, pdf_file_names, pdf_bytes_list, p_lang_list,
#            backend='pipeline', parse_method='auto', formula_enable=True,
#            table_enable=True, ..., f_dump_md=True, ...)
#
# Notes:
#  - backend already defaults to 'pipeline', so no GPU-backend juggling needed.
#  - The bbox-drawing and JSON dumps are turned OFF: they write large files we
#    do not read, and each call would otherwise leave several artefacts behind.
#  - Output goes to a PER-DOCUMENT temp directory. do_parse writes a tree whose
#    exact shape varies by version, so searching a shared directory would pick
#    up the previous document's markdown. A fresh directory per call makes the
#    rglob unambiguous.
# ---------------------------------------------------------------------------

def parse_mineru(pdf_path: Path) -> str:
    import shutil
    import tempfile
    from mineru.cli.common import do_parse

    out_dir = Path(tempfile.mkdtemp(prefix="mineru_"))
    try:
        do_parse(
            str(out_dir),
            [pdf_path.stem],
            [pdf_path.read_bytes()],
            ["en"],
            backend="pipeline",
            parse_method="auto",
            formula_enable=False,      # no formulas in these documents
            table_enable=True,         # the whole point
            f_draw_layout_bbox=False,  # debug images we do not read
            f_draw_span_bbox=False,
            f_dump_middle_json=False,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=False,
            f_dump_md=True,            # the only output we want
        )
        mds = sorted(out_dir.rglob("*.md"))
        if not mds:
            tree = [str(p.relative_to(out_dir)) for p in out_dir.rglob("*")][:20]
            raise RuntimeError(f"no markdown produced; output tree: {tree}")
        # Prefer a file whose stem matches the document, else take the largest.
        match = [m for m in mds if pdf_path.stem in m.stem]
        chosen = match[0] if match else max(mds, key=lambda p: p.stat().st_size)
        return chosen.read_text(encoding="utf-8", errors="replace")
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)

PARSERS = {
    "pymupdf": parse_pymupdf,
    "docling": parse_docling,
    "camelot": parse_camelot,
    "mineru": parse_mineru,
}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def main():
    args = parse_args()

    # Locate the documents through the benchmark loader, so doc_ids match the
    # rest of the pipeline.
    sys.path.insert(0, "scripts")
    from importlib import import_module
    from probe_extraction.config import load_config
    ex = import_module("01_extract")

    cfg = load_config(args.config)
    cfg.data.domains = args.domains
    docs = list(ex.load_benchmark(cfg))
    if args.max_docs:
        docs = docs[:args.max_docs]
    logger.info("Benchmark: %d documents in %s", len(docs), args.domains)

    # The loader gives Documents, but we need the PDF paths. source_path should
    # carry it; fall back to searching the benchmark directory by stem.
    def pdf_for(doc):
        sp = getattr(doc, "source_path", None)
        if sp and Path(sp).exists():
            return Path(sp)
        stem = doc.doc_id.split("__")[-1]
        hits = list(Path(cfg.benchmark_path).rglob(f"{stem}.pdf"))
        return hits[0] if hits else None

    report = {}
    for pname in args.parsers:
        if pname not in PARSERS:
            logger.warning("Unknown parser %r, skipping.", pname)
            continue
        fn = PARSERS[pname]
        out_dir = OUT_ROOT / pname
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("=" * 72)
        logger.info("PARSER: %s", pname)
        pres = {}

        for doc in docs:
            pdf = pdf_for(doc)
            if pdf is None:
                logger.warning("  %s: no PDF found, skipping", doc.doc_id)
                continue

            hashes, lengths, times = [], [], []
            first_text = None
            failed = None
            for r in range(args.repeats):
                t0 = time.time()
                try:
                    text = fn(pdf)
                except Exception as e:
                    failed = f"{type(e).__name__}: {e}"
                    logger.warning("  %s run %d FAILED: %s", doc.doc_id[:40], r + 1, failed)
                    if r == 0:
                        break
                    continue
                times.append(time.time() - t0)
                hashes.append(sha(text))
                lengths.append(len(text))
                if first_text is None:
                    first_text = text

            if failed and first_text is None:
                pres[doc.doc_id] = {"error": failed}
                continue

            (out_dir / f"{doc.doc_id}.txt").write_text(first_text, encoding="utf-8")
            deterministic = len(set(hashes)) == 1
            pres[doc.doc_id] = {
                "chars": lengths[0],
                "lengths": lengths,
                "hashes": hashes,
                "deterministic": deterministic,
                "mean_seconds": sum(times) / len(times) if times else None,
            }
            flag = "" if deterministic else "   <-- NOT DETERMINISTIC"
            logger.info("  %-44s %7d chars  %5.1fs%s",
                        doc.doc_id[:44], lengths[0],
                        sum(times) / len(times) if times else 0.0, flag)

        report[pname] = pres

    # ---- summary ----------------------------------------------------------
    logger.info("=" * 72)
    logger.info("SUMMARY")
    logger.info("%-12s %6s %10s %10s %8s", "parser", "docs", "mean chars", "mean secs", "stable")
    for pname, pres in report.items():
        ok = {k: v for k, v in pres.items() if "error" not in v}
        if not ok:
            logger.info("%-12s %6s %10s %10s %8s", pname, 0, "-", "-", "-")
            continue
        mc = sum(v["chars"] for v in ok.values()) / len(ok)
        ms = sum(v["mean_seconds"] or 0 for v in ok.values()) / len(ok)
        stable = all(v["deterministic"] for v in ok.values())
        logger.info("%-12s %6d %10.0f %10.1f %8s", pname, len(ok), mc, ms,
                    "yes" if stable else "NO")

    # cross-parser character-count comparison, per document
    logger.info("-" * 72)
    logger.info("CHARACTER COUNTS PER DOCUMENT")
    all_docs = sorted({d for pres in report.values() for d in pres})
    head = f"{'document':44}" + "".join(f"{p:>12}" for p in report)
    logger.info(head)
    for d in all_docs:
        row = f"{d[:44]:44}"
        for pname in report:
            v = report[pname].get(d)
            row += f"{v['chars']:>12}" if v and "error" not in v else f"{'-':>12}"
        logger.info(row)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "_report.json").write_text(json.dumps(report, indent=2))
    logger.info("=" * 72)
    logger.info("Text written under %s/<parser>/  -- read it directly.", OUT_ROOT)
    logger.info("Report: %s/_report.json", OUT_ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
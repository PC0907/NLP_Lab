#!/usr/bin/env python3
"""Investigate ONE financial document end-to-end to separate
'PDF-extractor failing' from 'model bad at this domain'.

For a chosen doc it produces, side by side:
  1. RAW extracted text from BOTH parsers (PyMuPDF and Docling) -> written to
     files so you can open and eyeball whether the financial TABLES survived.
  2. Per ERROR field: gold value, what the model extracted, and whether the gold
     value is FINDABLE in each parser's text (exact / normalized / numeric / not
     at all) -- so you can see if the value was even there to extract.
  3. A cause bucket per error (matcher-artifact / gold-not-in-text /
     scale-value-undocumented / genuine), reusing the taxonomy logic.

This lets you judge, field by field: was it the PARSER (value not in text /
table mangled) or the MODEL (value present, model still wrong) or the BENCHMARK
(undocumented scale/value convention)?

Usage (run from repo root):
  python scripts/investigate_financial_doc.py \
      --config configs/exp_qwen35_4b_10kq.yaml \
      --doc adp_10q_fy2025q2 \
      --outdir artifacts/qwen35_4b_10kq/results/investigate

If --doc is omitted, uses the first doc in the benchmark.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
from probe_extraction.config import load_config

load_benchmark = import_module("01_extract").load_benchmark
_fix = import_module("08_fixability_filter")
_normalize = _fix._normalize
value_in_text = _fix.value_in_text
_rs = import_module("rescore_regen")
lenient_ok = _rs.lenient_ok

# direct access to the two parser functions
from probe_extraction.data.pdf_utils import extract_text


def find_pdf(benchmark, doc):
    """Locate the doc object + its PDF path."""
    for d in benchmark:
        if doc is None or doc in d.doc_id:
            return d
    return None


def value_findable(gold, text_norm, max_len=80):
    if text_norm is None:
        return ("no_text", False)
    present, method = value_in_text(gold, text_norm, max_len)
    return (method, present)


def classify(gold, extracted, py_present, dl_present):
    if gold is None or str(gold).strip() == "":
        return "empty_gold"
    if lenient_ok(gold, extracted):
        return "matcher_artifact"
    # present in neither parser -> gold not in text (parser-independent ceiling)
    if not py_present and not dl_present:
        return "gold_not_in_text(both)"
    if py_present and not dl_present:
        return "in_pymupdf_only"
    if dl_present and not py_present:
        return "in_docling_only"
    # present in both, model still wrong -> genuine model error OR undocumented convention
    return "present_model_wrong"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--doc", default=None, help="doc_id substring")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    bench = load_benchmark(cfg)
    d = find_pdf(bench, args.doc)
    if d is None:
        print(f"no doc matching {args.doc!r}"); return 1
    doc_id = d.doc_id
    print(f"DOC: {doc_id}")

    pdf_path = getattr(d, "source_path", None) or getattr(d, "pdf_path", None) or getattr(d, "path", None)
    print(f"PDF path: {pdf_path}")
    if pdf_path is None:
        print("ERROR: could not resolve the PDF path from the doc object.")
        print("doc attrs:", [a for a in dir(d) if not a.startswith('_')])
        return 1

    outdir = Path(args.outdir) if args.outdir else (cfg.artifacts_path / "results" / "investigate")
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- 1. RAW text from both parsers ----
    print("\n=== extracting raw text with BOTH parsers ===")
    texts = {}
    for backend in ("pymupdf", "docling"):
        try:
            t = extract_text(Path(pdf_path), backend=backend)
        except Exception as e:
            t = f"<{backend} failed: {e}>"
        texts[backend] = t
        outfile = outdir / f"{doc_id}__{backend}.txt"
        outfile.write_text(t, encoding="utf-8")
        print(f"  {backend}: {len(t)} chars -> {outfile}")

    # sanity: compare to the benchmark's own already-extracted text (d.text),
    # which uses this config's pdf_extractor. If our re-extraction is wildly
    # shorter, the re-extraction failed (e.g. bad path) -> warn loudly.
    benchmark_text_len = len(getattr(d, "text", "") or "")
    print(f"  [sanity] benchmark d.text length: {benchmark_text_len} chars "
          f"(config pdf_extractor={getattr(cfg.data, 'pdf_extractor', '?')})")
    for backend in ("pymupdf", "docling"):
        if texts[backend].startswith("<") or len(texts[backend]) < 0.5 * max(benchmark_text_len, 1):
            print(f"  [WARN] {backend} extraction looks failed/short "
                  f"({len(texts[backend])} chars vs benchmark {benchmark_text_len}).")

    py_norm = _normalize(texts["pymupdf"]) if not texts["pymupdf"].startswith("<") else None
    dl_norm = _normalize(texts["docling"]) if not texts["docling"].startswith("<") else None

    # ---- 2 & 3. per-error-field findability + bucket ----
    labels_path = cfg.artifacts_path / "labels" / f"{doc_id}.json"
    if not labels_path.exists():
        print(f"\n[warn] no labels at {labels_path}; skipping per-field analysis.")
        return 0
    labels = json.load(labels_path.open())

    rows = []
    for fld in labels.get("labels", []):
        if int(fld.get("is_error", 0)) != 1:
            continue
        gold = fld.get("gold_value")
        extracted = fld.get("extracted_value")
        py_m, py_p = value_findable(gold, py_norm)
        dl_m, dl_p = value_findable(gold, dl_norm)
        bucket = classify(gold, extracted, py_p, dl_p)
        rows.append({
            "path": fld.get("path_str"), "gold": gold, "extracted": extracted,
            "in_pymupdf": py_p, "py_method": py_m,
            "in_docling": dl_p, "dl_method": dl_m,
            "bucket": bucket,
        })

    # console table
    print(f"\n=== {len(rows)} error fields ===")
    print(f"{'path':40s} {'pymupdf':8s} {'docling':8s} {'bucket'}")
    for r in rows:
        print(f"{r['path'][:40]:40s} {str(r['in_pymupdf']):8s} {str(r['in_docling']):8s} {r['bucket']}")
        print(f"    gold={str(r['gold'])[:50]!r}  extracted={str(r['extracted'])[:50]!r}")

    # bucket summary
    from collections import Counter
    cnt = Counter(r["bucket"] for r in rows)
    print(f"\n=== bucket summary ({len(rows)} errors) ===")
    for b, c in cnt.most_common():
        print(f"  {b:28s} {c}")

    # interpretation
    print("\n=== interpretation ===")
    parser_loss = cnt.get("gold_not_in_text(both)", 0)
    docling_helps = cnt.get("in_docling_only", 0)
    pymupdf_helps = cnt.get("in_pymupdf_only", 0)
    model_or_conv = cnt.get("present_model_wrong", 0)
    artifact = cnt.get("matcher_artifact", 0) + cnt.get("empty_gold", 0)
    print(f"  gold in NEITHER parser (true ceiling, no parser helps): {parser_loss}")
    print(f"  gold in Docling but NOT PyMuPDF (switching would help):  {docling_helps}")
    print(f"  gold in PyMuPDF but NOT Docling (PyMuPDF better here):   {pymupdf_helps}")
    print(f"  gold present in BOTH, model still wrong (MODEL or undoc. convention): {model_or_conv}")
    print(f"  matcher/empty artifacts (not real errors):              {artifact}")

    # write json
    (outdir / f"{doc_id}__investigation.json").write_text(json.dumps({
        "doc_id": doc_id, "pdf_path": str(pdf_path),
        "pymupdf_chars": len(texts["pymupdf"]), "docling_chars": len(texts["docling"]),
        "n_errors": len(rows), "buckets": dict(cnt), "fields": rows,
    }, indent=2, ensure_ascii=False))
    print(f"\nWrote investigation JSON + raw texts to {outdir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
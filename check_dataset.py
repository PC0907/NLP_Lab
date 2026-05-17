#!/usr/bin/env python3
"""Dataset gate report for the probe pipeline.

Run BEFORE adopting any candidate dataset into the probe pipeline.

History -- why this script has the checks it does:
  * The NDA dataset passed a casual look ("~2 spans/doc, looks populated")
    but was sparsely annotated: gold captured only some of the correct
    values, so labeling produced 55/55 fake errors. Sections 2 and 4 were
    added to surface that.
  * fcc_invoices then PASSED sections 1-5 but still failed in the pipeline,
    for two reasons section 1-5 did not check:
      (a) Per-document completeness: some docs annotate only a subset of
          fields (e.g. only start_date), so "not in gold" is unreliable
          PER DOCUMENT even when the dataset looks dense in aggregate.
      (b) Tabular structure: line-item fields are rows of a table. The
          pipeline's set-membership matching pools all row values into a
          bag and loses row alignment, turning correct extractions into
          fake mismatches.
    Sections 6 and 7 were added to catch these.

It deliberately renders NO pass/fail verdict. It reports numbers; you decide.
A dataset is a good fit for THIS pipeline when its fields are flat (one value
or an unordered set per field, NOT table rows) and gold is complete both in
aggregate AND per document.

What it measures (automated)
----------------------------
  * Document count (per split and pooled).
  * Per-label span totals.
  * Empty-gold rate per label -- fraction of docs with zero spans for that
    label. NDA's killer: whole docs with no gold at all.
  * Spans-per-populated-doc: min / median / max, and a note if a label sits
    suspiciously flat at exactly 1.0 (hints annotators caught only the first
    occurrence).
  * Offset integrity: whether each span's [start:end] slice of the document
    text equals the span's own stored "text" field. Mismatches mean the
    offsets are unreliable.
  * Heuristic under-annotation signal for regex-detectable field types
    (dates, money amounts): how many such strings exist in the document text
    versus how many gold captured. A large gap is concrete evidence of
    under-annotation. This is a HEURISTIC -- it has false positives (not
    every date in an invoice is a line-item date) -- so it is reported as a
    ratio for your interpretation, not a verdict.

What it cannot do
-----------------
  Completeness ("is EVERY correct value annotated?") cannot be decided by a
  script -- it needs a human reading the document against its gold. The
  MANUAL INSPECTION section prints, for a few sampled docs, every gold value
  with a window of surrounding text, so that read is fast.

Usage
-----
    python check_dataset.py <dataset_dir>
    python check_dataset.py ~/NLP_Lab/datasets/realkie/fcc_invoices
    python check_dataset.py ~/NLP_Lab/datasets/realkie/fcc_invoices --samples 8
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

SPLIT_FILES = ("train.csv", "val.csv", "test.csv")

# Heuristic detectors for regex-detectable entity types. Used only for the
# under-annotation signal; intentionally loose.
_MONEY_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d{2})?")
_DATE_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b"
)


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def load_pooled(dataset_dir: Path) -> list[dict]:
    """Load all split CSVs into a flat list of doc records (dedup by hash)."""
    records: list[dict] = []
    seen: set[str] = set()
    for split_file in SPLIT_FILES:
        csv_path = dataset_dir / split_file
        if not csv_path.exists():
            print(f"  note: missing split file {csv_path.name}")
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            doc_path = row.get("document_path")
            if not isinstance(doc_path, str):
                continue
            doc_hash = Path(doc_path).stem
            if doc_hash in seen:
                continue
            seen.add(doc_hash)

            labels_raw = row.get("labels")
            try:
                spans = json.loads(labels_raw) if isinstance(labels_raw, str) else []
            except json.JSONDecodeError:
                spans = []

            text = row.get("text")
            records.append({
                "doc_hash": doc_hash,
                "split": csv_path.stem,
                "text": text if isinstance(text, str) else "",
                "spans": spans if isinstance(spans, list) else [],
            })
    return records


def report_counts(records: list[dict]) -> list[str]:
    """Document and split counts. Returns the list of label names found."""
    banner("1. DOCUMENT COUNTS")
    by_split = Counter(r["split"] for r in records)
    for split, n in sorted(by_split.items()):
        print(f"  {split:8s} {n:5d} docs")
    print(f"  {'POOLED':8s} {len(records):5d} docs")

    label_set = sorted({
        s.get("label") for r in records for s in r["spans"]
        if isinstance(s, dict) and s.get("label")
    })
    print(f"\n  distinct labels: {len(label_set)}")
    return label_set


def report_label_density(records: list[dict], labels: list[str]) -> None:
    """Per-label span totals, empty-gold rate, spans-per-populated-doc."""
    banner("2. PER-LABEL DENSITY  (the NDA failure would show here)")

    n_docs = len(records)
    # spans-per-doc, per label
    per_doc_counts: dict[str, list[int]] = defaultdict(list)
    for r in records:
        c = Counter(
            s.get("label") for s in r["spans"]
            if isinstance(s, dict) and s.get("label")
        )
        for lab in labels:
            per_doc_counts[lab].append(c.get(lab, 0))

    header = f"  {'label':28s} {'spans':>7s} {'empty%':>7s} {'min':>4s} {'med':>5s} {'max':>5s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for lab in labels:
        counts = per_doc_counts[lab]
        total = sum(counts)
        n_empty = sum(1 for c in counts if c == 0)
        empty_pct = 100.0 * n_empty / n_docs if n_docs else 0.0
        populated = [c for c in counts if c > 0]
        if populated:
            mn, md, mx = min(populated), statistics.median(populated), max(populated)
        else:
            mn = md = mx = 0
        flat = " <- flat at 1.0" if populated and mn == mx == 1 else ""
        print(f"  {lab:28s} {total:7d} {empty_pct:6.1f}% "
              f"{mn:4d} {md:5.1f} {mx:5d}{flat}")

    print()
    print("  Reading this: a high empty% means many docs have NO gold for that")
    print("  field -- 'not in gold' then cannot be trusted to mean 'wrong'.")
    print("  'flat at 1.0' means every populated doc has exactly one span --")
    print("  suspicious for fields that naturally recur (annotators may have")
    print("  captured only the first occurrence).")


def report_offset_integrity(records: list[dict], sample: int = 60) -> None:
    """Check that span [start:end] slices match the span's stored text."""
    banner("3. OFFSET INTEGRITY")
    checked = mismatched = 0
    examples: list[str] = []
    for r in records[:sample]:
        text = r["text"]
        for s in r["spans"]:
            if not isinstance(s, dict):
                continue
            start, end = s.get("start"), s.get("end")
            stored = s.get("text")
            if not isinstance(start, int) or not isinstance(end, int):
                continue
            if not isinstance(stored, str):
                continue
            checked += 1
            sliced = text[start:end]
            if sliced != stored:
                mismatched += 1
                if len(examples) < 5:
                    examples.append(
                        f"    stored={stored!r}\n    sliced={sliced!r}"
                    )
    if checked == 0:
        print("  no checkable spans in the sampled docs.")
        return
    print(f"  checked {checked} spans in first {sample} docs: "
          f"{mismatched} mismatched ({100.0*mismatched/checked:.1f}%)")
    if examples:
        print("  examples of mismatch:")
        for e in examples:
            print(e)
    else:
        print("  all sampled offsets resolve correctly.")


def report_underannotation(records: list[dict], labels: list[str]) -> None:
    """Heuristic: regex-detectable entities in text vs captured by gold."""
    banner("4. UNDER-ANNOTATION HEURISTIC  (dates & money)")
    print("  Compares how many date-like / money-like strings appear in the")
    print("  document text against how many gold spans exist for date/money")
    print("  fields. A large text>>gold gap is evidence of under-annotation.")
    print("  HEURISTIC ONLY: not every match is a real field. Read the ratio,")
    print("  do not treat it as a verdict.\n")

    date_labels = [l for l in labels if "date" in l.lower()]
    money_labels = [l for l in labels
                    if any(k in l.lower() for k in ("rate", "amount", "total", "commission"))]
    print(f"  treating as DATE fields:  {date_labels or '(none)'}")
    print(f"  treating as MONEY fields: {money_labels or '(none)'}")

    text_dates = gold_dates = text_money = gold_money = 0
    for r in records:
        text = r["text"]
        text_dates += len(_DATE_RE.findall(text))
        text_money += len(_MONEY_RE.findall(text))
        for s in r["spans"]:
            if not isinstance(s, dict):
                continue
            lab = s.get("label")
            if lab in date_labels:
                gold_dates += 1
            elif lab in money_labels:
                gold_money += 1

    def line(kind: str, in_text: int, in_gold: int) -> None:
        ratio = (in_text / in_gold) if in_gold else float("inf")
        print(f"  {kind:8s} text-matches={in_text:7d}  gold-spans={in_gold:7d}  "
              f"text/gold={ratio:.2f}")

    print()
    line("dates", text_dates, gold_dates)
    line("money", text_money, gold_money)
    print()
    print("  ratio near 1.0 -> gold density is consistent with the text")
    print("  ratio >> 1.0   -> many candidate values are not in gold (or the")
    print("                    regex over-fires; confirm in section 5)")


def report_per_doc_coverage(records: list[dict], labels: list[str]) -> None:
    """Per-document field coverage -- catches the fcc_invoices partial-gold case.

    Section 2 measures whether a field is dense ACROSS the dataset. That is
    not enough: a dataset can be dense in aggregate yet have many individual
    documents where only a subset of fields is annotated. On those documents
    "extracted value not in gold" is unreliable for the un-annotated fields,
    exactly the fcc_invoices doc that annotated only start_date.

    This reports, per document, how many of the dataset's fields have at
    least one gold span -- and the distribution of that coverage.
    """
    banner("6. PER-DOCUMENT FIELD COVERAGE  (the fcc_invoices failure)")
    n_fields = len(labels)
    if n_fields == 0:
        print("  no labels.")
        return

    coverage_counts: list[int] = []  # how many fields covered, per doc
    for r in records:
        present = {
            s.get("label") for s in r["spans"]
            if isinstance(s, dict) and s.get("label") in labels
        }
        coverage_counts.append(len(present))

    n_docs = len(records)
    full = sum(1 for c in coverage_counts if c == n_fields)
    none = sum(1 for c in coverage_counts if c == 0)
    partial = n_docs - full - none

    print(f"  dataset has {n_fields} fields; per document, gold covers:")
    print(f"    all {n_fields} fields : {full:5d} docs ({100.0*full/n_docs:.1f}%)")
    print(f"    some fields    : {partial:5d} docs ({100.0*partial/n_docs:.1f}%)")
    print(f"    zero fields    : {none:5d} docs ({100.0*none/n_docs:.1f}%)")

    # Histogram of coverage counts.
    hist = Counter(coverage_counts)
    print("\n  coverage histogram (fields covered -> doc count):")
    for k in range(n_fields + 1):
        bar = "#" * min(50, hist.get(k, 0))
        print(f"    {k:2d} fields: {hist.get(k,0):5d}  {bar}")

    print()
    print("  Reading this: a high 'some fields' or 'zero fields' count means")
    print("  per-document gold is incomplete. The pipeline assumes that if a")
    print("  field is in the schema, gold for it on each doc is complete --")
    print("  partial coverage breaks that and produces fake errors.")


def report_tabular_check(records: list[dict], labels: list[str]) -> None:
    """Flag fields that look like table rows -- the other fcc_invoices failure.

    The pipeline's set-membership matching treats a field's values as an
    unordered set: an extracted value is correct if it equals ANY gold value.
    That is right for genuinely unordered fields (a document's set of party
    names) but wrong for ROWS OF A TABLE, where value position carries
    meaning and the same value (e.g. a rate of '0.00') recurs across rows.

    Heuristic: a field is "tabular-like" if populated docs commonly have many
    spans for it (median well above 1) AND those spans contain heavy
    duplication (the same value repeated). Both are hallmarks of table
    columns. Flat fields (party, jurisdiction) have few, mostly-distinct
    values per doc.
    """
    banner("7. TABULAR-STRUCTURE CHECK  (set-membership matching fitness)")
    print("  The pipeline matches multi-valued fields by set membership,")
    print("  which is correct for unordered sets but WRONG for table columns")
    print("  (row position matters, values repeat). This flags fields whose")
    print("  shape looks tabular.\n")

    header = f"  {'label':28s} {'med/doc':>8s} {'dup-rate':>9s}  assessment"
    print(header)
    print("  " + "-" * (len(header) + 8))
    for lab in labels:
        per_doc_vals: list[list[str]] = []
        for r in records:
            vals = [
                str(s.get("text", "")) for s in r["spans"]
                if isinstance(s, dict) and s.get("label") == lab
            ]
            if vals:
                per_doc_vals.append(vals)
        if not per_doc_vals:
            print(f"  {lab:28s} {'--':>8s} {'--':>9s}  (no gold)")
            continue

        med = statistics.median(len(v) for v in per_doc_vals)
        # duplication rate: 1 - (distinct / total), averaged over docs
        dup_rates = [
            1.0 - len(set(v)) / len(v) for v in per_doc_vals if v
        ]
        dup = statistics.mean(dup_rates) if dup_rates else 0.0

        if med >= 5 and dup >= 0.2:
            verdict = "TABULAR -- set-membership unsafe"
        elif med >= 5:
            verdict = "many values/doc -- inspect"
        else:
            verdict = "flat -- set-membership ok"
        print(f"  {lab:28s} {med:8.1f} {dup:8.1%}  {verdict}")

    print()
    print("  'TABULAR' means the field is a table column: many values per")
    print("  doc, heavy repetition. Set-membership matching loses row")
    print("  alignment for these and will manufacture fake errors. Such a")
    print("  field needs row-aligned matching, which the pipeline does not do.")


def report_manual_samples(records: list[dict], n_samples: int) -> None:
    """Print gold values with surrounding text for human completeness review."""
    banner(f"5. MANUAL INSPECTION  ({n_samples} sampled docs)")
    print("  This is the part a script cannot do for you. For each doc below,")
    print("  read the gold values against the text window and ask: is EVERY")
    print("  correct value of each field actually present in gold?\n")

    # Spread the sample across the pool deterministically.
    if not records:
        print("  no documents.")
        return
    step = max(1, len(records) // n_samples)
    picked = records[::step][:n_samples]

    for r in picked:
        print("  " + "-" * 68)
        print(f"  doc {r['doc_hash']}  (split={r['split']}, "
              f"text length={len(r['text'])} chars, {len(r['spans'])} spans)")
        by_label: dict[str, list[dict]] = defaultdict(list)
        for s in r["spans"]:
            if isinstance(s, dict) and s.get("label"):
                by_label[s["label"]].append(s)

        for lab in sorted(by_label):
            spans = by_label[lab]
            shown = spans[:4]
            vals = ", ".join(repr(s.get("text", "")) for s in shown)
            more = f"  (+{len(spans)-4} more)" if len(spans) > 4 else ""
            print(f"    {lab}: {len(spans)} span(s){more}")
            print(f"      {vals}")
        if not by_label:
            print("    (no gold spans at all)")
        # A short window of the document text for orientation.
        snippet = r["text"][:300].replace("\n", " ")
        print(f"    text head: {snippet!r}")
        print()


def main() -> int:
    ap = argparse.ArgumentParser(description="RealKIE dataset gate report.")
    ap.add_argument("dataset_dir", help="Path to a RealKIE dataset directory.")
    ap.add_argument("--samples", type=int, default=5,
                    help="How many docs to print in the manual section.")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser()
    if not dataset_dir.is_dir():
        print(f"ERROR: not a directory: {dataset_dir}")
        return 1

    print(f"Dataset gate report for: {dataset_dir}")

    records = load_pooled(dataset_dir)
    if not records:
        print("ERROR: no documents loaded -- check the CSV files exist.")
        return 1

    labels = report_counts(records)
    report_label_density(records, labels)
    report_offset_integrity(records)
    report_underannotation(records, labels)
    report_per_doc_coverage(records, labels)
    report_tabular_check(records, labels)
    report_manual_samples(records, args.samples)

    banner("END OF REPORT")
    print("  No verdict is given by design. Decide based on:")
    print("   - section 2: is empty% low enough that 'not in gold' is")
    print("     meaningful for the fields you care about?")
    print("   - section 4: is the text/gold ratio near 1.0?")
    print("   - section 6: do MOST docs cover ALL fields? Partial per-doc")
    print("     coverage produces fake errors even if the dataset looks dense.")
    print("   - section 7: are the fields flat, not tabular? Tabular fields")
    print("     break set-membership matching.")
    print("   - section 5: does your eyeball read confirm gold is complete?")
    print()
    print("  A dataset fits this pipeline only if: gold is complete per")
    print("  document (6), fields are flat not tabular (7), and a manual")
    print("  read confirms completeness (5).")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
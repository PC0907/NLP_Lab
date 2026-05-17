"""RealKIE benchmark loader.

RealKIE (Townsend et al., 2024 — arXiv:2403.20101) provides five enterprise
key-information-extraction datasets. This loader currently supports the NDA
dataset; other datasets (charities, fcc_invoices, s1, resource_contracts) can
be added by registering them in REALKIE_DATASETS below.

Unlike ExtractBench, RealKIE is natively a *span-labeling* benchmark: each
document is annotated with a list of typed character spans rather than a
target JSON object. This loader bridges the two paradigms so the rest of the
pipeline (extractor, labeling, probes, LODO) sees ordinary Documents:

  * Each distinct span label becomes a field in a synthesized JSON Schema.
    For NDA the labels are Party, Jurisdiction, and Effective Date.
  * All three NDA labels occur multiple times per document (Party ~3.8x,
    Jurisdiction ~1.7x, Effective Date ~1.7x), so every field is modeled as
    an array of strings.
  * The per-document span list is converted into a gold JSON object by
    grouping spans by label and collecting their normalized text.

Directory layout (relative to benchmark_path):

    realkie/                       <- benchmark_path points HERE
    └── nda/
        ├── train.csv              # official splits — annotations live here
        ├── val.csv
        ├── test.csv
        ├── raw_export.csv         # unsplit union (unused)
        ├── old_split_files/       # superseded (unused)
        ├── files/                 # source PDFs (used only for source_path)
        ├── ocr/                   # per-doc OCR JSON (unused)
        └── images/                # page renders (unused)

Each split CSV has these columns (alongside several "Unnamed: 0*" index-dump
columns left over from repeated re-saves, which we ignore):

    document_path   "nda/files/<hash>.pdf"  -- <hash> is the doc identifier
    text            full OCR'd document text -- fed directly to the model
    labels          JSON string: [{"label","start","end","text"}, ...]
    original_filename  the pre-anonymization filename (kept as metadata)

Because RealKIE ships OCR text inline in the `text` column, this loader does
NOT touch pdf_utils -- there is no PDF parsing step, and therefore none of the
PDF-backend / truncation variability that affects the ExtractBench loader.

Usage is via the benchmark dispatcher in scripts/01_extract.py: set
`benchmark: real_kie` in the config and point `benchmark_path` at the
directory that contains the `nda/` subdirectory.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from probe_extraction.data.base import Benchmark, Document

logger = logging.getLogger(__name__)


# ============================================================================
# Per-dataset configuration
# ============================================================================
# Each RealKIE dataset has its own label set, hence its own synthesized schema.
# To add another dataset (e.g. fcc_invoices), inspect its label vocabulary and
# add a {schema, label_to_field} entry to REALKIE_DATASETS. Nothing else in
# this file is dataset-specific.

# Each field is an array in "set_membership" match mode. That mode (handled by
# Matcher._emit_set_membership_array_label) emits one label PER extracted
# element and scores each element correct if it matches ANY gold element --
# the right semantics here because (a) every element should be its own probe
# sample, and (b) the model's element order need not match gold's.
#
# The per-element evaluation_config is "string_case_insensitive". Rationale:
# RealKIE gold spans carry OCR/annotation noise (newlines, trailing
# punctuation) that the loader's _normalize_span_text already strips, so gold
# is clean; the remaining expected gap between gold and a correct model
# extraction is casing (e.g. "washington" vs "Washington"). EXACT would
# wrongly flag those; FUZZY (token-Jaccard) risks false MATCHES on short
# values like dates or one-word jurisdictions. CASE_INSENSITIVE is the
# calibrated middle. Revisit if first-run labels look systematically wrong.

_NDA_ELEMENT_SCHEMA: dict[str, Any] = {
    "type": "string",
    "evaluation_config": "string_case_insensitive",
}

_NDA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "party": {
            "type": "array",
            "x-match-mode": "set_membership",
            "items": _NDA_ELEMENT_SCHEMA,
            "description": (
                "Names of the organizations or individuals who are parties "
                "to (i.e. signatories of) this non-disclosure agreement. "
                "Include every distinct party named in the agreement."
            ),
        },
        "jurisdiction": {
            "type": "array",
            "x-match-mode": "set_membership",
            "items": _NDA_ELEMENT_SCHEMA,
            "description": (
                "The governing-law jurisdiction(s) of the agreement: the "
                "state, country, or legal jurisdiction whose laws govern it, "
                "as stated in the governing-law / applicable-law clause."
            ),
        },
        "effective_date": {
            "type": "array",
            "x-match-mode": "set_membership",
            "items": _NDA_ELEMENT_SCHEMA,
            "description": (
                "The effective date(s) of the agreement, exactly as written "
                "in the document (the date on which the agreement takes "
                "effect)."
            ),
        },
    },
    "required": ["party", "jurisdiction", "effective_date"],
}

# Maps RealKIE's raw span labels onto the (filesystem/JSON-safe) schema field
# names. The schema field names are deliberately lowercase with underscores --
# RealKIE's "Effective Date" label has a space, which is valid JSON but risky
# as a field-path component downstream, so it is renamed here.
_NDA_LABEL_TO_FIELD: dict[str, str] = {
    "Party": "party",
    "Jurisdiction": "jurisdiction",
    "Effective Date": "effective_date",
}

# --- fcc_invoices -----------------------------------------------------------
# Line-item fields only. The header fields (Agency, Payment Terms, Agency
# Commission, Advertiser, Gross Total, Net Amount Due) are excluded: the gate
# report showed 20-32% empty-gold rates for several of them whose cause was
# not verified, so they are left out rather than risk NDA-style coverage noise.
# The five line-item fields cleared the gate: dense, low empty%, offsets clean.
# All five are multi-valued (a table has many rows) -> set_membership mode.

_INVOICE_ELEMENT_SCHEMA: dict[str, Any] = {
    "type": "string",
    "evaluation_config": "string_case_insensitive",
}

_INVOICE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "line_item_description": {
            "type": "array",
            "x-match-mode": "set_membership",
            "items": _INVOICE_ELEMENT_SCHEMA,
            "description": (
                "The description of each line item / advertising spot in the "
                "invoice's main table (e.g. the program name or daypart). One "
                "entry per row of the table."
            ),
        },
        "line_item_rate": {
            "type": "array",
            "x-match-mode": "set_membership",
            "items": _INVOICE_ELEMENT_SCHEMA,
            "description": (
                "The rate / cost amount for each line item in the invoice's "
                "main table, exactly as written. One entry per table row."
            ),
        },
        "line_item_days": {
            "type": "array",
            "x-match-mode": "set_membership",
            "items": _INVOICE_ELEMENT_SCHEMA,
            "description": (
                "The days-of-week or scheduling field for each line item in "
                "the invoice's main table, exactly as written. One entry per "
                "table row."
            ),
        },
        "line_item_start_date": {
            "type": "array",
            "x-match-mode": "set_membership",
            "items": _INVOICE_ELEMENT_SCHEMA,
            "description": (
                "The start date for each line item in the invoice's main "
                "table, exactly as written. One entry per table row."
            ),
        },
        "line_item_end_date": {
            "type": "array",
            "x-match-mode": "set_membership",
            "items": _INVOICE_ELEMENT_SCHEMA,
            "description": (
                "The end date for each line item in the invoice's main "
                "table, exactly as written. One entry per table row."
            ),
        },
    },
    "required": [
        "line_item_description", "line_item_rate", "line_item_days",
        "line_item_start_date", "line_item_end_date",
    ],
}

_INVOICE_LABEL_TO_FIELD: dict[str, str] = {
    "Line Item - Description": "line_item_description",
    "Line Item - Rate": "line_item_rate",
    "Line Item - Days": "line_item_days",
    "Line Item - Start Date": "line_item_start_date",
    "Line Item - End Date": "line_item_end_date",
}

REALKIE_DATASETS: dict[str, dict[str, Any]] = {
    "nda": {
        "schema": _NDA_SCHEMA,
        "label_to_field": _NDA_LABEL_TO_FIELD,
    },
    "fcc_invoices": {
        "schema": _INVOICE_SCHEMA,
        "label_to_field": _INVOICE_LABEL_TO_FIELD,
    },
}

# Official split files, in load order. raw_export.csv (the unsplit union) and
# old_split_files/ (a superseded export) are intentionally excluded.
SPLIT_FILES: tuple[str, ...] = ("train.csv", "val.csv", "test.csv")


# ============================================================================
# Span-text normalization
# ============================================================================

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_span_text(raw: Any) -> str:
    """Clean a RealKIE annotation span into a usable gold value.

    RealKIE span boundaries are loose: they frequently include OCR line
    breaks mid-value (e.g. "Alder\\nBioPharmaceuticals, Inc.,") and trailing
    punctuation carried in from the surrounding sentence. Both are annotation
    artifacts, not part of the real value. We collapse all runs of whitespace
    (including newlines) to single spaces and strip trailing boundary
    punctuation.

    Casefolding is deliberately NOT applied here: matching leniency belongs
    in the labeling stage (value_compare), not in the stored gold, which
    should stay human-readable.
    """
    s = _WHITESPACE_RE.sub(" ", str(raw)).strip()
    s = s.rstrip(" .,;:")
    return s.strip()


# ============================================================================
# Loader
# ============================================================================

class RealKIE(Benchmark):
    """Loader for the RealKIE benchmark (NDA dataset).

    Args:
        benchmark_path: Path to the directory that contains the RealKIE
            dataset subdirectories (i.e. the directory holding `nda/`).
        domains: RealKIE dataset names to include (e.g. ["nda"]). Must be
            keys of REALKIE_DATASETS. None / empty = all registered datasets.
        max_documents: Cap on total documents yielded. None = no cap. Useful
            for fast pipeline validation (the --limit flag in 01_extract.py
            flows through to here).
        pdf_backend: Accepted for signature symmetry with ExtractBench so the
            benchmark dispatcher can treat loaders uniformly. RealKIE uses the
            OCR text shipped in the CSV, so this argument is ignored.

    Notes:
        All three split CSVs are read at construction time and merged into a
        single document pool keyed by document hash; duplicates across splits
        (should not occur, but handled defensively) keep their first
        occurrence. The original split is preserved in Document.metadata.

        For leave-one-document-out CV, document identity is what matters, not
        the train/val/test split -- which is why the splits are pooled.
    """

    def __init__(
        self,
        benchmark_path: str | Path,
        *,
        domains: list[str] | None = None,
        max_documents: int | None = None,
        pdf_backend: str | None = None,  # accepted for dispatcher symmetry; unused
    ) -> None:
        self.benchmark_path = Path(benchmark_path)
        self.max_documents = max_documents

        if pdf_backend is not None:
            logger.debug(
                "RealKIE ignores pdf_backend=%r (uses OCR text from the CSV)",
                pdf_backend,
            )

        if not self.benchmark_path.is_dir():
            raise FileNotFoundError(
                f"RealKIE benchmark_path not found: {self.benchmark_path}. "
                f"It should be the directory containing dataset subdirectories "
                f"(e.g. a 'nda/' folder)."
            )

        # Resolve which datasets to load.
        if domains:
            unknown = set(domains) - set(REALKIE_DATASETS)
            if unknown:
                raise ValueError(
                    f"Unknown RealKIE dataset(s) requested: {sorted(unknown)}. "
                    f"Registered datasets: {sorted(REALKIE_DATASETS)}"
                )
            self._selected_domains = list(domains)
        else:
            self._selected_domains = sorted(REALKIE_DATASETS)

        # Cache schemas (small, queried repeatedly by get_schema).
        self._schemas: dict[str, dict[str, Any]] = {
            d: REALKIE_DATASETS[d]["schema"] for d in self._selected_domains
        }

        # Read every split CSV and build the flat document record list.
        self._records: list[dict[str, Any]] = self._load_records()

        if self.max_documents is not None:
            self._records = self._records[: self.max_documents]

        logger.info(
            "RealKIE initialized: %d dataset(s), %d document(s)",
            len(self._selected_domains), len(self._records),
        )

    # ------------------------------------------------------------------------
    # Benchmark interface
    # ------------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "real_kie"

    @property
    def domains(self) -> list[str]:
        return list(self._selected_domains)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[Document]:
        for record in self._records:
            yield self._build_document(record)

    def get_schema(self, domain: str) -> dict[str, Any]:
        if domain not in self._schemas:
            raise KeyError(
                f"Schema for domain {domain!r} not loaded. Available: "
                f"{sorted(self._schemas.keys())}"
            )
        return self._schemas[domain]

    # ------------------------------------------------------------------------
    # Internal: reading CSVs into flat records
    # ------------------------------------------------------------------------

    def _load_records(self) -> list[dict[str, Any]]:
        """Read all split CSVs for every selected dataset into flat records.

        Each record is a plain dict; the heavy DataFrame is dropped once the
        records are extracted. Documents are deduplicated by doc_id across
        splits (first occurrence wins).
        """
        records: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for domain in self._selected_domains:
            dataset_dir = self.benchmark_path / domain
            if not dataset_dir.is_dir():
                raise FileNotFoundError(
                    f"RealKIE dataset directory not found: {dataset_dir}"
                )

            n_before = len(records)
            for split_file in SPLIT_FILES:
                csv_path = dataset_dir / split_file
                if not csv_path.exists():
                    logger.warning("Missing split file, skipping: %s", csv_path)
                    continue

                split = csv_path.stem  # "train" / "val" / "test"
                # Full read (no usecols): the "Unnamed: 0*" cruft columns are
                # harmless and ignored; reading everything is robust to minor
                # column-set differences between splits.
                df = pd.read_csv(csv_path)

                for _, row in df.iterrows():
                    record = self._row_to_record(domain, split, row)
                    if record is None:
                        continue
                    if record["doc_id"] in seen_ids:
                        logger.debug(
                            "Duplicate doc_id %s seen in split %r; keeping "
                            "the first occurrence.", record["doc_id"], split,
                        )
                        continue
                    seen_ids.add(record["doc_id"])
                    records.append(record)

            logger.info(
                "  dataset %r: %d document(s) loaded across splits",
                domain, len(records) - n_before,
            )

        return records

    @staticmethod
    def _row_to_record(
        domain: str,
        split: str,
        row: "pd.Series",
    ) -> dict[str, Any] | None:
        """Turn one CSV row into a flat record dict, or None if unusable.

        The doc_id is "{domain}__{hash}", where {hash} is the stem of
        document_path ("nda/files/<hash>.pdf" -> "<hash>"). The hash is hex,
        so the resulting doc_id is filesystem-safe.
        """
        doc_path = row.get("document_path")
        if not isinstance(doc_path, str) or not doc_path.strip():
            logger.warning(
                "Row in %s/%s has no document_path; skipping.", domain, split,
            )
            return None

        doc_hash = Path(doc_path).stem
        if not doc_hash:
            logger.warning(
                "Row in %s/%s has an empty document hash (%r); skipping.",
                domain, split, doc_path,
            )
            return None

        return {
            "domain": domain,
            "doc_id": f"{domain}__{doc_hash}",
            "split": split,
            "document_path": doc_path,
            "text": row.get("text"),
            "labels": row.get("labels"),
            "original_filename": row.get("original_filename"),
        }

    # ------------------------------------------------------------------------
    # Internal: building Document objects
    # ------------------------------------------------------------------------

    def _build_document(self, record: dict[str, Any]) -> Document:
        """Construct a Document from a flat record.

        Sets extraction_error (and empty text) if the OCR text is missing, so
        downstream extraction stages skip the document but it still appears in
        run summaries -- mirroring how ExtractBench reports PDF failures.
        """
        domain = record["domain"]
        schema = self._schemas[domain]
        label_to_field = REALKIE_DATASETS[domain]["label_to_field"]
        schema_fields = list(schema["properties"].keys())

        # Document text: OCR text straight from the CSV. No PDF parsing.
        text = record["text"]
        extraction_error: str | None = None
        if not isinstance(text, str) or not text.strip():
            text = ""
            extraction_error = "empty OCR text in RealKIE export"
            logger.warning(
                "Document %s has empty OCR text; marking as extraction error.",
                record["doc_id"],
            )

        # Gold: convert the span list into a schema-shaped JSON object.
        gold = self._labels_to_gold(
            record["labels"], label_to_field, schema_fields, record["doc_id"],
        )

        return Document(
            doc_id=record["doc_id"],
            domain=domain,
            text=text,
            schema=schema,
            gold=gold,
            source_path=self.benchmark_path / record["document_path"],
            metadata={
                "benchmark": "real_kie",
                "split": record["split"],
                "original_filename": record.get("original_filename"),
            },
            extraction_error=extraction_error,
        )

    @staticmethod
    def _labels_to_gold(
        labels_raw: Any,
        label_to_field: dict[str, str],
        schema_fields: list[str],
        doc_id: str,
    ) -> dict[str, list[str]]:
        """Convert a RealKIE span list (JSON string) into a gold JSON object.

        Spans are grouped by label, mapped to schema field names, normalized,
        and deduplicated (case-insensitively) within each field. Every schema
        field is present in the result, as an array -- empty if the document
        has no span for that label.

        A document with missing/unparseable labels yields an all-empty gold
        object and a warning, rather than raising: a single bad row should not
        abort a whole extraction run.
        """
        gold: dict[str, list[str]] = {f: [] for f in schema_fields}
        seen: dict[str, set[str]] = {f: set() for f in schema_fields}

        if not isinstance(labels_raw, str) or not labels_raw.strip():
            logger.warning(
                "Document %s has no labels; gold will be empty.", doc_id,
            )
            return gold

        try:
            spans = json.loads(labels_raw)
        except json.JSONDecodeError as e:
            logger.warning(
                "Document %s: could not parse labels JSON (%s); gold empty.",
                doc_id, e,
            )
            return gold

        if not isinstance(spans, list):
            logger.warning(
                "Document %s: labels payload is not a list (%s); gold empty.",
                doc_id, type(spans).__name__,
            )
            return gold

        for span in spans:
            if not isinstance(span, dict):
                continue

            label = span.get("label")
            field = label_to_field.get(label)
            if field is None:
                logger.warning(
                    "Document %s: span label %r is not in the schema; "
                    "skipping this span.", doc_id, label,
                )
                continue

            value = _normalize_span_text(span.get("text", ""))
            if not value:
                continue

            dedupe_key = value.casefold()
            if dedupe_key in seen[field]:
                continue
            seen[field].add(dedupe_key)
            gold[field].append(value)

        return gold
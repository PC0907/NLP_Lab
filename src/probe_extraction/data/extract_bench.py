"""ExtractBench benchmark loader.

ExtractBench (Contextual AI, 2026) provides 35 PDFs across 5 domains, each
paired with a JSON Schema and human-validated gold annotations.

Repository: https://github.com/ContextualAI/extract-bench

Directory layout (relative to benchmark_path):

    dataset/
    ├── {domain_top}/                  # e.g., "hiring", "finance"
    │   └── {domain_sub}/              # e.g., "resume", "10kq"
    │       ├── {domain_sub}-schema.json
    │       └── pdf+gold/
    │           ├── {doc}.pdf
    │           └── {doc}.gold.json

We treat "{domain_top}/{domain_sub}" as a single domain identifier (e.g.,
"hiring/resume", "finance/10kq").

Schema files have the form:
    {
      "name": "...",
      "description": "...",
      "schema_definition": { ...actual JSON Schema... }
    }
The pipeline only uses `schema_definition`; we strip the outer wrapper at
load time so downstream code doesn't have to know about it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

from probe_extraction.data.base import Benchmark, Document
from probe_extraction.data.pdf_utils import (
    PDFExtractionError,
    extract_text,
    get_pdf_metadata,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

SCHEMA_SUFFIX = "-schema.json"
GOLD_SUFFIX = ".gold.json"
PDF_GOLD_DIR = "pdf+gold"
DATASET_SUBDIR = "dataset"


# ============================================================================
# Loader
# ============================================================================

class ExtractBench(Benchmark):
    """Loader for the ExtractBench benchmark.

    Args:
        benchmark_path: Path to the cloned extract-bench repo (the directory
            that contains `dataset/`).
        domains: List of domain identifiers to include (e.g.,
            ["hiring/resume", "finance/10kq"]). Empty list = all discovered
            domains.
        max_documents: Cap on total documents yielded across all domains.
            None = no cap. Useful for fast iteration during development.
        pdf_min_chars: Passed through to extract_text — minimum chars of
            extracted text required to consider extraction successful.

    Notes:
        Domain discovery is filesystem-based: any directory containing a
        `*-schema.json` file and a `pdf+gold/` subdirectory is treated as
        a domain. This means the loader keeps working if ExtractBench adds
        new domains in future releases.

        Iteration is lazy: documents are loaded one at a time. PDF text
        extraction happens during iteration, not during construction, so
        constructing the loader is cheap.
    """

    def __init__(
        self,
        benchmark_path: str | Path,
        *,
        domains: list[str] | None = None,
        max_documents: int | None = None,
        pdf_min_chars: int = 50,
    ) -> None:
        self.benchmark_path = Path(benchmark_path)
        self.dataset_root = self.benchmark_path / DATASET_SUBDIR
        self.max_documents = max_documents
        self.pdf_min_chars = pdf_min_chars

        if not self.dataset_root.exists():
            raise FileNotFoundError(
                f"ExtractBench dataset directory not found: {self.dataset_root}. "
                f"Did you clone https://github.com/ContextualAI/extract-bench "
                f"into {self.benchmark_path}?"
            )

        # Discover all available domains, then filter to requested ones.
        all_domains = self._discover_domains()
        if not all_domains:
            raise FileNotFoundError(
                f"No domains found under {self.dataset_root}. Expected to find "
                f"directories with `*-schema.json` files and `pdf+gold/` subdirs."
            )

        if domains:
            unknown = set(domains) - set(all_domains)
            if unknown:
                raise ValueError(
                    f"Unknown domains requested: {sorted(unknown)}. "
                    f"Available: {sorted(all_domains)}"
                )
            self._selected_domains = list(domains)
        else:
            self._selected_domains = sorted(all_domains)

        # Cache schemas (small, used repeatedly by get_schema)
        self._schemas: dict[str, dict[str, Any]] = {
            d: self._load_schema(d) for d in self._selected_domains
        }

        # Pre-compute the list of (domain, pdf_path, gold_path) tuples.
        # Cheap (just a directory listing per domain), and lets us implement
        # __len__ in O(1).
        self._docs: list[tuple[str, Path, Path]] = self._discover_documents()

        if self.max_documents is not None:
            self._docs = self._docs[: self.max_documents]

        logger.info(
            "ExtractBench initialized: %d domain(s), %d document(s)",
            len(self._selected_domains), len(self._docs),
        )

    # ------------------------------------------------------------------------
    # Benchmark interface
    # ------------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "extract_bench"

    @property
    def domains(self) -> list[str]:
        return list(self._selected_domains)

    def __len__(self) -> int:
        return len(self._docs)

    def __iter__(self) -> Iterator[Document]:
        for domain, pdf_path, gold_path in self._docs:
            yield self._build_document(domain, pdf_path, gold_path)

    def get_schema(self, domain: str) -> dict[str, Any]:
        if domain not in self._schemas:
            raise KeyError(
                f"Schema for domain {domain!r} not loaded. Available: "
                f"{sorted(self._schemas.keys())}"
            )
        return self._schemas[domain]

    # ------------------------------------------------------------------------
    # Internal: discovery
    # ------------------------------------------------------------------------

    def _discover_domains(self) -> list[str]:
        """Find all valid domain directories under dataset_root.

        A directory is a domain if it contains exactly one `*-schema.json`
        file and a `pdf+gold/` subdirectory.

        Returns domain identifiers like "hiring/resume" — the path relative
        to dataset_root.
        """
        domains: list[str] = []
        # Walk two levels deep: dataset/{domain_top}/{domain_sub}/
        for top_dir in sorted(self.dataset_root.iterdir()):
            if not top_dir.is_dir():
                continue
            for sub_dir in sorted(top_dir.iterdir()):
                if not sub_dir.is_dir():
                    continue
                if self._is_valid_domain_dir(sub_dir):
                    domain = f"{top_dir.name}/{sub_dir.name}"
                    domains.append(domain)
                else:
                    logger.debug(
                        "Skipping %s: missing schema or pdf+gold directory",
                        sub_dir,
                    )
        return domains

    @staticmethod
    def _is_valid_domain_dir(path: Path) -> bool:
        """A domain dir has a *-schema.json file and a pdf+gold/ subdir."""
        has_schema = any(path.glob(f"*{SCHEMA_SUFFIX}"))
        has_pdf_gold = (path / PDF_GOLD_DIR).is_dir()
        return has_schema and has_pdf_gold

    def _domain_dir(self, domain: str) -> Path:
        """Resolve a domain identifier to its directory path."""
        return self.dataset_root / domain

    def _load_schema(self, domain: str) -> dict[str, Any]:
        """Load the schema for a given domain.

        ExtractBench schemas come in two formats:
          1. Wrapped: {"name": ..., "schema_definition": {...actual schema...}}
             (e.g., hiring/resume)
          2. Plain: {...JSON Schema directly...}
             (e.g., academic/research)

        We detect the format and return the inner JSON Schema in both cases.
        """
        domain_dir = self._domain_dir(domain)
        schema_files = list(domain_dir.glob(f"*{SCHEMA_SUFFIX}"))

        if not schema_files:
            raise FileNotFoundError(
                f"No schema file (*{SCHEMA_SUFFIX}) found in {domain_dir}"
            )
        if len(schema_files) > 1:
            raise ValueError(
                f"Multiple schema files in {domain_dir}: {schema_files}. "
                f"Expected exactly one."
            )

        with schema_files[0].open("r", encoding="utf-8") as f:
            raw = json.load(f)

        # Format detection:
        # - Wrapped form has 'schema_definition' as a top-level key.
        # - Plain form has JSON Schema keys (type/properties) at top level.
        if "schema_definition" in raw:
            schema = raw["schema_definition"]
        elif "type" in raw or "properties" in raw:
            schema = raw
        else:
            raise ValueError(
                f"Schema file {schema_files[0]} has unrecognized format. "
                f"Expected either a 'schema_definition' wrapper or a plain "
                f"JSON Schema with 'type'/'properties' at top level. "
                f"Got top-level keys: {list(raw.keys())}"
            )

        if not isinstance(schema, dict):
            raise ValueError(
                f"Schema in {schema_files[0]} is not a dict: {type(schema)}"
            )

        return schema
    def _discover_documents(self) -> list[tuple[str, Path, Path]]:
        """Find all (domain, pdf_path, gold_path) triples across selected domains.

        A document is included only if BOTH the PDF and the gold file exist.
        Lone golds (no PDF) and lone PDFs (no gold) are logged as warnings
        and skipped.
        """
        docs: list[tuple[str, Path, Path]] = []
        for domain in self._selected_domains:
            pdf_gold_dir = self._domain_dir(domain) / PDF_GOLD_DIR
            if not pdf_gold_dir.is_dir():
                logger.warning("Missing pdf+gold dir for domain %s", domain)
                continue

            # Index gold files by their stem (filename without .gold.json)
            golds: dict[str, Path] = {}
            for p in pdf_gold_dir.glob(f"*{GOLD_SUFFIX}"):
                stem = p.name[: -len(GOLD_SUFFIX)]
                golds[stem] = p

            # Index PDFs by their stem (filename without .pdf)
            pdfs: dict[str, Path] = {}
            for p in pdf_gold_dir.glob("*.pdf"):
                pdfs[p.stem] = p

            paired_stems = sorted(golds.keys() & pdfs.keys())
            unpaired_golds = sorted(golds.keys() - pdfs.keys())
            unpaired_pdfs = sorted(pdfs.keys() - golds.keys())

            if unpaired_golds:
                logger.warning(
                    "Domain %s: %d gold files without PDF: %s",
                    domain, len(unpaired_golds), unpaired_golds,
                )
            if unpaired_pdfs:
                logger.warning(
                    "Domain %s: %d PDFs without gold: %s",
                    domain, len(unpaired_pdfs), unpaired_pdfs,
                )

            for stem in paired_stems:
                docs.append((domain, pdfs[stem], golds[stem]))

        return docs

    # ------------------------------------------------------------------------
    # Internal: building Document objects
    # ------------------------------------------------------------------------

    def _build_document(
        self,
        domain: str,
        pdf_path: Path,
        gold_path: Path,
    ) -> Document:
        """Construct a Document, extracting PDF text and loading gold JSON.

        On extraction failure, returns a Document with empty text and
        `extraction_error` set, so the caller can decide whether to skip.
        """
        doc_id = self._make_doc_id(domain, pdf_path)
        schema = self._schemas[domain]

        # Load gold (must succeed; gold files are part of the benchmark spec)
        with gold_path.open("r", encoding="utf-8") as f:
            gold = json.load(f)

        # Extract PDF text (may fail; that's OK, we report it)
        text = ""
        extraction_error: str | None = None
        try:
            text = extract_text(pdf_path, min_chars=self.pdf_min_chars)
        except (PDFExtractionError, FileNotFoundError) as e:
            extraction_error = str(e)
            logger.warning(
                "PDF extraction failed for %s: %s", pdf_path.name, e,
            )

        # Best-effort metadata; never fail the document over this
        try:
            metadata = get_pdf_metadata(pdf_path)
        except Exception as e:
            logger.debug("Metadata fetch failed for %s: %s", pdf_path.name, e)
            metadata = {}

        return Document(
            doc_id=doc_id,
            domain=domain,
            text=text,
            schema=schema,
            gold=gold,
            source_path=pdf_path,
            metadata=metadata,
            extraction_error=extraction_error,
        )

    @staticmethod
    def _make_doc_id(domain: str, pdf_path: Path) -> str:
        """Build a filesystem-safe doc_id.

        Format: "{domain_with_underscores}__{filename_stem_with_underscores}"
        Example:
            domain="hiring/resume", pdf="Resume-IT.pdf"
            -> "hiring__resume__Resume-IT"

        Spaces in filenames (some ExtractBench filenames have them) are
        replaced with underscores.
        """
        domain_part = domain.replace("/", "__")
        # Replace any filesystem-unfriendly characters in the stem
        stem = pdf_path.stem
        for bad in [" ", "/", "\\", ":", "*", "?", '"', "<", ">", "|"]:
            stem = stem.replace(bad, "_")
        return f"{domain_part}__{stem}"
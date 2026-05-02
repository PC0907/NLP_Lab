"""Abstract benchmark interface.

A Benchmark is anything that yields (document_text, schema, gold) tuples.
Concrete implementations (e.g., ExtractBench) handle the specifics of where
files live and how they're parsed; downstream code only sees Documents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


# ============================================================================
# Document: the unit of work in our pipeline
# ============================================================================

@dataclass
class Document:
    """A single benchmark document with its schema and gold annotations.

    Attributes:
        doc_id: Stable, filesystem-safe identifier (used as filename for
            saved artifacts). Example: "hiring__resume__Resume-IT".
        domain: Top-level benchmark category, e.g. "hiring/resume".
        text: Extracted text content of the source document. May be empty if
            extraction failed; loaders should set extraction_error in that case.
        schema: JSON Schema (as a dict) defining the extraction target.
        gold: Human-validated extraction output (as a dict) conforming to schema.
        source_path: Path to the original file (PDF, etc.) for traceability.
        metadata: Free-form dict for loader-specific info (page count, OCR
            confidence, etc.). Not used by the pipeline; available for analysis.
        extraction_error: If text extraction failed, the error message; else None.
            Documents with extraction errors should be skipped by extraction
            stages but kept in the document list so we can report on them.
    """

    doc_id: str
    domain: str
    text: str
    schema: dict[str, Any]
    gold: dict[str, Any]
    source_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    extraction_error: str | None = None

    def __post_init__(self) -> None:
        # doc_id must be filesystem-safe — we use it directly as a filename.
        if not self.doc_id:
            raise ValueError("Document.doc_id cannot be empty")
        bad_chars = set('/\\:*?"<>|')
        if any(c in bad_chars for c in self.doc_id):
            raise ValueError(
                f"Document.doc_id contains filesystem-unsafe characters: {self.doc_id!r}"
            )


# ============================================================================
# Benchmark: abstract base class
# ============================================================================

class Benchmark(ABC):
    """Abstract benchmark loader.

    Subclasses implement the specifics of locating files and parsing them.
    The pipeline interacts only with the abstract interface below.

    Implementations should be lazy where possible (yielding documents one at
    a time rather than loading everything into memory upfront), so we can
    iterate over large benchmarks without OOM.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[Document]:
        """Yield documents one at a time.

        Implementations should respect any filtering (domain selection, max
        documents) configured at construction time.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Total number of documents the iterator will yield.

        Used for progress bars; should be O(1) — don't iterate to count.
        """
        ...

    @abstractmethod
    def get_schema(self, domain: str) -> dict[str, Any]:
        """Return the JSON Schema for a given domain.

        Useful for stages that don't need to iterate documents but need
        the schema (e.g., schema-aware prompt construction).
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this benchmark (e.g., 'extract_bench').

        Used in artifact paths and logging.
        """
        ...

    @property
    @abstractmethod
    def domains(self) -> list[str]:
        """List of domain identifiers covered by this benchmark instance.

        Should reflect any filtering applied at construction time.
        """
        ...
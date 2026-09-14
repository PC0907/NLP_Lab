"""PDF text extraction.

Three backends, a disk cache, and clear error handling. Extracting text from a
PDF is one job, not a framework.

If the PDF is image-only (scanned without OCR) or otherwise yields no text,
this module raises rather than silently succeeding with empty content.

BACKENDS
--------
  pymupdf   Reads the embedded text layer. Fast, no models. Layout-naive: on
            ruled tables it emits one cell per line with no row delimiters, so
            binding a value to its row depends entirely on sequence position.
  docling   Layout- and table-aware; runs neural layout/table-structure models.
            Emits Markdown, preserving rows as table syntax -- but its column
            boundaries can be wrong (observed merging "Team" and "Time" into a
            single cell).
  camelot   Reads ruled/stream tables directly from the PDF vector content. No
            model, no GPU, ~0.8s/doc. Separates columns correctly on
            born-digital tables. NOTE: extracts TABLES ONLY -- body prose is
            dropped, so it is unsuitable for text-heavy domains (academic
            papers, credit agreements) and is meaningful only where the
            document IS a table.

CACHE
-----
Parsing is deterministic (verified: 3 repeats per document per backend give
identical hashes), so results are cached to disk at

    data/parsed/<backend>/<cache_key>.txt

When a cache entry exists it is read instead of re-parsing. This removes
parsing from the critical path of every extraction run -- docling takes
30-56s per financial document -- and, more importantly, makes the text a model
saw a durable artifact rather than something re-derived each time. Set the
environment variable PDF_CACHE_DISABLE=1 to bypass it.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Docling is heavy (downloads neural models on first use) — import lazily and
# reuse the converter across calls.
_docling_converter = None

CACHE_ROOT = Path("data/parsed")


# ============================================================================
# Custom exception
# ============================================================================

class PDFExtractionError(Exception):
    """Raised when PDF text extraction fails or produces no usable text."""


# ============================================================================
# Cache helpers
# ============================================================================

def _cache_path(backend: str, cache_key: str) -> Path:
    return CACHE_ROOT / backend / f"{cache_key}.txt"


def _cache_read(backend: str, cache_key: str | None) -> str | None:
    if cache_key is None or os.environ.get("PDF_CACHE_DISABLE"):
        return None
    p = _cache_path(backend, cache_key)
    if not p.exists():
        return None
    try:
        text = p.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("Cache read failed for %s: %s", p, e)
        return None
    logger.debug("Cache hit: %s (%d chars)", p, len(text))
    return text


def _cache_write(backend: str, cache_key: str | None, text: str) -> None:
    if cache_key is None or os.environ.get("PDF_CACHE_DISABLE"):
        return
    p = _cache_path(backend, cache_key)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    except Exception as e:
        # A cache write failure must never fail a run.
        logger.warning("Cache write failed for %s: %s", p, e)


# ============================================================================
# Main entry point
# ============================================================================

def extract_text(
    pdf_path: str | Path,
    *,
    page_separator: str = "\n\n",
    min_chars: int = 50,
    backend: str = "pymupdf",
    cache_key: str | None = None,
) -> str:
    """Extract text from a PDF file.

    Args:
        pdf_path: Path to the PDF file.
        page_separator: String inserted between pages (PyMuPDF only; Docling
            and Camelot produce a single stream).
        min_chars: Minimum characters of extracted text required.
        backend: "pymupdf", "docling", or "camelot". See module docstring for
            what each does and where each fails.
        cache_key: Identifier for the disk cache, normally the doc_id. When
            given, a cached parse is reused if present and a fresh parse is
            written on success. When None, caching is skipped entirely.

    Returns:
        The full extracted text.

    Raises:
        FileNotFoundError: If the PDF path doesn't exist.
        PDFExtractionError: If extraction fails or yields too little text.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    cached = _cache_read(backend, cache_key)
    if cached is not None:
        if len(cached) < min_chars:
            raise PDFExtractionError(
                f"Cached parse of {pdf_path.name} has only {len(cached)} chars "
                f"(min: {min_chars}, backend: {backend})."
            )
        return cached

    if backend == "pymupdf":
        text = _extract_text_pymupdf(pdf_path, page_separator)
    elif backend == "docling":
        text = _extract_text_docling(pdf_path)
    elif backend == "camelot":
        text = _extract_text_camelot(pdf_path)
    else:
        raise ValueError(f"Unknown PDF extraction backend: {backend!r}")

    if len(text) < min_chars:
        raise PDFExtractionError(
            f"PDF {pdf_path.name} yielded only {len(text)} chars of text "
            f"(min: {min_chars}, backend: {backend}). "
            f"Possibly image-only or corrupt."
        )

    _cache_write(backend, cache_key, text)
    return text


def _extract_text_pymupdf(pdf_path: Path, page_separator: str) -> str:
    """PyMuPDF backend: fast, plain-text per page, joined with separator."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise PDFExtractionError(f"Could not open {pdf_path}: {e}") from e

    try:
        pages: list[str] = []
        for page_num, page in enumerate(doc):
            try:
                page_text = page.get_text("text")
            except Exception as e:
                logger.warning(
                    "Failed to extract page %d of %s: %s",
                    page_num, pdf_path.name, e,
                )
                page_text = ""
            pages.append(page_text)
        return page_separator.join(pages).strip()
    finally:
        doc.close()


def _extract_text_docling(pdf_path: Path) -> str:
    """Docling backend: layout-aware, exports to Markdown.

    Lazy-loads the DocumentConverter on first use. First call downloads
    neural layout/table-structure models (~hundreds of MB), so it's slow.
    Subsequent calls reuse the loaded converter.
    """
    global _docling_converter
    if _docling_converter is None:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as e:
            raise PDFExtractionError(
                "Docling not installed. Add `docling` to requirements.txt "
                "or `pip install docling`."
            ) from e
        logger.info("Initializing Docling converter (downloads models on first use)...")
        _docling_converter = DocumentConverter()

    try:
        result = _docling_converter.convert(str(pdf_path))
        return result.document.export_to_markdown().strip()
    except Exception as e:
        raise PDFExtractionError(
            f"Docling failed on {pdf_path.name}: {e}"
        ) from e


def _extract_text_camelot(pdf_path: Path) -> str:
    """Camelot backend: ruled/stream table extraction from vector content.

    TABLES ONLY. Body prose is not returned, so this backend is appropriate
    only where the document IS a table (e.g. results tables, some filings).
    On a text-heavy document it will either return very little or raise.

    Tries `lattice` first (reads ruling lines; needs Ghostscript), falling back
    to `stream` (infers columns from whitespace). Rows are serialised
    pipe-delimited, which preserves the row grouping that flat text extraction
    destroys.
    """
    try:
        import camelot
    except ImportError as e:
        raise PDFExtractionError(
            "Camelot not installed. `pip install \"camelot-py[cv]\"` "
            "(lattice mode also needs Ghostscript)."
        ) from e

    chunks: list[str] = []
    used_flavor = None
    for flavor in ("lattice", "stream"):
        try:
            tables = camelot.read_pdf(str(pdf_path), pages="all", flavor=flavor)
        except Exception as e:
            logger.debug("camelot %s failed on %s: %s", flavor, pdf_path.name, e)
            continue
        if len(tables) == 0:
            continue
        used_flavor = flavor
        chunks.append(f"[camelot: {len(tables)} tables, flavor={flavor}]")
        for i, t in enumerate(tables):
            chunks.append(f"\n--- table {i} (page {t.page}) ---")
            for _, row in t.df.iterrows():
                cells = [str(c).replace("\n", " ").strip() for c in row.tolist()]
                chunks.append(" | ".join(cells))
        break  # first flavor that finds anything wins

    if not chunks:
        raise PDFExtractionError(
            f"Camelot found no tables in {pdf_path.name} with either flavor. "
            f"Expected for text-heavy documents -- camelot extracts tables only."
        )
    logger.debug("camelot used flavor=%s on %s", used_flavor, pdf_path.name)
    return "\n".join(chunks)


# ============================================================================
# Metadata helper (used by loaders to populate Document.metadata)
# ============================================================================

def get_pdf_metadata(pdf_path: str | Path) -> dict:
    """Get basic metadata about a PDF without extracting full text.

    Cheap to call. Returns page count and any embedded metadata. Useful for
    populating Document.metadata or for filtering (e.g., skip 200-page PDFs
    if you only want short docs).

    Returns:
        Dict with keys: page_count, title, author, has_text (bool indicating
        whether at least one page has extractable text).
    """
    pdf_path = Path(pdf_path)
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return {"error": str(e)}

    try:
        page_count = doc.page_count
        meta = doc.metadata or {}

        # Cheap "is there any text?" check — sample first 3 pages
        has_text = False
        for page in doc[: min(3, page_count)]:
            if page.get_text("text").strip():
                has_text = True
                break

        return {
            "page_count": page_count,
            "title": meta.get("title", "") or "",
            "author": meta.get("author", "") or "",
            "has_text": has_text,
        }
    finally:
        doc.close()
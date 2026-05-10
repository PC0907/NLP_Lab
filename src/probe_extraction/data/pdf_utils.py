"""PDF text extraction.

Wraps PyMuPDF (fitz) into a simple text-extraction function with sensible
defaults and clear error handling. Kept deliberately minimal: extracting
text from a PDF is one job, not a framework.

If the PDF is image-only (scanned without OCR) or otherwise yields no text,
this module returns an empty string AND raises a flag the caller can check —
it does not silently succeed with empty content.
"""

from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Docling is heavy (downloads neural models on first use) — import lazily.
_docling_converter = None

# ============================================================================
# Custom exception
# ============================================================================

class PDFExtractionError(Exception):
    """Raised when PDF text extraction fails or produces no usable text."""


# ============================================================================
# Main entry point
# ============================================================================

def extract_text(
    pdf_path: str | Path,
    *,
    page_separator: str = "\n\n",
    min_chars: int = 50,
    backend: str = "pymupdf",
) -> str:
    """Extract text from a PDF file.

    Args:
        pdf_path: Path to the PDF file.
        page_separator: String inserted between pages (PyMuPDF only;
            Docling produces a single Markdown stream).
        min_chars: Minimum characters of extracted text required.
        backend: "pymupdf" (fast, layout-naive) or "docling" (slower,
            layout-aware, better for tables and complex documents).

    Returns:
        The full extracted text.

    Raises:
        FileNotFoundError: If the PDF path doesn't exist.
        PDFExtractionError: If extraction fails or yields too little text.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if backend == "pymupdf":
        text = _extract_text_pymupdf(pdf_path, page_separator)
    elif backend == "docling":
        text = _extract_text_docling(pdf_path)
    else:
        raise ValueError(f"Unknown PDF extraction backend: {backend!r}")

    if len(text) < min_chars:
        raise PDFExtractionError(
            f"PDF {pdf_path.name} yielded only {len(text)} chars of text "
            f"(min: {min_chars}, backend: {backend}). "
            f"Possibly image-only or corrupt."
        )

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
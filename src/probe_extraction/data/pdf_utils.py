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
) -> str:
    """Extract text from a PDF file.

    Args:
        pdf_path: Path to the PDF file.
        page_separator: String inserted between pages. Default is a blank
            line, which preserves rough page boundaries without being noisy.
        min_chars: If the extracted text has fewer characters than this,
            raise PDFExtractionError. PDFs that are image-only or corrupt
            often "succeed" with a few stray characters; this catches them.

    Returns:
        The full extracted text, with pages joined by `page_separator`.

    Raises:
        FileNotFoundError: If the PDF path doesn't exist.
        PDFExtractionError: If the PDF can't be opened, or the extracted
            text is shorter than `min_chars`.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise PDFExtractionError(f"Could not open {pdf_path}: {e}") from e

    try:
        pages: list[str] = []
        for page_num, page in enumerate(doc):
            try:
                page_text = page.get_text("text")  # plain-text mode
            except Exception as e:
                logger.warning(
                    "Failed to extract page %d of %s: %s",
                    page_num, pdf_path.name, e,
                )
                page_text = ""
            pages.append(page_text)

        full_text = page_separator.join(pages).strip()
    finally:
        doc.close()

    if len(full_text) < min_chars:
        raise PDFExtractionError(
            f"PDF {pdf_path.name} yielded only {len(full_text)} chars of text "
            f"(min: {min_chars}). Possibly image-only or corrupt."
        )

    return full_text


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
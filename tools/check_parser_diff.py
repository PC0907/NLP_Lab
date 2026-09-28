"""Verify Docling and PyMuPDF actually produced DIFFERENT extracted text for the
same document. If identical, the Docling integration isn't really running."""
import sys
sys.path.insert(0, "src")
from pathlib import Path
from probe_extraction.data.pdf_utils import extract_text  # adjust if name differs
import glob

# pick one PDF that exists in the benchmark
pdfs = glob.glob("data/extract-bench/**/*.pdf", recursive=True)
pdf = pdfs[0]
print("PDF:", pdf)
print("=" * 70)

# extract with both backends directly
try:
    t_py = extract_text(Path(pdf), backend="pymupdf")
except Exception as e:
    t_py = f"<pymupdf failed: {e}>"
try:
    t_dl = extract_text(Path(pdf), backend="docling")
except Exception as e:
    t_dl = f"<docling failed: {e}>"

print(f"PyMuPDF length: {len(t_py)} chars")
print(f"Docling length: {len(t_dl)} chars")
print(f"Identical?      {t_py == t_dl}")
print(f"Length ratio:   {len(t_dl)/max(len(t_py),1):.2f}")
print("=" * 70)
print("PyMuPDF first 400 chars:")
print(repr(t_py[:400]))
print("-" * 70)
print("Docling first 400 chars:")
print(repr(t_dl[:400]))
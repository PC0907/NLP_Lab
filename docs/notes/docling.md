# Docling vs PyMuPDF — parsing-quality investigation

_Probe-based trust signals · ExtractBench · Qwen3.5-4B · 16 June 2026_

## Question
Does a layout/table-aware PDF parser (Docling) recover more gold values from the
documents than the simple parser (PyMuPDF)? If so, it raises the ceiling on
extraction accuracy. Metric: **gold-coverage** = fraction of gold annotation
values that appear in the extracted text (`08b_gold_coverage.py`).

## Setup / fixes needed to run Docling at all
The prior Docling attempt covered only 7 docs because of environment errors:
- `ImportError: libGL.so.1` (OpenCV dependency on a headless node) — fixed with
  `pip install opencv-python-headless`.
- OCR engine auto-installed (RapidOCR).
After this, Docling ran cleanly on A100 across all four domains (academic 4/6,
10kq 7/7, credit 10/10, swimming 5/5 successful; 2 academic docs failed at the
*LLM JSON* stage, not Docling).

## Two measurement artifacts found and fixed (the comparison was unfair to Docling)
Initial results showed Docling *equal or worse*, which was suspicious given its
structured markdown output. Direct text diff confirmed Docling DOES produce
different (markdown, structured) text — so it was running. The artifacts were in
the **matcher**, not the parser:

1. **Matcher did not handle Docling's markdown/entity formatting.** Docling emits
   HTML entities (`AT&amp;T` vs gold `AT&T`), soft-hyphen line breaks
   (`hand\xad written` vs `handwritten`), unicode dashes, and irregular spacing
   (`J. S.  Denker`). The old `_normalize` missed these → false "missing".
   Fixed `_normalize`: `html.unescape`, join soft-hyphen line breaks, unify
   dashes, collapse all whitespace. (Helped academic +4pts; negligible on
   financial/swimming domains, which have few such artifacts.)
2. **Empty-gold inflation.** Many fields have empty/null gold (academic 60/165,
   swimming ~79, 10kq ~269). These count as "not in text" and deflate the rate.
   Fixed: report the rate on **non-empty fields only** (the fair parsing rate).

## Fair results (non-empty fields, improved matcher)

| Domain   | PyMuPDF | Docling | Δ     | Matched field set? |
|----------|---------|---------|-------|--------------------|
| swimming | 97.8%   | 92.7%   | −5.1  | yes (508 = 508)    |
| credit   | 57.1%   | 57.9%   | +0.8  | yes (126 = 126)    |
| 10kq     | 62.9%   | 62.9%   |  0.0  | yes (≈8837)        |
| academic | 85.0%   | 81.9%   |  —    | NO (80 vs 105)*    |

*academic is not a fair head-to-head: Docling's doc set differs (2 parse
failures + different selection), so the field counts differ. Exclude from the
comparison or match doc sets.

## Conclusion
**Docling does not improve gold-coverage.** On matched field sets it is equal
(credit, 10kq) or slightly worse (swimming). Crucially, the matched-domain
numbers did not move after fixing the artifacts — the original "≈ equal" on
credit/10kq was already correct; only academic had been distorted.

**Why a better parser doesn't help:** the ~37–43% of financial gold values that
are "missing" are absent from *both* parsers' text — i.e. they are not present in
the document as literal strings (derived values, paraphrased annotations,
metadata). No PDF parser can recover values that are not literally there. The
parsing-quality ceiling is set by the **benchmark's gold annotations**, not the
extractor.

**Decision:** retain PyMuPDF — equivalent coverage at ~10× faster extraction.

## Caveats / open
- Gold-coverage measures **presence**, not **extractability**. Docling's
  structural benefit (if any) would show as a lower model **error rate** on
  cleaner text, not as higher coverage. A direct error-rate comparison
  (PyMuPDF vs Docling labels) is the remaining check to fully rule Docling in/out.
- academic head-to-head needs matched doc sets to be fair.
- The improved `_normalize` is now used for ALL gold-in-text checks (fixability,
  coverage), so prior fixability numbers should be re-confirmed with it.

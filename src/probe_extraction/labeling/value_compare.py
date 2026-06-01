"""Primitive value comparisons.

The smallest piece of the labeling pipeline: given two values (one from
gold, one from the model), decide whether they're the same. The matcher
calls into these for every leaf field.

Why a separate module:
  - These functions are pure and easy to unit-test.
  - The matching strategy (strict / fuzzy / semantic / auto) is per-call, not
    global — the matcher decides based on the schema's evaluation_config
    or its own defaults.
  - Adding new comparators (e.g., LLM-judge for semantic strings) means
    extending this module, not touching the matcher.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================================
# Strategies
# ============================================================================

class ComparisonStrategy(str, Enum):
    """How to compare two values.

    String values can use any of: EXACT, CASE_INSENSITIVE, FUZZY.
    Numeric values: NUMBER (with tolerance) or EXACT.
    Booleans / nulls: EXACT.
    URLs / emails: get a normalization pass before comparison.
    AUTO: type-aware default (numeric -> date -> case-insensitive string).
    """

    EXACT = "exact"
    CASE_INSENSITIVE = "case_insensitive"
    FUZZY = "fuzzy"
    NUMBER = "number"
    URL = "url"
    EMAIL = "email"
    DATE = "date"
    AUTO = "auto"


# ============================================================================
# Public entry point
# ============================================================================

def compare_values(
    gold: Any,
    extracted: Any,
    *,
    strategy: ComparisonStrategy = ComparisonStrategy.EXACT,
    fuzzy_threshold: float = 0.85,
    number_tolerance: float = 0.01,
) -> bool:
    """Return True if `extracted` matches `gold` under `strategy`.

    Both arguments are primitive JSON values (str / int / float / bool /
    None). Containers (dict / list) are handled by the matcher, not here.

    None handling:
      - both None → True (match)
      - one None, other not → False (mismatch)
      - empty string is treated as None for matching purposes
    """
    # Normalize empty strings to None so they're handled like nulls.
    g = None if gold == "" else gold
    e = None if extracted == "" else extracted

    # Symmetric None handling.
    if g is None and e is None:
        return True
    if g is None or e is None:
        return False

    # Dispatch on strategy.
    if strategy == ComparisonStrategy.EXACT:
        return _compare_exact(g, e)
    if strategy == ComparisonStrategy.CASE_INSENSITIVE:
        return _compare_case_insensitive(g, e)
    if strategy == ComparisonStrategy.FUZZY:
        return _compare_fuzzy(g, e, threshold=fuzzy_threshold)
    if strategy == ComparisonStrategy.NUMBER:
        return _compare_number(g, e, tolerance=number_tolerance)
    if strategy == ComparisonStrategy.URL:
        return _compare_url(g, e)
    if strategy == ComparisonStrategy.EMAIL:
        return _compare_email(g, e)
    if strategy == ComparisonStrategy.DATE:
        return _compare_date(g, e)
    if strategy == ComparisonStrategy.AUTO:
        return _compare_auto(
            g, e, fuzzy_threshold=fuzzy_threshold, number_tolerance=number_tolerance
        )

    raise ValueError(f"Unsupported comparison strategy: {strategy!r}")


# ============================================================================
# Strategy implementations
# ============================================================================

def _compare_auto(g: Any, e: Any, *, fuzzy_threshold: float, number_tolerance: float) -> bool:
    """Type-aware comparison for fields with no explicit evaluation_config.

    Cascade:
      1. If both look numeric -> numeric tolerance (handles 49.7 == 49.70,
         strips $ and commas).
      2. Else if both look like real dates (and neither is a fiscal-period
         string) -> parse and compare as dates.
      3. Else -> case-insensitive string compare.

    Deliberately does NOT use fuzzy for the string fallback, to avoid
    suppressing real errors that share tokens (e.g. 'FY2025 Q2' vs
    'FY2025 Q1'), and deliberately guards date parsing against fiscal-period
    strings for the same reason.
    """
    # 1. Numeric first.
    if _looks_numeric(g) and _looks_numeric(e):
        return _compare_number(g, e, tolerance=number_tolerance)
    # 2. Date — but never date-parse fiscal-period strings.
    if not _looks_like_period(g) and not _looks_like_period(e):
        gd = _try_parse_date_flexible(str(g))
        ed = _try_parse_date_flexible(str(e))
        if gd is not None and ed is not None:
            return gd == ed
    # 3. Fallback: case-insensitive string.
    return _compare_case_insensitive(g, e)


def _looks_numeric(v: Any) -> bool:
    if isinstance(v, bool):
        return False  # don't treat True/False as 1/0
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, str):
        s = v.strip().replace(",", "").replace("$", "")
        try:
            float(s)
            return True
        except ValueError:
            return False
    return False


def _compare_exact(g: Any, e: Any) -> bool:
    """Strict equality with type coercion for numeric/boolean strings."""
    if type(g) is type(e):
        return g == e

    g_str, e_str = str(g), str(e)
    if g_str == e_str:
        return True

    if isinstance(g, bool) or isinstance(e, bool):
        return g_str.lower() == e_str.lower()

    return False


def _compare_case_insensitive(g: Any, e: Any) -> bool:
    return str(g).strip().lower() == str(e).strip().lower()


def _compare_fuzzy(g: Any, e: Any, *, threshold: float) -> bool:
    """Token-Jaccard similarity above threshold counts as a match."""
    g_tokens = _tokenize_for_fuzzy(str(g))
    e_tokens = _tokenize_for_fuzzy(str(e))
    if not g_tokens or not e_tokens:
        return g_tokens == e_tokens
    intersection = g_tokens & e_tokens
    union = g_tokens | e_tokens
    similarity = len(intersection) / len(union)
    return similarity >= threshold


def _compare_number(g: Any, e: Any, *, tolerance: float) -> bool:
    """Numeric comparison with relative tolerance.

    Coerces strings to numbers if possible (stripping commas and $ first).
    Returns False if either side can't be parsed as a number.
    """
    g_n = _to_number(g)
    e_n = _to_number(e)
    if g_n is None or e_n is None:
        return False

    # Exact match (handles 49.7 == 49.70, 100 == 100.0).
    if g_n == e_n:
        return True

    # Relative tolerance for non-zero values.
    if g_n == 0:
        return abs(e_n) <= tolerance
    return abs(g_n - e_n) / abs(g_n) <= tolerance


def _to_number(v: Any) -> float | None:
    """Parse a value to float, stripping commas and currency symbols."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().replace(",", "").replace("$", "")
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _compare_url(g: Any, e: Any) -> bool:
    return normalize_url(str(g)) == normalize_url(str(e))


def _compare_email(g: Any, e: Any) -> bool:
    return normalize_email(str(g)) == normalize_email(str(e))


def _compare_date(g: Any, e: Any) -> bool:
    """Date comparison with tolerant parsing.

    If both parse as dates, compares as dates. Otherwise falls back to
    case-insensitive string compare of the original values.
    """
    g_date = _try_parse_date_flexible(str(g))
    e_date = _try_parse_date_flexible(str(e))
    if g_date is not None and e_date is not None:
        return g_date == e_date
    return _compare_case_insensitive(g, e)


# ============================================================================
# Normalization helpers (exported)
# ============================================================================

_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")


def normalize_email(s: str) -> str:
    s = s.strip()
    m = _MD_LINK_RE.match(s)
    if m:
        s = m.group(1)
    if s.lower().startswith("mailto:"):
        s = s[len("mailto:") :]
    return s.lower().strip()


def normalize_url(s: str) -> str:
    s = s.strip()
    m = _MD_LINK_RE.match(s)
    if m:
        s = m.group(1)
    s = s.lower()
    for scheme in ("https://", "http://"):
        if s.startswith(scheme):
            s = s[len(scheme) :]
            break
    if s.startswith("www."):
        s = s[len("www.") :]
    if s.endswith("/"):
        s = s[:-1]
    return s


# ============================================================================
# Internal helpers
# ============================================================================

_PUNCT_CHARS = set('.,;:!?"\'()[]{}<>—–-')


def _tokenize_for_fuzzy(s: str) -> set[str]:
    cleaned = "".join(c if c not in _PUNCT_CHARS else " " for c in s)
    return {tok for tok in cleaned.lower().split() if tok}


# Fiscal-period / quarter strings must NEVER be treated as dates:
# 'FY2025 Q2' vs 'FY2025 Q1' are a real (systematic) mismatch, not a date.
_PERIOD_RE = re.compile(r"\b(Q[1-4]|H[12]|FY\s?\d{2,4})\b", re.IGNORECASE)


def _looks_like_period(v: Any) -> bool:
    return bool(_PERIOD_RE.search(str(v)))


# Strict strptime formats, tried first (fast, predictable).
_DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%SZ",
    "%B %d, %Y",       # March 15, 2024
    "%b %d, %Y",       # Mar 15, 2024
    "%d %B %Y",        # 15 March 2024
    "%d %b %Y",        # 15 Mar 2024
    "%m/%d/%Y",        # 03/15/2024
    "%d/%m/%Y",        # 15/03/2024  (ambiguous; tried after US order)
    "%B %Y",           # March 2024
]


def _try_parse_date_flexible(s: str) -> datetime | None:
    """Parse a date conservatively.

    Only attempts parsing if the string actually looks date-like: contains a
    month name with a digit, or digits joined by - or /. This prevents bare
    numbers (e.g. swim times '45.42', years '2024', counts) and arbitrary
    strings from being misread as dates. Fiscal-period strings are rejected
    by the caller via _looks_like_period.
    """
    s = s.strip()
    if not s or _looks_like_period(s):
        return None

    has_month = bool(re.search(r"[A-Za-z]{3,}", s)) and bool(re.search(r"\d", s))
    has_sep_numeric = bool(re.search(r"\d[/-]\d", s))
    if not (has_month or has_sep_numeric):
        return None

    # Try strict formats first.
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue

    # Fall back to dateutil for anything else date-shaped.
    try:
        from dateutil import parser as _dateutil_parser
        return _dateutil_parser.parse(s, fuzzy=False)
    except (ValueError, OverflowError, TypeError, ImportError):
        return None
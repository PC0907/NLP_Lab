"""Primitive value comparisons.

The smallest piece of the labeling pipeline: given two values (one from
gold, one from the model), decide whether they're the same. The matcher
calls into these for every leaf field.

Why a separate module:
  - These functions are pure and easy to unit-test.
  - The matching strategy (strict / fuzzy / semantic) is per-call, not
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
    """

    EXACT = "exact"
    CASE_INSENSITIVE = "case_insensitive"
    FUZZY = "fuzzy"
    NUMBER = "number"
    URL = "url"
    EMAIL = "email"
    DATE = "date"


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

    Type coercion: where possible, we attempt sensible cross-type matching
    (e.g., int 2024 vs str "2024" with NUMBER or DATE strategy).

    None handling:
      - both None → True (match)
      - one None, other not → False (mismatch)
      - empty string is treated as None for matching purposes

    Args:
        gold: The reference value from the gold annotation.
        extracted: The model's extracted value.
        strategy: Which comparison strategy to use.
        fuzzy_threshold: Minimum token-Jaccard similarity for FUZZY.
        number_tolerance: Relative tolerance for NUMBER (1% default).

    Returns:
        True if values are considered equivalent under the strategy.
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

    raise ValueError(f"Unsupported comparison strategy: {strategy!r}")


# ============================================================================
# Strategy implementations
# ============================================================================

def _compare_exact(g: Any, e: Any) -> bool:
    """Strict equality with type coercion for numeric/boolean strings.

    Examples:
        2024 == 2024            → True
        2024 == "2024"          → True (numeric coercion)
        True == "true"          → True (boolean coercion)
        "Yann" == "yann"        → False (case differs)
    """
    if type(g) is type(e):
        return g == e

    # Cross-type coercion for common cases.
    g_str, e_str = str(g), str(e)
    if g_str == e_str:
        return True

    # Booleans render as "True"/"False" in Python str(), but JSON uses
    # lowercase. Normalize.
    if isinstance(g, bool) or isinstance(e, bool):
        return g_str.lower() == e_str.lower()

    return False


def _compare_case_insensitive(g: Any, e: Any) -> bool:
    return str(g).strip().lower() == str(e).strip().lower()


def _compare_fuzzy(g: Any, e: Any, *, threshold: float) -> bool:
    """Token-Jaccard similarity above threshold counts as a match.

    Tokens are whitespace-separated, lowercased, with punctuation stripped.
    Jaccard = |intersection| / |union|. Threshold of 0.85 means most tokens
    must match.

    This is a placeholder for the more sophisticated `string_semantic`
    comparison ExtractBench specifies (which uses an LLM judge). We're
    documenting this limitation rather than approximating it badly.
    """
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

    Coerces strings to numbers if possible. Returns False if either side
    can't be parsed as a number.
    """
    try:
        g_n = float(g)
        e_n = float(e)
    except (TypeError, ValueError):
        return False

    # Exact integer match.
    if g_n == e_n:
        return True

    # Relative tolerance for non-zero values.
    if g_n == 0:
        return abs(e_n) < tolerance
    return abs(g_n - e_n) / abs(g_n) <= tolerance


def _compare_url(g: Any, e: Any) -> bool:
    return normalize_url(str(g)) == normalize_url(str(e))


def _compare_email(g: Any, e: Any) -> bool:
    return normalize_email(str(g)) == normalize_email(str(e))


def _compare_date(g: Any, e: Any) -> bool:
    """Date comparison with tolerant parsing.

    Tries multiple common formats. If both parse, compares as dates. If
    only one parses or neither does, falls back to case-insensitive string
    compare of the original values.

    Examples:
        "2024-03-15" == "2024-03-15"     → True (exact ISO)
        "2024" == 2024                    → True (year-only, coerced)
        "March 15, 2024" == "2024-03-15"  → True (parsed both)
        "Spring 2010" == "Spring 2010"    → True (string fallback)
        "Spring 2010" == "spring 2010"    → True (string fallback, ci)
    """
    g_date = _try_parse_date(str(g))
    e_date = _try_parse_date(str(e))
    if g_date is not None and e_date is not None:
        return g_date == e_date
    # Either failed to parse → fall back to string compare.
    return _compare_case_insensitive(g, e)


# ============================================================================
# Normalization helpers (exported)
# ============================================================================

# Markdown link wrapping: gold sometimes has [foo@bar.com](mailto:foo@bar.com)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")


def normalize_email(s: str) -> str:
    """Normalize an email address for comparison.

    - Strips markdown link wrapping: [x@y](mailto:x@y) → x@y
    - Strips leading mailto:
    - Lowercases
    - Strips surrounding whitespace
    """
    s = s.strip()
    m = _MD_LINK_RE.match(s)
    if m:
        s = m.group(1)
    if s.lower().startswith("mailto:"):
        s = s[len("mailto:") :]
    return s.lower().strip()


def normalize_url(s: str) -> str:
    """Normalize a URL for comparison.

    - Lowercases scheme and host
    - Strips trailing slash
    - Treats http and https as equivalent (returns the path-and-after part)
    - Strips surrounding whitespace
    """
    s = s.strip()
    m = _MD_LINK_RE.match(s)
    if m:
        s = m.group(1)
    s = s.lower()
    # Strip scheme
    for scheme in ("https://", "http://"):
        if s.startswith(scheme):
            s = s[len(scheme) :]
            break
    # Strip leading "www."
    if s.startswith("www."):
        s = s[len("www.") :]
    # Strip trailing slash
    if s.endswith("/"):
        s = s[:-1]
    return s


# ============================================================================
# Internal helpers
# ============================================================================

# Punctuation characters to strip during fuzzy tokenization.
_PUNCT_CHARS = set('.,;:!?"\'()[]{}<>—–-')


def _tokenize_for_fuzzy(s: str) -> set[str]:
    """Lowercase, strip punctuation, split on whitespace, return as set."""
    cleaned = "".join(c if c not in _PUNCT_CHARS else " " for c in s)
    return {tok for tok in cleaned.lower().split() if tok}


# Date formats we'll try, in order. Most-specific first.
_DATE_FORMATS = [
    "%Y-%m-%d",       # 2024-03-15
    "%Y/%m/%d",       # 2024/03/15
    "%Y-%m-%dT%H:%M:%S.%fZ",  # 2024-03-15T10:00:00.000Z (gold sometimes uses this)
    "%Y-%m-%dT%H:%M:%SZ",
    "%B %d, %Y",      # March 15, 2024
    "%d %B %Y",       # 15 March 2024
    "%B %Y",          # March 2024
    "%Y",             # 2024
]


def _try_parse_date(s: str) -> datetime | None:
    s = s.strip()
    if not s:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None
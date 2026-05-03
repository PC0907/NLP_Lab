"""Unit tests for primitive value comparisons.

These are pure-function tests with no I/O dependency. They run fast and
form the foundation for the matcher's correctness. If these break,
nothing downstream can be trusted.
"""

from __future__ import annotations

import pytest

from probe_extraction.labeling.value_compare import (
    ComparisonStrategy,
    compare_values,
    normalize_email,
    normalize_url,
)


# ============================================================================
# None / empty handling (cross-strategy)
# ============================================================================

@pytest.mark.parametrize("strategy", list(ComparisonStrategy))
class TestNullHandling:
    """Behavior should be consistent across strategies for null cases."""

    def test_both_none(self, strategy: ComparisonStrategy) -> None:
        assert compare_values(None, None, strategy=strategy) is True

    def test_gold_none_extracted_value(self, strategy: ComparisonStrategy) -> None:
        assert compare_values(None, "anything", strategy=strategy) is False

    def test_gold_value_extracted_none(self, strategy: ComparisonStrategy) -> None:
        assert compare_values("anything", None, strategy=strategy) is False

    def test_empty_string_treated_as_none(
        self, strategy: ComparisonStrategy
    ) -> None:
        # Empty string in either position should be normalized to None.
        assert compare_values("", None, strategy=strategy) is True
        assert compare_values(None, "", strategy=strategy) is True
        assert compare_values("", "", strategy=strategy) is True


# ============================================================================
# EXACT
# ============================================================================

class TestExact:
    def test_identical_strings(self) -> None:
        assert compare_values(
            "Yann LeCun", "Yann LeCun", strategy=ComparisonStrategy.EXACT
        ) is True

    def test_different_strings(self) -> None:
        assert compare_values(
            "Yann LeCun", "Geoff Hinton", strategy=ComparisonStrategy.EXACT
        ) is False

    def test_case_difference_is_a_mismatch(self) -> None:
        assert compare_values(
            "Yann LeCun", "yann lecun", strategy=ComparisonStrategy.EXACT
        ) is False

    def test_int_int(self) -> None:
        assert compare_values(2024, 2024, strategy=ComparisonStrategy.EXACT) is True
        assert compare_values(2024, 2023, strategy=ComparisonStrategy.EXACT) is False

    def test_int_string_coercion(self) -> None:
        # JSON often gives us numeric values as strings; lenient coercion.
        assert compare_values(
            2024, "2024", strategy=ComparisonStrategy.EXACT
        ) is True
        assert compare_values(
            "2024", 2024, strategy=ComparisonStrategy.EXACT
        ) is True

    def test_bool_string_coercion(self) -> None:
        assert compare_values(
            True, "true", strategy=ComparisonStrategy.EXACT
        ) is True
        assert compare_values(
            False, "false", strategy=ComparisonStrategy.EXACT
        ) is True
        assert compare_values(
            True, "false", strategy=ComparisonStrategy.EXACT
        ) is False


# ============================================================================
# CASE_INSENSITIVE
# ============================================================================

class TestCaseInsensitive:
    def test_case_difference_matches(self) -> None:
        assert compare_values(
            "Yann LeCun", "yann lecun",
            strategy=ComparisonStrategy.CASE_INSENSITIVE,
        ) is True

    def test_whitespace_stripped(self) -> None:
        assert compare_values(
            "  Yann  ", "yann",
            strategy=ComparisonStrategy.CASE_INSENSITIVE,
        ) is True

    def test_real_difference_still_fails(self) -> None:
        assert compare_values(
            "Yann LeCun", "Geoffrey Hinton",
            strategy=ComparisonStrategy.CASE_INSENSITIVE,
        ) is False


# ============================================================================
# FUZZY
# ============================================================================

class TestFuzzy:
    def test_identical(self) -> None:
        assert compare_values(
            "machine learning", "machine learning",
            strategy=ComparisonStrategy.FUZZY,
        ) is True

    def test_one_word_different_passes_threshold(self) -> None:
        # 2 of 3 tokens shared → Jaccard = 2/4 = 0.5, below default 0.85.
        assert compare_values(
            "deep learning models", "deep machine models",
            strategy=ComparisonStrategy.FUZZY,
        ) is False

    def test_punctuation_stripped(self) -> None:
        # "Y. Le Cun" tokenizes to {y, le, cun}; "Y Le Cun" same.
        assert compare_values(
            "Y. Le Cun", "Y Le Cun",
            strategy=ComparisonStrategy.FUZZY,
        ) is True

    def test_lowered(self) -> None:
        assert compare_values(
            "Yann LeCun", "YANN LECUN",
            strategy=ComparisonStrategy.FUZZY,
        ) is True

    def test_completely_different_fails(self) -> None:
        assert compare_values(
            "machine learning", "quantum computing",
            strategy=ComparisonStrategy.FUZZY,
        ) is False

    def test_threshold_respected(self) -> None:
        # With low threshold even partial overlaps match.
        assert compare_values(
            "alpha beta gamma", "alpha delta epsilon",
            strategy=ComparisonStrategy.FUZZY,
            fuzzy_threshold=0.1,
        ) is True
        # With high threshold same input fails.
        assert compare_values(
            "alpha beta gamma", "alpha delta epsilon",
            strategy=ComparisonStrategy.FUZZY,
            fuzzy_threshold=0.9,
        ) is False


# ============================================================================
# NUMBER
# ============================================================================

class TestNumber:
    def test_identical_int(self) -> None:
        assert compare_values(42, 42, strategy=ComparisonStrategy.NUMBER) is True

    def test_identical_float(self) -> None:
        assert compare_values(
            3.14, 3.14, strategy=ComparisonStrategy.NUMBER
        ) is True

    def test_within_tolerance(self) -> None:
        # Default tolerance 1%
        assert compare_values(
            100.0, 100.5, strategy=ComparisonStrategy.NUMBER
        ) is True
        assert compare_values(
            100.0, 101.5, strategy=ComparisonStrategy.NUMBER
        ) is False

    def test_string_coercion(self) -> None:
        assert compare_values(
            "42", 42, strategy=ComparisonStrategy.NUMBER
        ) is True

    def test_unparseable_returns_false(self) -> None:
        assert compare_values(
            "not a number", 42, strategy=ComparisonStrategy.NUMBER
        ) is False

    def test_zero_handling(self) -> None:
        assert compare_values(0, 0, strategy=ComparisonStrategy.NUMBER) is True
        assert compare_values(
            0, 0.001, strategy=ComparisonStrategy.NUMBER, number_tolerance=0.01,
        ) is True
        assert compare_values(
            0, 0.5, strategy=ComparisonStrategy.NUMBER
        ) is False


# ============================================================================
# DATE
# ============================================================================

class TestDate:
    def test_iso_dates_match(self) -> None:
        assert compare_values(
            "2024-03-15", "2024-03-15", strategy=ComparisonStrategy.DATE
        ) is True

    def test_year_only(self) -> None:
        assert compare_values(
            "2024", "2024", strategy=ComparisonStrategy.DATE
        ) is True

    def test_year_int_vs_string(self) -> None:
        assert compare_values(
            2024, "2024", strategy=ComparisonStrategy.DATE
        ) is True

    def test_iso_with_timezone(self) -> None:
        # Gold sometimes has these from spreadsheet exports.
        assert compare_values(
            "2024-03-15T10:00:00.000Z",
            "2024-03-15T10:00:00.000Z",
            strategy=ComparisonStrategy.DATE,
        ) is True

    def test_named_month(self) -> None:
        assert compare_values(
            "March 15, 2024", "2024-03-15",
            strategy=ComparisonStrategy.DATE,
        ) is True

    def test_unparseable_falls_back_to_string_compare(self) -> None:
        # "Spring 2010" doesn't parse; should fall back to ci string compare.
        assert compare_values(
            "Spring 2010", "spring 2010",
            strategy=ComparisonStrategy.DATE,
        ) is True
        assert compare_values(
            "Spring 2010", "Fall 2010",
            strategy=ComparisonStrategy.DATE,
        ) is False


# ============================================================================
# URL
# ============================================================================

class TestUrl:
    def test_identical(self) -> None:
        assert compare_values(
            "https://example.com",
            "https://example.com",
            strategy=ComparisonStrategy.URL,
        ) is True

    def test_http_vs_https(self) -> None:
        assert compare_values(
            "http://example.com",
            "https://example.com",
            strategy=ComparisonStrategy.URL,
        ) is True

    def test_trailing_slash(self) -> None:
        assert compare_values(
            "https://example.com",
            "https://example.com/",
            strategy=ComparisonStrategy.URL,
        ) is True

    def test_www_prefix(self) -> None:
        assert compare_values(
            "https://www.example.com",
            "https://example.com",
            strategy=ComparisonStrategy.URL,
        ) is True

    def test_real_difference(self) -> None:
        assert compare_values(
            "https://example.com",
            "https://other.com",
            strategy=ComparisonStrategy.URL,
        ) is False

    def test_normalize_url_function(self) -> None:
        assert normalize_url("HTTPS://Example.com/") == "example.com"
        assert normalize_url("http://www.example.com") == "example.com"


# ============================================================================
# EMAIL
# ============================================================================

class TestEmail:
    def test_identical(self) -> None:
        assert compare_values(
            "foo@bar.com", "foo@bar.com",
            strategy=ComparisonStrategy.EMAIL,
        ) is True

    def test_case_insensitive(self) -> None:
        assert compare_values(
            "Foo@Bar.COM", "foo@bar.com",
            strategy=ComparisonStrategy.EMAIL,
        ) is True

    def test_markdown_link_wrapping(self) -> None:
        # Real example: gold has [foo@bar.com](mailto:foo@bar.com)
        assert compare_values(
            "[foo@bar.com](mailto:foo@bar.com)",
            "foo@bar.com",
            strategy=ComparisonStrategy.EMAIL,
        ) is True

    def test_mailto_prefix(self) -> None:
        assert compare_values(
            "mailto:foo@bar.com", "foo@bar.com",
            strategy=ComparisonStrategy.EMAIL,
        ) is True

    def test_real_difference(self) -> None:
        assert compare_values(
            "foo@bar.com", "baz@bar.com",
            strategy=ComparisonStrategy.EMAIL,
        ) is False

    def test_normalize_email_function(self) -> None:
        assert normalize_email("Foo@Bar.COM") == "foo@bar.com"
        assert normalize_email("[x@y.z](mailto:x@y.z)") == "x@y.z"
        assert normalize_email("mailto:foo@bar.com") == "foo@bar.com"
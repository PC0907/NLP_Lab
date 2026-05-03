"""Unit tests for the matcher.

Strategy:
  - Build small synthetic schemas inline (no benchmark dependency).
  - Test each error category (match, value_mismatch, hallucination,
    omission, type_mismatch) with focused minimal cases.
  - Test array handling (object arrays positional, primitive arrays as sets).
  - Test schema-driven strategy selection.
  - Test recursion / nested structures.
"""

from __future__ import annotations

import pytest

from probe_extraction.labeling.matcher import (
    FieldLabel,
    Matcher,
    label_extraction,
)


# ============================================================================
# Helpers
# ============================================================================

def find_label(labels: list[FieldLabel], path_str: str) -> FieldLabel:
    """Look up a label by path_str. Fails the test if not found."""
    for lab in labels:
        if lab.path_str == path_str:
            return lab
    pytest.fail(
        f"Expected label for path {path_str!r}; found: "
        f"{[lab.path_str for lab in labels]}"
    )


# ============================================================================
# Top-level helpers
# ============================================================================

class TestLabelExtractionEntry:
    def test_minimal_match(self) -> None:
        schema = {"type": "object", "properties": {
            "title": {"type": "string", "evaluation_config": "string_exact"},
        }}
        result = label_extraction(
            doc_id="doc1",
            domain="test",
            schema=schema,
            gold={"title": "Hello"},
            extracted={"title": "Hello"},
        )
        assert result.doc_id == "doc1"
        assert result.domain == "test"
        assert result.n_total == 1
        assert result.n_errors == 0
        assert result.error_rate == 0.0


# ============================================================================
# Single-field error categories
# ============================================================================

class TestErrorCategories:
    def _label_one(self, gold_val, ext_val, eval_config="string_exact"):
        """Helper: label a one-field schema with given gold/extracted."""
        schema = {"type": "object", "properties": {
            "x": {"type": "string", "evaluation_config": eval_config},
        }}
        return label_extraction(
            doc_id="d", domain="t",
            schema=schema,
            gold={"x": gold_val},
            extracted={"x": ext_val},
        )

    def test_match(self) -> None:
        result = self._label_one("hello", "hello")
        lab = find_label(result.labels, "x")
        assert lab.is_error == 0
        assert lab.error_type == "match"

    def test_value_mismatch(self) -> None:
        result = self._label_one("hello", "world")
        lab = find_label(result.labels, "x")
        assert lab.is_error == 1
        assert lab.error_type == "value_mismatch"
        assert result.n_value_mismatches == 1

    def test_hallucination(self) -> None:
        result = self._label_one(None, "made up")
        lab = find_label(result.labels, "x")
        assert lab.is_error == 1
        assert lab.error_type == "hallucination"
        assert result.n_hallucinations == 1
        assert lab.gold_present is False
        assert lab.extracted_present is True

    def test_omission(self) -> None:
        result = self._label_one("real value", None)
        lab = find_label(result.labels, "x")
        assert lab.is_error == 1
        assert lab.error_type == "omission"
        assert result.n_omissions == 1
        assert lab.gold_present is True
        assert lab.extracted_present is False

    def test_both_empty_is_match(self) -> None:
        result = self._label_one(None, None)
        lab = find_label(result.labels, "x")
        assert lab.is_error == 0
        assert lab.error_type == "match"

    def test_empty_string_normalized_to_none(self) -> None:
        result = self._label_one("", None)
        lab = find_label(result.labels, "x")
        assert lab.is_error == 0  # both treated as empty
        assert lab.error_type == "match"


# ============================================================================
# Schema-driven strategy
# ============================================================================

class TestStrategySelection:
    def test_string_exact_used(self) -> None:
        schema = {"type": "object", "properties": {
            "x": {"type": "string", "evaluation_config": "string_exact"},
        }}
        result = label_extraction(
            doc_id="d", domain="t", schema=schema,
            gold={"x": "Hello"}, extracted={"x": "hello"},
        )
        # EXACT case-sensitive: "Hello" != "hello"
        assert find_label(result.labels, "x").is_error == 1

    def test_string_case_insensitive_used(self) -> None:
        schema = {"type": "object", "properties": {
            "x": {"type": "string",
                  "evaluation_config": "string_case_insensitive"},
        }}
        result = label_extraction(
            doc_id="d", domain="t", schema=schema,
            gold={"x": "Hello"}, extracted={"x": "hello"},
        )
        assert find_label(result.labels, "x").is_error == 0

    def test_integer_exact(self) -> None:
        schema = {"type": "object", "properties": {
            "x": {"type": "integer", "evaluation_config": "integer_exact"},
        }}
        # Lenient string coercion: "2024" matches 2024 under EXACT.
        result = label_extraction(
            doc_id="d", domain="t", schema=schema,
            gold={"x": 2024}, extracted={"x": "2024"},
        )
        assert find_label(result.labels, "x").is_error == 0

    def test_unknown_strategy_falls_back(self) -> None:
        schema = {"type": "object", "properties": {
            "x": {"type": "string", "evaluation_config": "unknown_strategy"},
        }}
        # Should not raise; falls back to default (EXACT for leaves).
        result = label_extraction(
            doc_id="d", domain="t", schema=schema,
            gold={"x": "Hello"}, extracted={"x": "Hello"},
        )
        assert find_label(result.labels, "x").is_error == 0


# ============================================================================
# Nested objects
# ============================================================================

class TestNestedObjects:
    def test_nested_field_paths(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "personalInfo": {
                    "type": "object",
                    "properties": {
                        "fullName": {"type": "string",
                                     "evaluation_config": "string_exact"},
                        "email": {"type": "string",
                                  "evaluation_config": "string_exact"},
                    },
                },
            },
        }
        result = label_extraction(
            doc_id="d", domain="t", schema=schema,
            gold={"personalInfo": {"fullName": "Alice", "email": "a@b.com"}},
            extracted={"personalInfo": {"fullName": "Alice", "email": "wrong@b.com"}},
        )
        assert result.n_total == 2
        assert find_label(result.labels, "personalInfo.fullName").is_error == 0
        assert find_label(result.labels, "personalInfo.email").is_error == 1


# ============================================================================
# Object arrays (positional)
# ============================================================================

class TestObjectArrays:
    SCHEMA = {
        "type": "object",
        "properties": {
            "authors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string",
                                 "evaluation_config": "string_exact"},
                    },
                },
            },
        },
    }

    def test_aligned_arrays(self) -> None:
        result = label_extraction(
            doc_id="d", domain="t", schema=self.SCHEMA,
            gold={"authors": [{"name": "Alice"}, {"name": "Bob"}]},
            extracted={"authors": [{"name": "Alice"}, {"name": "Bob"}]},
        )
        assert result.n_total == 2
        assert result.n_errors == 0
        assert find_label(result.labels, "authors.0.name").is_error == 0
        assert find_label(result.labels, "authors.1.name").is_error == 0

    def test_extracted_shorter(self) -> None:
        # Model produced fewer authors than gold → omissions.
        result = label_extraction(
            doc_id="d", domain="t", schema=self.SCHEMA,
            gold={"authors": [{"name": "Alice"}, {"name": "Bob"}]},
            extracted={"authors": [{"name": "Alice"}]},
        )
        assert result.n_total == 2
        assert find_label(result.labels, "authors.0.name").is_error == 0
        bob_label = find_label(result.labels, "authors.1.name")
        assert bob_label.is_error == 1
        assert bob_label.error_type == "omission"

    def test_extracted_longer(self) -> None:
        # Model produced extra authors not in gold → hallucinations.
        result = label_extraction(
            doc_id="d", domain="t", schema=self.SCHEMA,
            gold={"authors": [{"name": "Alice"}]},
            extracted={"authors": [{"name": "Alice"}, {"name": "Bogus"}]},
        )
        assert result.n_total == 2
        assert find_label(result.labels, "authors.0.name").is_error == 0
        bogus_label = find_label(result.labels, "authors.1.name")
        assert bogus_label.is_error == 1
        assert bogus_label.error_type == "hallucination"

    def test_value_mismatch_at_position(self) -> None:
        result = label_extraction(
            doc_id="d", domain="t", schema=self.SCHEMA,
            gold={"authors": [{"name": "Alice"}, {"name": "Bob"}]},
            extracted={"authors": [{"name": "Alice"}, {"name": "Carol"}]},
        )
        assert result.n_value_mismatches == 1
        bob_label = find_label(result.labels, "authors.1.name")
        assert bob_label.is_error == 1
        assert bob_label.error_type == "value_mismatch"


# ============================================================================
# Primitive arrays (set comparison)
# ============================================================================

class TestPrimitiveArrays:
    SCHEMA = {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "array",
                "evaluation_config": "array_llm",
                "items": {"type": "string",
                          "evaluation_config": "string_semantic"},
            },
        },
    }

    def test_identical_sets(self) -> None:
        result = label_extraction(
            doc_id="d", domain="t", schema=self.SCHEMA,
            gold={"keywords": ["nlp", "deep learning"]},
            extracted={"keywords": ["nlp", "deep learning"]},
        )
        # Single field for the whole array.
        assert result.n_total == 1
        assert find_label(result.labels, "keywords").is_error == 0

    def test_order_doesnt_matter(self) -> None:
        result = label_extraction(
            doc_id="d", domain="t", schema=self.SCHEMA,
            gold={"keywords": ["a", "b", "c"]},
            extracted={"keywords": ["c", "a", "b"]},
        )
        assert find_label(result.labels, "keywords").is_error == 0

    def test_case_doesnt_matter(self) -> None:
        result = label_extraction(
            doc_id="d", domain="t", schema=self.SCHEMA,
            gold={"keywords": ["NLP", "Deep Learning"]},
            extracted={"keywords": ["nlp", "deep learning"]},
        )
        assert find_label(result.labels, "keywords").is_error == 0

    def test_different_sets(self) -> None:
        result = label_extraction(
            doc_id="d", domain="t", schema=self.SCHEMA,
            gold={"keywords": ["a", "b"]},
            extracted={"keywords": ["a", "c"]},
        )
        lab = find_label(result.labels, "keywords")
        assert lab.is_error == 1
        assert lab.error_type == "value_mismatch"

    def test_gold_empty_extracted_has_values(self) -> None:
        result = label_extraction(
            doc_id="d", domain="t", schema=self.SCHEMA,
            gold={"keywords": []},
            extracted={"keywords": ["fabricated"]},
        )
        lab = find_label(result.labels, "keywords")
        assert lab.is_error == 1
        assert lab.error_type == "hallucination"

    def test_gold_has_values_extracted_empty(self) -> None:
        result = label_extraction(
            doc_id="d", domain="t", schema=self.SCHEMA,
            gold={"keywords": ["real"]},
            extracted={"keywords": []},
        )
        lab = find_label(result.labels, "keywords")
        assert lab.is_error == 1
        assert lab.error_type == "omission"

    def test_both_empty_is_match(self) -> None:
        result = label_extraction(
            doc_id="d", domain="t", schema=self.SCHEMA,
            gold={"keywords": []},
            extracted={"keywords": []},
        )
        lab = find_label(result.labels, "keywords")
        assert lab.is_error == 0
        assert lab.error_type == "match"


# ============================================================================
# Type mismatches
# ============================================================================

class TestTypeMismatches:
    def test_dict_vs_string(self) -> None:
        schema = {
            "type": "object",
            "properties": {"x": {"type": "object"}},
        }
        result = label_extraction(
            doc_id="d", domain="t", schema=schema,
            gold={"x": {"key": "val"}},
            extracted={"x": "wrong type"},
        )
        lab = find_label(result.labels, "x")
        assert lab.error_type == "type_mismatch"
        assert lab.is_error == 1

    def test_list_vs_dict(self) -> None:
        schema = {"type": "object", "properties": {"x": {}}}
        result = label_extraction(
            doc_id="d", domain="t", schema=schema,
            gold={"x": [1, 2, 3]},
            extracted={"x": {"a": 1}},
        )
        lab = find_label(result.labels, "x")
        assert lab.error_type == "type_mismatch"


# ============================================================================
# anyOf resolution
# ============================================================================

class TestAnyOf:
    def test_anyof_with_value_picks_string(self) -> None:
        # Mimics the resume schema: skills can be array OR object OR null.
        schema = {
            "type": "object",
            "properties": {
                "skills": {
                    "anyOf": [
                        {
                            "type": "array",
                            "items": {"type": "string",
                                      "evaluation_config": "string_semantic"},
                        },
                        {"type": "null"},
                    ],
                },
            },
        }
        result = label_extraction(
            doc_id="d", domain="t", schema=schema,
            gold={"skills": ["python", "rust"]},
            extracted={"skills": ["python", "rust"]},
        )
        # Should pick the array variant and treat as primitive array.
        assert result.n_total == 1
        assert find_label(result.labels, "skills").is_error == 0


# ============================================================================
# Realistic mini-schema (academic/research style)
# ============================================================================

class TestRealisticAcademic:
    SCHEMA = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "evaluation_config": "string_semantic"},
            "ids": {"type": "string", "evaluation_config": "string_exact"},
            "publication_type": {"type": "string",
                                 "evaluation_config": "string_exact"},
            "authors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string",
                                 "evaluation_config": "string_semantic"},
                    },
                },
            },
        },
    }

    def test_perfect_extraction(self) -> None:
        gold = {
            "title": "Some Paper",
            "ids": "10.1234/abc",
            "publication_type": "Conference Paper",
            "authors": [{"name": "Alice"}, {"name": "Bob"}],
        }
        result = label_extraction(
            doc_id="d", domain="t", schema=self.SCHEMA,
            gold=gold, extracted=gold,
        )
        assert result.n_errors == 0
        assert result.n_total == 4

    def test_mixed_outcome(self) -> None:
        gold = {
            "title": "Some Paper",
            "ids": "10.1234/abc",
            "publication_type": "Conference Paper",
            "authors": [{"name": "Alice"}, {"name": "Bob"}],
        }
        extracted = {
            "title": "Some Paper",         # match
            "ids": None,                    # omission
            "publication_type": "Article",  # value_mismatch
            "authors": [{"name": "Alice"}, {"name": "Carol"}],  # 1 match, 1 mismatch
        }
        result = label_extraction(
            doc_id="d", domain="t", schema=self.SCHEMA,
            gold=gold, extracted=extracted,
        )
        assert result.n_total == 5
        assert result.n_errors == 3
        assert result.n_omissions == 1
        assert result.n_value_mismatches == 2  # publication_type + authors.1.name
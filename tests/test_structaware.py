"""Tests for the AUTO comparison strategy and the structure-aware matcher mode.

These cover the DeepSeek-R1 labeling fix: the model emits flat values where the
gold schema nests objects (e.g. "B. Boser" vs {"name": "B. Boser", ...}). Under
the original strict matcher this is a wholesale type_mismatch (inflated ~95%
error rate); under structure-aware + AUTO it is matched on value.
"""
from __future__ import annotations

from probe_extraction.labeling.matcher import label_extraction
from probe_extraction.labeling.value_compare import ComparisonStrategy, compare_values


AUTO = ComparisonStrategy.AUTO


# ---------------------------------------------------------------------------
# AUTO comparison strategy
# ---------------------------------------------------------------------------

def test_auto_numeric_currency_and_commas():
    assert compare_values("$1,234", "1234", strategy=AUTO)
    assert compare_values("49.7", "49.70", strategy=AUTO)
    assert compare_values(1000000, "1000000", strategy=AUTO)


def test_auto_numeric_real_mismatch_not_suppressed():
    # 0.018 relative diff > 1% tolerance -> genuine mismatch stays flagged.
    assert not compare_values("11.1", "11.3", strategy=AUTO)


def test_auto_case_insensitive_string():
    assert compare_values("New York", "new york", strategy=AUTO)
    assert not compare_values("New York", "New Jersey", strategy=AUTO)


def test_auto_date_formats_equal():
    assert compare_values("March 15, 2024", "2024-03-15", strategy=AUTO)


def test_auto_fiscal_period_never_a_date():
    # FY/quarter strings must stay distinguishable, not be parsed as dates.
    assert not compare_values("FY2025 Q2", "FY2025 Q1", strategy=AUTO)
    assert compare_values("FY2025 Q2", "fy2025 q2", strategy=AUTO)


# ---------------------------------------------------------------------------
# Structure-aware matcher: primitive-vs-object
# ---------------------------------------------------------------------------

_AUTHOR_SCHEMA = {
    "type": "object",
    "properties": {
        "author": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "affiliation": {"type": "string"},
            },
        }
    },
}


def _label(gold, extracted, *, structure_aware, schema=_AUTHOR_SCHEMA):
    return label_extraction(
        doc_id="d", domain="academic/research", schema=schema,
        gold=gold, extracted=extracted,
        leaf_default=ComparisonStrategy.AUTO if structure_aware else ComparisonStrategy.EXACT,
        structure_aware=structure_aware,
    )


def test_strict_flags_flat_string_as_type_mismatch():
    gold = {"author": {"name": "B. Boser", "affiliation": "AT&T Bell Laboratories"}}
    extracted = {"author": "B. Boser"}
    res = _label(gold, extracted, structure_aware=False)
    # Strict: object-vs-primitive -> a single type_mismatch error.
    assert res.n_errors == 1
    assert any(l.error_type == "type_mismatch" for l in res.labels)


def test_structaware_matches_flat_string_against_object_leaf():
    gold = {"author": {"name": "B. Boser", "affiliation": "AT&T Bell Laboratories"}}
    extracted = {"author": "B. Boser"}
    res = _label(gold, extracted, structure_aware=True)
    # Structure-aware: "B. Boser" matches the gold object's `name` leaf.
    assert res.n_errors == 0
    lab = [l for l in res.labels if l.path_str == "author"][0]
    assert lab.is_error == 0
    assert lab.comparison_strategy.startswith("structaware_")


def test_structaware_still_flags_genuinely_wrong_flat_value():
    gold = {"author": {"name": "B. Boser", "affiliation": "AT&T Bell Laboratories"}}
    extracted = {"author": "Someone Else"}
    res = _label(gold, extracted, structure_aware=True)
    assert res.n_errors == 1


def test_structaware_array_of_objects_vs_array_of_strings():
    schema = {
        "type": "object",
        "properties": {
            "authors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            }
        },
    }
    gold = {"authors": [{"name": "Y. LeCun"}, {"name": "B. Boser"}]}
    extracted = {"authors": ["Y. LeCun", "B. Boser"]}
    strict = _label(gold, extracted, structure_aware=False, schema=schema)
    relaxed = _label(gold, extracted, structure_aware=True, schema=schema)
    assert strict.n_errors == 2          # two type_mismatches
    assert relaxed.n_errors == 0         # both matched on value

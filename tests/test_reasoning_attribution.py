"""Tests for field-localized reasoning attribution (CPU, numpy-only).

Loaded directly by file path so the torch-importing package chain is avoided.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_MOD = Path(__file__).resolve().parents[1] / "src" / "probe_extraction" / "extraction" / "reasoning_attribution.py"
_spec = importlib.util.spec_from_file_location("reasoning_attribution", _MOD)
ra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ra)


def _states(token_strings, dim=4, seed=0):
    rng = np.random.default_rng(seed)
    return {10: rng.standard_normal((len(token_strings), dim)).astype(np.float32)}


def test_full_value_match_pools_only_matched_tokens():
    toks = ["The", " answer", " is", " Johann", " Sebastian", " Bach", " ."]
    st = _states(toks)
    out = ra.attribute_field(toks, st, "Johann Sebastian Bach")
    assert out["match_type"] == "full"
    assert out["features"]["mentioned"] == 1.0
    assert out["features"]["match_full"] == 1.0
    # Bach spans tokens 3,4,5 -> pooled mean of exactly those rows.
    expected = st[10][[3, 4, 5]].astype(np.float32).mean(axis=0).astype(np.float16)
    np.testing.assert_allclose(out["attr_vec"][10], expected, rtol=1e-3)


def test_absent_value_flags_not_mentioned_and_zero_vector():
    toks = ["Thinking", " about", " something", " else"]
    st = _states(toks)
    out = ra.attribute_field(toks, st, "Tambourine")
    assert out["match_type"] == "none"
    assert out["features"]["mentioned"] == 0.0
    assert np.all(out["attr_vec"][10] == 0)


def test_partial_fallback_matches_longest_word():
    toks = ["It", " is", " in", " California", " somewhere"]
    st = _states(toks)
    out = ra.attribute_field(toks, st, "Lake Tahoe, California")
    assert out["match_type"] == "partial"
    assert out["features"]["mentioned"] == 1.0
    assert out["features"]["match_full"] == 0.0


def test_word_boundary_avoids_substring_false_positive():
    toks = ["Mozart", " wrote", " it"]
    st = _states(toks)
    out = ra.attribute_field(toks, st, "art")  # must NOT match inside 'Mozart'
    assert out["match_type"] == "none"
    assert out["features"]["mentioned"] == 0.0


def test_integer_float_value_matches_bare_integer():
    toks = ["count", " is", " 5", " total"]
    st = _states(toks)
    out = ra.attribute_field(toks, st, 5.0)
    assert out["features"]["mentioned"] == 1.0


def test_mention_count_and_positions():
    toks = ["Bach", " and", " later", " Bach", " again", " end"]
    st = _states(toks)
    out = ra.attribute_field(toks, st, "Bach")
    assert out["features"]["mention_count"] == 2.0
    assert 0.0 <= out["features"]["first_frac"] < out["features"]["last_frac"] <= 1.0


def test_none_and_bool_values_are_not_searchable():
    toks = ["true", " maybe", " none"]
    st = _states(toks)
    assert ra.attribute_field(toks, st, None)["features"]["mentioned"] == 0.0
    assert ra.attribute_field(toks, st, True)["features"]["mentioned"] == 0.0


def test_features_to_array_is_fixed_length():
    toks = ["x", " Bach"]
    st = _states(toks)
    out = ra.attribute_field(toks, st, "Bach")
    arr = ra.features_to_array(out["features"])
    assert arr.shape == (len(ra.FEATURE_NAMES),)

"""Tests for leaf addressing in a parsed JSON document.

Stage 12's measured repair rate depends entirely on replacing the right leaf.
A wrong set would not raise -- it would quietly produce a hybrid record that
does not correspond to what was flagged, and the resulting number would look
perfectly plausible. So the failure modes are pinned down here.
"""

from __future__ import annotations

import pytest

from probe_extraction.utils.jsonpath import json_get, json_set, with_replacements


def rec():
    return {
        "band_name": "Led Zeppelin",
        "formation_year": 1969,
        "members": [{"name": "Jimmy Page"}, {"name": "John Bonham"}],
        "label": None,
        "meta": {"0": "string-key-zero"},
    }


# --- json_get --------------------------------------------------------------

def test_reads_a_top_level_leaf():
    assert json_get(rec(), ["band_name"]) == (True, "Led Zeppelin")


def test_reads_through_a_list_index():
    assert json_get(rec(), ["members", 1, "name"]) == (True, "John Bonham")


def test_negative_index_works():
    assert json_get(rec(), ["members", -1, "name"]) == (True, "John Bonham")


def test_a_stored_null_is_found_not_treated_as_absent():
    """A regenerated null is a real answer; a missing path is not. Conflating
    them would misclassify a genuine replacement as 'unavailable'."""
    found, value = json_get(rec(), ["label"])
    assert found is True and value is None


def test_missing_key_reports_not_found():
    assert json_get(rec(), ["nope"]) == (False, None)


def test_missing_nested_key_reports_not_found():
    assert json_get(rec(), ["members", 0, "nope"]) == (False, None)


def test_index_out_of_range_reports_not_found():
    assert json_get(rec(), ["members", 5, "name"]) == (False, None)


def test_descending_into_a_scalar_reports_not_found():
    assert json_get(rec(), ["band_name", "deeper"]) == (False, None)


def test_int_path_element_can_match_a_string_object_key():
    assert json_get(rec(), ["meta", 0]) == (True, "string-key-zero")


def test_booleans_are_not_treated_as_list_indices():
    """True == 1 in Python. Indexing a list with it would silently address the
    wrong element."""
    assert json_get(rec(), ["members", True, "name"]) == (False, None)


def test_empty_path_returns_the_whole_object():
    found, value = json_get(rec(), [])
    assert found is True and value["band_name"] == "Led Zeppelin"


# --- json_set --------------------------------------------------------------

def test_sets_a_top_level_leaf():
    r = rec()
    assert json_set(r, ["formation_year"], 1968) is True
    assert r["formation_year"] == 1968


def test_sets_through_a_list_index():
    r = rec()
    assert json_set(r, ["members", 0, "name"], "J. Page") is True
    assert r["members"][0]["name"] == "J. Page"
    assert r["members"][1]["name"] == "John Bonham"   # neighbour untouched


def test_refuses_to_create_a_missing_key():
    """Creating a key would fabricate a field the model never emitted."""
    r = rec()
    assert json_set(r, ["invented"], "x") is False
    assert "invented" not in r


def test_refuses_to_create_a_missing_nested_path():
    r = rec()
    assert json_set(r, ["nowhere", "deep"], "x") is False
    assert "nowhere" not in r


def test_refuses_an_out_of_range_index():
    r = rec()
    assert json_set(r, ["members", 9, "name"], "x") is False
    assert len(r["members"]) == 2


def test_can_overwrite_an_existing_null():
    r = rec()
    assert json_set(r, ["label"], "Atlantic") is True
    assert r["label"] == "Atlantic"


def test_setting_an_empty_path_is_refused():
    r = rec()
    assert json_set(r, [], "x") is False


def test_boolean_last_element_is_refused():
    r = rec()
    assert json_set(r, ["members", True], "x") is False


# --- with_replacements -----------------------------------------------------

def test_builds_a_hybrid_without_mutating_the_original():
    original = rec()
    hybrid, failed = with_replacements(original, {("formation_year",): 1968})
    assert hybrid["formation_year"] == 1968
    assert original["formation_year"] == 1969      # source untouched
    assert failed == []


def test_only_the_named_paths_change():
    original = rec()
    hybrid, _ = with_replacements(original, {("members", 0, "name"): "X"})
    assert hybrid["members"][0]["name"] == "X"
    assert hybrid["members"][1]["name"] == "John Bonham"
    assert hybrid["band_name"] == "Led Zeppelin"


def test_failed_paths_are_reported_not_silently_dropped():
    hybrid, failed = with_replacements(rec(), {("nope",): 1, ("band_name",): "X"})
    assert failed == [("nope",)]
    assert hybrid["band_name"] == "X"


def test_repeated_builds_from_one_source_do_not_interfere():
    """Each budget builds its own hybrid from the same original; a shared
    mutation would make later budgets inherit earlier swaps."""
    original = rec()
    a, _ = with_replacements(original, {("formation_year",): 1968})
    b, _ = with_replacements(original, {("band_name",): "Deep Purple"})
    assert a["band_name"] == "Led Zeppelin"
    assert b["formation_year"] == 1969


def test_no_replacements_yields_an_equal_copy():
    original = rec()
    hybrid, failed = with_replacements(original, {})
    assert hybrid == original and hybrid is not original and failed == []

"""Tests for Stage 12's repair/damage accounting (scripts/12_regen_evaluate.py).

This is the stage that replaces Stage 9's invented repair=0.7 / damage=0.05 with
measurement, so the accounting is the result. A miscount here would not raise --
it would produce a clean-looking repair rate that is simply wrong, which is the
one failure this project cannot ship. Every branch of the outcome table is
pinned down below, with an injected labeler so the arithmetic is checked without
a real matcher, gold annotations, or any artifacts on disk.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s12 = _load("stage12", "scripts/12_regen_evaluate.py")

# The accounting itself lives in the package (importable by name, so the
# parallel pass can ship it to a worker process); the script keeps only the
# budget curve built on top of it.
from probe_extraction.regen import evaluate as rg  # noqa: E402


# ---------------------------------------------------------------------------
# Which resamples are allowed to count
# ---------------------------------------------------------------------------

def _sample(idx, parsed, *, finish="stop", err=None):
    return {"index": idx, "parsed_json": parsed, "parse_error": err,
            "finish_reason": finish}


def test_truncated_samples_are_excluded():
    """A generation that hit max_new_tokens only has JSON because the parser
    repaired it. Counting that as the model's answer would credit or blame it
    for text it never produced."""
    payload = {"samples": [_sample(0, {"a": 1}, finish="length"),
                           _sample(1, {"a": 2})]}
    samples, stats = rg.usable_samples(payload)
    assert samples == [{"a": 2}]
    assert stats["truncated"] == 1 and stats["usable"] == 1


def test_parse_failures_are_excluded():
    payload = {"samples": [_sample(0, None, err="no JSON found"),
                           _sample(1, {"a": 2})]}
    samples, stats = rg.usable_samples(payload)
    assert samples == [{"a": 2}] and stats["parse_error"] == 1


def test_non_object_json_is_excluded():
    payload = {"samples": [_sample(0, ["a", "b"]), _sample(1, {"a": 2})]}
    samples, stats = rg.usable_samples(payload)
    assert samples == [{"a": 2}] and stats["no_json"] == 1


def test_usable_samples_keep_their_order():
    payload = {"samples": [_sample(0, {"a": 1}), _sample(1, {"a": 2})]}
    samples, stats = rg.usable_samples(payload)
    assert samples == [{"a": 1}, {"a": 2}]
    assert stats["total"] == 2 and stats["usable"] == 2


def test_a_document_with_no_usable_sample_yields_nothing():
    payload = {"samples": [_sample(0, {"a": 1}, finish="length")]}
    samples, _ = rg.usable_samples(payload)
    assert samples == []


# ---------------------------------------------------------------------------
# What the regeneration offers for one field
# ---------------------------------------------------------------------------

def test_first_uses_sample_zero_even_when_outvoted():
    samples = [{"a": "x"}, {"a": "y"}, {"a": "y"}]
    assert rg.select_value(samples, ["a"], "first") == (True, "x")


def test_first_does_not_fall_through_to_a_later_sample():
    """If a missing path fell through to sample 1, `first` would silently
    become best-of-k and overstate what a single regeneration call buys."""
    samples = [{"b": 1}, {"a": "y"}]
    assert rg.select_value(samples, ["a"], "first") == (False, None)


def test_vote_takes_the_majority_value():
    samples = [{"a": "x"}, {"a": "y"}, {"a": "y"}]
    assert rg.select_value(samples, ["a"], "vote") == (True, "y")


def test_vote_ignores_samples_that_lack_the_path():
    samples = [{"b": 1}, {"a": "y"}, {"a": "y"}]
    assert rg.select_value(samples, ["a"], "vote") == (True, "y")


def test_vote_breaks_ties_by_first_appearance():
    """Deterministic, and independent of dict iteration order."""
    samples = [{"a": "x"}, {"a": "y"}]
    assert rg.select_value(samples, ["a"], "vote") == (True, "x")
    assert rg.select_value(list(reversed(samples)), ["a"], "vote") == (True, "y")


def test_vote_handles_unhashable_values():
    samples = [{"a": [1, 2]}, {"a": [1, 2]}, {"a": [3]}]
    assert rg.select_value(samples, ["a"], "vote") == (True, [1, 2])


def test_no_usable_samples_means_unavailable():
    assert rg.select_value([], ["a"], "first") == (False, None)
    assert rg.select_value([], ["a"], "vote") == (False, None)


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError):
        rg.select_value([{"a": 1}], ["a"], "best_of_k")


# ---------------------------------------------------------------------------
# Per-field swap accounting
# ---------------------------------------------------------------------------

def _labeler_from(gold: dict, paths: dict, *, drop=(), calls=None):
    """A stand-in matcher: a field is an error when its value differs from
    `gold`. `drop` names path_strs to omit from the output, simulating a swap
    that changed the record's shape."""
    from probe_extraction.utils.jsonpath import json_get

    def labeler(record):
        if calls is not None:
            calls.append(record)
        out = {}
        for ps, path in paths.items():
            if ps in drop:
                continue
            found, val = json_get(record, path)
            out[ps] = int(not found or val != gold[ps])
        return out
    return labeler


def test_a_repair_is_counted_and_lowers_the_error_count():
    original = {"a": "wrong", "b": "right"}
    paths = {"a": ["a"], "b": ["b"]}
    gold = {"a": "right_a", "b": "right"}
    scored = [("a", ["a"], 1), ("b", ["b"], 0)]
    out = rg.evaluate_document(original, scored, [{"a": "right_a", "b": "right"}],
                                "first", _labeler_from(gold, paths))
    rec_a = out["fields"][0]
    assert rec_a["self"] == "repaired" and rec_a["net_delta"] == -1
    assert out["base_errors"] == 1


def test_damage_is_counted_and_raises_the_error_count():
    original = {"a": "right_a"}
    paths, gold = {"a": ["a"]}, {"a": "right_a"}
    scored = [("a", ["a"], 0)]
    out = rg.evaluate_document(original, scored, [{"a": "broken"}], "first",
                                _labeler_from(gold, paths))
    assert out["fields"][0]["self"] == "damaged"
    assert out["fields"][0]["net_delta"] == 1


def test_a_different_but_still_wrong_answer_is_unchanged():
    original = {"a": "wrong1"}
    paths, gold = {"a": ["a"]}, {"a": "right"}
    out = rg.evaluate_document(original, [("a", ["a"], 1)], [{"a": "wrong2"}],
                                "first", _labeler_from(gold, paths))
    assert out["fields"][0]["self"] == "unchanged"
    assert out["fields"][0]["status"] == "swapped"
    assert out["fields"][0]["net_delta"] == 0


def test_an_identical_resample_is_flagged_as_such_and_skips_labeling():
    """Budget spent on a field where the model repeats itself buys nothing.
    That has to be visible, not hidden inside `unchanged`."""
    calls = []
    original = {"a": "same"}
    paths, gold = {"a": ["a"]}, {"a": "right"}
    out = rg.evaluate_document(original, [("a", ["a"], 1)], [{"a": "same"}],
                                "first", _labeler_from(gold, paths, calls=calls))
    assert out["fields"][0]["status"] == "identical"
    assert out["fields"][0]["self"] == "unchanged"
    assert out["fields"][0]["net_delta"] == 0
    assert calls == []          # no swap happened, so nothing to re-label


def test_a_field_the_resample_lacks_is_unavailable():
    original = {"a": "wrong"}
    paths, gold = {"a": ["a"]}, {"a": "right"}
    out = rg.evaluate_document(original, [("a", ["a"], 1)], [{"z": 1}], "first",
                                _labeler_from(gold, paths))
    assert out["fields"][0]["status"] == "unavailable"
    assert out["fields"][0]["self"] == "unavailable"
    assert out["fields"][0]["net_delta"] == 0


def test_net_delta_charges_collateral_damage_to_the_swap():
    """Swapping one field can break a neighbour -- positional alignment inside
    an array, for instance. The budget must be charged for that, or selective
    regeneration would look better than it is."""
    original = {"items": [{"v": "wrong"}, {"v": "right_1"}]}
    paths = {"items.0.v": ["items", 0, "v"], "items.1.v": ["items", 1, "v"]}
    gold = {"items.0.v": "right_0", "items.1.v": "right_1"}
    scored = [("items.0.v", ["items", 0, "v"], 1), ("items.1.v", ["items", 1, "v"], 0)]

    def labeler(record):
        from probe_extraction.utils.jsonpath import json_get
        out = {}
        for ps, path in paths.items():
            _f, val = json_get(record, path)
            out[ps] = int(val != gold[ps])
        # The swap knocks the neighbour out of alignment.
        if json_get(record, ["items", 0, "v"])[1] == "right_0":
            out["items.1.v"] = 1
        return out

    out = rg.evaluate_document(original, scored, [{"items": [{"v": "right_0"},
                                                              {"v": "right_1"}]}],
                                "first", labeler)
    rec = out["fields"][0]
    assert rec["self"] == "repaired"      # the field itself was fixed ...
    assert rec["net_delta"] == 0          # ... but it cost a neighbour


def test_a_leaf_destroyed_by_the_swap_is_reported_as_lost():
    original = {"a": "wrong", "b": "right"}
    paths = {"a": ["a"], "b": ["b"]}
    gold = {"a": "right_a", "b": "right"}
    scored = [("a", ["a"], 1), ("b", ["b"], 0)]
    out = rg.evaluate_document(original, scored, [{"a": "right_a", "b": "right"}],
                                "first", _labeler_from(gold, paths, drop=("a",)))
    rec = out["fields"][0]
    assert rec["self"] == "lost"
    # A field that no longer exists is neither credited nor blamed: its
    # original label is carried forward.
    assert rec["net_delta"] == 0


def test_the_original_record_is_never_mutated():
    """Every budget builds its own hybrid from this same record; a shared
    mutation would let one field's swap leak into every later measurement."""
    original = {"a": "wrong", "b": "right"}
    paths = {"a": ["a"], "b": ["b"]}
    gold = {"a": "right_a", "b": "right"}
    scored = [("a", ["a"], 1), ("b", ["b"], 0)]
    rg.evaluate_document(original, scored, [{"a": "right_a", "b": "broken"}],
                          "first", _labeler_from(gold, paths))
    assert original == {"a": "wrong", "b": "right"}


def test_each_field_is_measured_against_the_original_not_the_previous_swap():
    """The per-field effects must be independent, since the curve sums them."""
    original = {"a": "wrong_a", "b": "wrong_b"}
    paths = {"a": ["a"], "b": ["b"]}
    gold = {"a": "right_a", "b": "right_b"}
    scored = [("a", ["a"], 1), ("b", ["b"], 1)]
    out = rg.evaluate_document(original, scored,
                                [{"a": "right_a", "b": "right_b"}], "first",
                                _labeler_from(gold, paths))
    assert [f["self"] for f in out["fields"]] == ["repaired", "repaired"]
    assert [f["net_delta"] for f in out["fields"]] == [-1, -1]


def test_joint_checkpoint_swaps_every_flagged_field_at_once():
    original = {"a": "wrong_a", "b": "wrong_b"}
    paths = {"a": ["a"], "b": ["b"]}
    gold = {"a": "right_a", "b": "right_b"}
    scored = [("a", ["a"], 1), ("b", ["b"], 1)]
    out = rg.evaluate_document(
        original, scored, [{"a": "right_a", "b": "right_b"}], "first",
        _labeler_from(gold, paths), joint_sets={"ck": ["a", "b"]})
    assert out["joint"]["ck"]["n_swapped"] == 2
    assert out["joint"]["ck"]["errors"] == 0       # both repaired together


def test_joint_checkpoint_with_nothing_flagged_keeps_the_baseline():
    original = {"a": "wrong_a"}
    paths, gold = {"a": ["a"]}, {"a": "right_a"}
    out = rg.evaluate_document(original, [("a", ["a"], 1)], [{"a": "right_a"}],
                                "first", _labeler_from(gold, paths),
                                joint_sets={"ck": []})
    assert out["joint"]["ck"] == {"n_swapped": 0, "errors": 1}


# ---------------------------------------------------------------------------
# The budget curve over measured outcomes
# ---------------------------------------------------------------------------

def _curve(y, net_delta, self_out, scores=None, budgets=(0.0, 0.5, 1.0),
           regime="global", doc_ids=None):
    y = np.asarray(y, dtype=np.int64)
    n = len(y)
    return s12.measured_curve(
        np.asarray(scores if scores is not None else y, dtype=np.float64),
        y,
        np.asarray(doc_ids if doc_ids is not None else np.zeros(n), dtype=np.int64),
        np.asarray(net_delta, dtype=np.int64),
        np.array(self_out, dtype=object),
        list(budgets), regime, rng=np.random.default_rng(0))


def test_zero_budget_leaves_the_error_rate_untouched():
    c = _curve([1, 0, 1, 0], [-1, 0, -1, 0], ["repaired", "unchanged"] * 2)
    row = c["rows"][0]
    assert row["budget"] == 0.0 and row["n_flagged"] == 0
    assert row["final_errors"] == 2
    assert row["final_error_rate"] == 0.5


def test_final_error_count_is_the_baseline_plus_the_flagged_deltas():
    y = [1, 1, 0, 0]
    net = [-1, -1, 1, 0]
    outs = ["repaired", "repaired", "damaged", "unchanged"]
    c = _curve(y, net, outs, budgets=(1.0,))
    row = c["rows"][0]
    assert row["n_flagged"] == 4
    assert row["final_errors"] == 2 + (-1 - 1 + 1 + 0)
    assert row["final_error_rate"] == pytest.approx(1 / 4)


def test_measured_rates_exclude_fields_with_no_regenerated_value():
    """An unavailable field was never regenerated. Leaving it in the
    denominator would silently deflate the measured repair rate."""
    y = [1, 1, 0, 0]
    net = [-1, 0, 1, 0]
    outs = ["repaired", "unavailable", "damaged", "unavailable"]
    row = _curve(y, net, outs, budgets=(1.0,))["rows"][0]
    assert row["measured_repair_rate"] == pytest.approx(1.0)   # 1 of 1 available
    assert row["measured_damage_rate"] == pytest.approx(1.0)   # 1 of 1 available
    assert row["outcomes"]["unavailable"] == 2
    assert row["n_available"] == 2


def test_outcome_counts_only_cover_the_flagged_fields():
    y = [1, 1]
    net = [-1, -1]
    outs = ["repaired", "repaired"]
    # scores put field 0 on top, so a 50% budget flags exactly one.
    row = _curve(y, net, outs, scores=[1.0, 0.0], budgets=(0.5,))["rows"][0]
    assert row["n_flagged"] == 1
    assert row["outcomes"]["repaired"] == 1
    assert row["final_errors"] == 1


def test_recall_and_precision_use_the_labels_before_regeneration():
    y = [1, 1, 0, 0]
    row = _curve(y, [0] * 4, ["unchanged"] * 4, scores=[1.0, 0.9, 0.8, 0.7],
                 budgets=(0.5,))["rows"][0]
    assert row["errors_caught"] == 2
    assert row["recall"] == pytest.approx(1.0)
    assert row["precision"] == pytest.approx(1.0)


def test_selective_risk_is_the_residual_error_rate_among_unflagged_fields():
    y = [1, 1, 0, 0]
    row = _curve(y, [0] * 4, ["unchanged"] * 4, scores=[1.0, 0.9, 0.8, 0.7],
                 budgets=(0.5,))["rows"][0]
    assert row["selective_risk"] == pytest.approx(0.0)


def test_error_rate_reduction_is_positive_when_repairs_outweigh_damage():
    y = [1, 1, 0, 0]
    net = [-1, -1, 1, 0]
    outs = ["repaired", "repaired", "damaged", "unchanged"]
    row = _curve(y, net, outs, budgets=(1.0,))["rows"][0]
    assert row["error_rate_reduction"] == pytest.approx(1 / 4)


def test_a_regeneration_that_damages_more_than_it_repairs_shows_a_loss():
    """The measurement has to be able to come out negative -- that is the whole
    point of measuring instead of assuming a 0.7 repair rate."""
    y = [1, 0, 0, 0]
    net = [0, 1, 1, 0]
    outs = ["unchanged", "damaged", "damaged", "unchanged"]
    row = _curve(y, net, outs, budgets=(1.0,))["rows"][0]
    assert row["final_errors"] == 3
    assert row["error_rate_reduction"] < 0


def test_per_doc_regime_spends_inside_each_document():
    y = [1, 0, 1, 0]
    row = _curve(y, [-1, 0, -1, 0], ["repaired", "unchanged"] * 2,
                 scores=[1.0, 0.0, 1.0, 0.0], budgets=(0.5,),
                 regime="per_doc", doc_ids=[0, 0, 1, 1])["rows"][0]
    assert row["n_flagged"] == 2
    assert row["outcomes"]["repaired"] == 2
    assert row["final_errors"] == 0

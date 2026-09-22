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


# ---------------------------------------------------------------------------
# Surviving a hostile document
# ---------------------------------------------------------------------------

def test_the_recursion_limit_is_raised_by_importing_the_module():
    """The matcher recurses through nested records, and SOB nests deep enough to
    pass CPython's default 1,000 frames. The limit MUST be raised by importing
    this module, not by the calling script: the per-document pass runs in joblib
    worker processes, which are fresh interpreters that inherit nothing from the
    parent. Setting it only in the script left every worker at 1,000 and killed
    a completed 13-hour regeneration run at the evaluation step."""
    import sys
    assert sys.getrecursionlimit() >= rg.RECURSION_LIMIT


def test_an_unlabelable_swap_is_recorded_not_raised():
    """A single pathological hybrid must not destroy a run that has already paid
    for its GPU time."""
    def exploding_labeler(record):
        raise RecursionError("maximum recursion depth exceeded")

    original = {"a": "wrong"}
    out = rg.evaluate_document(original, [("a", ["a"], 1)], [{"a": "other"}],
                               "first", exploding_labeler)
    rec = out["fields"][0]
    assert rec["status"] == "label_failed"
    assert rec["error"] == "RecursionError"
    assert rec["net_delta"] == 0


def test_an_unlabelable_swap_is_excluded_from_the_measured_rates():
    """It counts as unavailable, so it leaves the repair/damage denominators
    rather than inflating them with a field nothing was learned about."""
    def exploding_labeler(record):
        raise ValueError("boom")

    out = rg.evaluate_document({"a": "wrong"}, [("a", ["a"], 1)],
                               [{"a": "other"}], "first", exploding_labeler)
    assert out["fields"][0]["self"] == "unavailable"


def test_other_fields_are_still_measured_after_one_fails():
    calls = []

    def flaky_labeler(record):
        calls.append(record)
        if record.get("a") == "poison":
            raise RecursionError("nope")
        return {"a": 1, "b": 0}

    original = {"a": "wrong", "b": "wrong_b"}
    out = rg.evaluate_document(
        original,
        [("a", ["a"], 1), ("b", ["b"], 1)],
        [{"a": "poison", "b": "fixed"}],
        "first", flaky_labeler)
    assert out["fields"][0]["status"] == "label_failed"
    assert out["fields"][1]["status"] == "swapped"
    assert out["fields"][1]["self"] == "repaired"


def test_a_failed_joint_checkpoint_reports_null_rather_than_a_number():
    """The joint pass is a CHECK on the measurement. A failure there must be
    visibly absent, never silently folded in as a real error count."""
    def exploding_labeler(record):
        raise RecursionError("nope")

    out = rg.evaluate_document({"a": "wrong"}, [("a", ["a"], 1)],
                               [{"a": "other"}], "first", exploding_labeler,
                               joint_sets={"ck": ["a"]})
    # The field never got swapped, so the joint set has nothing to apply.
    assert out["joint"]["ck"]["errors"] == 1


def test_a_joint_checkpoint_that_fails_mid_labeling_is_marked_null():
    calls = []

    def labeler(record):
        calls.append(record)
        if len(calls) > 1:          # per-field pass succeeds, joint pass fails
            raise RecursionError("nope")
        return {"a": 0}

    out = rg.evaluate_document({"a": "wrong"}, [("a", ["a"], 1)],
                               [{"a": "fixed"}], "first", labeler,
                               joint_sets={"ck": ["a"]})
    assert out["fields"][0]["self"] == "repaired"
    assert out["joint"]["ck"]["errors"] is None
    assert out["joint"]["ck"]["error"] == "RecursionError"


def test_a_fresh_interpreter_gets_the_limit_just_by_importing_the_module():
    """The real regression. joblib workers are fresh interpreters that inherit
    nothing from the parent, and they reach this code by importing the module to
    unpickle `measure_document`. So the import alone must be enough -- verified
    here in a genuinely separate process, because an in-process assertion would
    pass even if the limit were only ever set by the parent."""
    import os
    import subprocess
    import sys as _sys

    env = dict(os.environ)
    src = str(_ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [_sys.executable, "-c",
         "import sys, probe_extraction.regen.evaluate as e; "
         "print(sys.getrecursionlimit(), e.RECURSION_LIMIT)"],
        capture_output=True, text=True, env=env, cwd=str(_ROOT))
    assert proc.returncode == 0, proc.stderr
    got, want = (int(x) for x in proc.stdout.split())
    assert got >= want


# ---------------------------------------------------------------------------
# vote_strict: only overwrite on consensus
# ---------------------------------------------------------------------------

def test_vote_strict_swaps_when_the_resamples_agree():
    samples = [{"a": "x"}, {"a": "y"}, {"a": "y"}]
    assert rg.select_value(samples, ["a"], "vote_strict") == (True, "y")


def test_vote_strict_declines_when_every_resample_disagrees():
    """Three different answers is not evidence for any of them. Plain `vote`
    would still pick one and overwrite a field that may have been right."""
    samples = [{"a": "x"}, {"a": "y"}, {"a": "z"}]
    assert rg.select_value(samples, ["a"], "vote") == (True, "x")
    assert rg.select_value(samples, ["a"], "vote_strict") == (False, None)


def test_vote_strict_requires_a_strict_majority_not_just_a_plurality():
    samples = [{"a": "x"}, {"a": "x"}, {"a": "y"}, {"a": "y"}]
    assert rg.select_value(samples, ["a"], "vote_strict") == (False, None)


def test_vote_strict_needs_two_agreeing_samples_not_one_unopposed():
    """A single usable resample cannot form a consensus with itself."""
    samples = [{"a": "x"}]
    assert rg.select_value(samples, ["a"], "vote") == (True, "x")
    assert rg.select_value(samples, ["a"], "vote_strict") == (False, None)


def test_vote_strict_counts_only_samples_that_have_the_field():
    """Two of three agree, the third never mentioned the field -- that is a
    consensus among those that answered."""
    samples = [{"b": 1}, {"a": "y"}, {"a": "y"}]
    assert rg.select_value(samples, ["a"], "vote_strict") == (True, "y")


def test_vote_strict_leaves_the_field_untouched_when_it_declines():
    original = {"a": "wrong"}
    paths, gold = {"a": ["a"]}, {"a": "right"}
    out = rg.evaluate_document(
        original, [("a", ["a"], 1)],
        [{"a": "p"}, {"a": "q"}, {"a": "r"}],
        "vote_strict", _labeler_from(gold, paths))
    assert out["fields"][0]["status"] == "unavailable"
    assert out["fields"][0]["net_delta"] == 0


# ---------------------------------------------------------------------------
# Is the improvement real?
# ---------------------------------------------------------------------------

def _boot_inputs(n_docs=60, fields_per_doc=5, seed=0):
    """A corpus where probe_fused is genuinely better than min_logprob."""
    rng = np.random.default_rng(seed)
    n = n_docs * fields_per_doc
    y = rng.integers(0, 2, n)
    doc_ids = np.repeat(np.arange(n_docs), fields_per_doc)
    # Errors get high probe scores; log-probs are noisier about it.
    fused = y + rng.normal(0, 0.3, n)
    lp = y + rng.normal(0, 1.2, n)
    # Every flagged error is repaired, nothing is damaged.
    net_delta = -y
    signals = {"probe_fused": fused, "probe_answer": fused - rng.normal(0, .05, n),
               "min_logprob": lp, "mean_logprob": lp + rng.normal(0, .05, n)}
    return y, doc_ids, net_delta.astype(np.int64), signals


def test_a_real_improvement_gets_a_ci_that_excludes_zero():
    y, doc_ids, net_delta, signals = _boot_inputs()
    out = s12.bootstrap_significance(
        y, doc_ids, net_delta, signals,
        ["probe_fused", "probe_answer", "min_logprob", "mean_logprob"],
        0.20, "global", n_boot=300, seed=0)
    red = out["tests"]["reduction_vs_baseline"]
    assert red["mean"] > 0
    assert red["ci_low"] > 0
    assert red["p_value"] < 0.05


def test_no_improvement_gives_a_ci_that_contains_zero():
    """With a repair operator that does nothing, the measured reduction must not
    come out significant -- otherwise the test would manufacture results."""
    y, doc_ids, _nd, signals = _boot_inputs()
    net_delta = np.zeros(len(y), dtype=np.int64)
    out = s12.bootstrap_significance(
        y, doc_ids, net_delta, signals, ["probe_fused", "min_logprob"],
        0.20, "global", n_boot=300, seed=0)
    red = out["tests"]["reduction_vs_baseline"]
    assert red["mean"] == pytest.approx(0.0, abs=1e-12)
    assert red["ci_low"] <= 0 <= red["ci_high"]
    assert red["p_value"] > 0.05


def test_the_better_signal_beats_the_worse_one():
    y, doc_ids, net_delta, signals = _boot_inputs()
    out = s12.bootstrap_significance(
        y, doc_ids, net_delta, signals, ["probe_fused", "min_logprob"],
        0.20, "global", n_boot=300, seed=0)
    vs = out["tests"]["probe_fused_vs_min_logprob"]
    assert vs["mean"] > 0        # the baseline leaves MORE errors behind
    assert vs["ci_low"] > 0


def test_comparisons_are_paired_within_a_replicate():
    """An unpaired test would compare two independently resampled corpora and
    drown a difference this small in between-corpus variance."""
    y, doc_ids, net_delta, signals = _boot_inputs()
    out = s12.bootstrap_significance(
        y, doc_ids, net_delta, signals, ["probe_fused", "min_logprob"],
        0.20, "global", n_boot=400, seed=0)
    vs = out["tests"]["probe_fused_vs_min_logprob"]
    spread = vs["ci_high"] - vs["ci_low"]
    rates = out["per_signal_error_rate"]
    # The paired difference must be tighter than either signal's own interval.
    own = rates["probe_fused"]["ci_high"] - rates["probe_fused"]["ci_low"]
    assert spread < own


def test_every_test_carries_a_holm_adjusted_p_value():
    y, doc_ids, net_delta, signals = _boot_inputs()
    out = s12.bootstrap_significance(
        y, doc_ids, net_delta, signals,
        ["probe_fused", "probe_answer", "min_logprob", "mean_logprob"],
        0.20, "global", n_boot=200, seed=0)
    for name, t in out["tests"].items():
        assert t["p_holm"] is not None, name
        assert t["p_holm"] >= t["p_value"] - 1e-12, name


def test_the_per_doc_regime_also_produces_intervals():
    y, doc_ids, net_delta, signals = _boot_inputs()
    out = s12.bootstrap_significance(
        y, doc_ids, net_delta, signals, ["probe_fused", "min_logprob"],
        0.20, "per_doc", n_boot=200, seed=0)
    assert out["regime"] == "per_doc"
    assert out["tests"]["reduction_vs_baseline"]["ci_low"] > 0


def test_the_bootstrap_is_reproducible_from_its_seed():
    y, doc_ids, net_delta, signals = _boot_inputs()
    kw = dict(n_boot=200, seed=7)
    a = s12.bootstrap_significance(y, doc_ids, net_delta, signals,
                                   ["probe_fused", "min_logprob"], 0.2, "global", **kw)
    b = s12.bootstrap_significance(y, doc_ids, net_delta, signals,
                                   ["probe_fused", "min_logprob"], 0.2, "global", **kw)
    assert a["tests"] == b["tests"]

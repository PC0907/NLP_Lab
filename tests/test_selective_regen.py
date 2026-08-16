"""Tests for Stage 9's selective-regeneration curves (scripts/09_...py).

These curves are the project's actual deliverable (cost vs quality), so the
budget accounting has to be exactly right: how many fields get flagged, how many
errors that catches, what risk is left behind, and what the simulated repair
does to the final error rate. Oracle and random signals give us known-answer
anchors to test against.
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


s9 = _load("stage09", "scripts/09_selective_regeneration_sob.py")


def _rng(seed: int = 0):
    return np.random.default_rng(seed)


# --- flagging --------------------------------------------------------------

def test_global_budget_flags_the_right_count():
    scores = np.linspace(0, 1, 100)
    for b, k in [(0.0, 0), (0.05, 5), (0.2, 20), (0.5, 50), (1.0, 100)]:
        assert s9._flag_global(scores, b, _rng()).sum() == k


def test_global_budget_takes_the_highest_scores():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    flagged = s9._flag_global(scores, 0.5, _rng())
    assert flagged.tolist() == [False, True, False, True]


def test_global_ties_are_broken_randomly_not_by_position():
    """A degenerate all-equal score must behave like chance. If ties fell back
    to array order it would inherit document ordering and look better than it is."""
    scores = np.zeros(200)
    picks = {tuple(np.flatnonzero(s9._flag_global(scores, 0.1, _rng(s)))) for s in range(5)}
    assert len(picks) > 1


def test_per_doc_budget_spends_within_each_document():
    doc_ids = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.4, 0.8])
    flagged = s9._flag_per_doc(scores, doc_ids, 0.5, _rng())
    # 2 of doc 0's 4 fields, 1 of doc 1's 2 fields -- its top scorers.
    assert flagged[:4].sum() == 2 and flagged[4:].sum() == 1
    assert flagged[3] and flagged[5]


def test_per_doc_budget_rounds_up_so_a_small_budget_still_acts():
    """You cannot regenerate a fraction of a field: any non-zero budget must
    reach at least one field per document, or small documents get ignored."""
    doc_ids = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.5, 0.9, 0.2, 0.4, 0.8])
    flagged = s9._flag_per_doc(scores, doc_ids, 0.01, _rng())
    assert flagged[:3].sum() == 1 and flagged[3:].sum() == 1


def test_per_doc_budget_endpoints():
    doc_ids = np.array([0, 0, 1, 1, 1])
    scores = np.arange(5, dtype=float)
    assert s9._flag_per_doc(scores, doc_ids, 0.0, _rng()).sum() == 0
    assert s9._flag_per_doc(scores, doc_ids, 1.0, _rng()).all()


# --- curve metrics ---------------------------------------------------------

def _curve(scores, y, doc_ids, budgets, regime="global", repair=(1.0,), damage=0.0):
    return s9.curve(scores, y, doc_ids, budgets, regime,
                    repair_rates=list(repair), damage_rate=damage, rng=_rng())


def test_oracle_catches_every_error_once_the_budget_covers_them():
    y = np.array([1] * 20 + [0] * 80)
    scores = y.astype(float)
    doc_ids = np.zeros(100, dtype=int)
    rows = _curve(scores, y, doc_ids, [0.2])["rows"]
    r = rows[0]
    assert r["n_flagged"] == 20
    assert r["recall"] == pytest.approx(1.0)
    assert r["precision"] == pytest.approx(1.0)
    assert r["selective_risk"] == pytest.approx(0.0)


def test_random_signal_catches_about_the_budget_share_of_errors():
    rng = _rng(3)
    y = (rng.random(4000) < 0.3).astype(int)
    rows = _curve(rng.random(4000), y, np.zeros(4000, dtype=int), [0.25])["rows"]
    assert rows[0]["recall"] == pytest.approx(0.25, abs=0.06)


def test_selective_risk_is_the_error_rate_among_unflagged_fields():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0])
    scores = np.array([9.0, 8.0, 0, 0, 0, 0, 0, 0, 0.5, 0])
    rows = _curve(scores, y, np.zeros(10, dtype=int), [0.2])["rows"]
    r = rows[0]
    assert r["errors_caught"] == 2
    assert r["selective_risk"] == pytest.approx(1 / 8)  # one error left in 8 kept


def test_full_budget_leaves_no_residual_risk():
    y = np.array([1, 0, 1, 0])
    rows = _curve(np.arange(4, dtype=float), y, np.zeros(4, dtype=int), [1.0])["rows"]
    assert rows[0]["recall"] == pytest.approx(1.0)
    assert rows[0]["selective_risk"] == pytest.approx(0.0)


def test_zero_budget_changes_nothing():
    y = np.array([1, 0, 1, 0])
    rows = _curve(np.arange(4, dtype=float), y, np.zeros(4, dtype=int), [0.0])["rows"]
    r = rows[0]
    assert r["n_flagged"] == 0 and r["recall"] == pytest.approx(0.0)
    assert r["precision"] is None
    assert r["selective_risk"] == pytest.approx(0.5)
    assert r["break_even_repair_rate"] is None


# --- the repair simulation -------------------------------------------------

def test_perfect_repair_removes_exactly_the_caught_errors():
    y = np.array([1] * 20 + [0] * 80)
    rows = _curve(y.astype(float), y, np.zeros(100, dtype=int), [0.1],
                  repair=(1.0,), damage=0.0)["rows"]
    # 10 of 20 errors caught and fixed -> 10 errors left in 100 fields.
    assert rows[0]["final_error_rate"]["1"] == pytest.approx(0.10)


def test_partial_repair_only_fixes_its_share():
    y = np.array([1] * 20 + [0] * 80)
    rows = _curve(y.astype(float), y, np.zeros(100, dtype=int), [0.2],
                  repair=(0.5,), damage=0.0)["rows"]
    # All 20 caught, half repaired -> 10 remain.
    assert rows[0]["final_error_rate"]["0.5"] == pytest.approx(0.10)


def test_damage_to_correct_fields_is_charged_against_the_budget():
    """Regenerating a field that was already right can break it -- the effect
    that made blanket regeneration net-negative in the ExtractBench track."""
    y = np.array([1] * 10 + [0] * 90)
    rows = _curve(y.astype(float), y, np.zeros(100, dtype=int), [0.5],
                  repair=(1.0,), damage=0.5)["rows"]
    r = rows[0]
    # 50 flagged: 10 real errors (all fixed) + 40 correct, half of them broken.
    assert r["errors_caught"] == 10
    assert r["final_error_rate"]["1"] == pytest.approx(20 / 100)


def test_break_even_repair_rate_matches_its_definition():
    y = np.array([1] * 10 + [0] * 90)
    rows = _curve(y.astype(float), y, np.zeros(100, dtype=int), [0.5],
                  repair=(1.0,), damage=0.1)["rows"]
    # 10 caught, 40 needless regenerations at 10% damage -> need pi >= 0.4.
    assert rows[0]["break_even_repair_rate"] == pytest.approx(0.4)


def test_break_even_is_zero_when_the_budget_is_spent_perfectly():
    y = np.array([1] * 20 + [0] * 80)
    rows = _curve(y.astype(float), y, np.zeros(100, dtype=int), [0.2],
                  repair=(1.0,), damage=0.2)["rows"]
    assert rows[0]["break_even_repair_rate"] == pytest.approx(0.0)


# --- AURC ------------------------------------------------------------------

def test_a_better_signal_has_lower_aurc():
    rng = _rng(7)
    y = (rng.random(1000) < 0.3).astype(int)
    doc_ids = np.zeros(1000, dtype=int)
    budgets = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    good = y + rng.normal(0, 0.3, 1000)        # informative
    noise = rng.random(1000)                   # uninformative
    a_good = _curve(good, y, doc_ids, budgets)["aurc"]
    a_noise = _curve(noise, y, doc_ids, budgets)["aurc"]
    a_oracle = _curve(y.astype(float), y, doc_ids, budgets)["aurc"]
    assert a_oracle < a_good < a_noise

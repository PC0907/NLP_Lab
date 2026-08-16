"""Tests for Stage 8's controls (scripts/08_attribution_controls.py).

The controls only mean something if they change exactly what they claim to
change: same shape, same answer block, same layer -- only the reasoning block's
content differs. These tests pin that down, plus the Holm correction and the
bootstrap CI that the paper's significance claims rest on.
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


s8 = _load("stage08", "scripts/08_attribution_controls.py")


def _doc(n_fields: int = 4, dim: int = 6, layer: int = 19, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "doc_id": "d0",
        "y": rng.integers(0, 2, n_fields),
        "answer": {layer: rng.standard_normal((n_fields, dim)).astype(np.float32)},
        "attr": {layer: rng.standard_normal((n_fields, dim)).astype(np.float32)},
        "scalars": rng.standard_normal((n_fields, 3)).astype(np.float32),
    }


# --- feature construction --------------------------------------------------

def test_all_fused_variants_share_one_shape():
    """Dimensionality is the thing being controlled for -- if the controls were
    a different width, the comparison would be confounded by feature count."""
    d = _doc()
    rng = np.random.default_rng(0)
    shapes = {v: s8.build_control_features(d, 19, v, rng).shape
              for v in ("fused_attr", "ctrl_docmean", "ctrl_shuffled", "ctrl_random")}
    assert len(set(shapes.values())) == 1
    assert shapes["fused_attr"] == (4, 12)  # answer(6) + reasoning block(6)


def test_answer_variant_is_the_bare_answer_block():
    d = _doc()
    out = s8.build_control_features(d, 19, "answer", np.random.default_rng(0))
    np.testing.assert_allclose(out, d["answer"][19])


def test_every_variant_preserves_the_answer_block_untouched():
    d = _doc()
    rng = np.random.default_rng(0)
    for v in ("fused_attr", "ctrl_docmean", "ctrl_shuffled", "ctrl_random"):
        out = s8.build_control_features(d, 19, v, rng)
        np.testing.assert_allclose(out[:, :6], d["answer"][19], atol=1e-6)


def test_docmean_control_is_constant_within_the_document():
    """This is the whole point of the control: it rebuilds Stage 6's doc-level
    pooling out of the attribution tokens, so it cannot re-rank fields."""
    d = _doc()
    out = s8.build_control_features(d, 19, "ctrl_docmean", np.random.default_rng(0))
    block = out[:, 6:]
    assert np.allclose(block, block[0])
    np.testing.assert_allclose(block[0], d["attr"][19].mean(axis=0), atol=1e-6)


def test_shuffled_control_is_a_permutation_of_the_same_vectors():
    """Same multiset of reasoning vectors, attached to the wrong fields."""
    d = _doc(n_fields=5, seed=3)
    out = s8.build_control_features(d, 19, "ctrl_shuffled", np.random.default_rng(1))
    block = out[:, 6:]
    orig = sorted(map(tuple, np.round(d["attr"][19], 5).tolist()))
    got = sorted(map(tuple, np.round(block, 5).tolist()))
    assert orig == got


def test_shuffled_control_actually_moves_rows():
    d = _doc(n_fields=8, seed=5)
    moved = False
    for s in range(5):
        block = s8.build_control_features(d, 19, "ctrl_shuffled",
                                          np.random.default_rng(s))[:, 6:]
        if not np.allclose(block, d["attr"][19]):
            moved = True
            break
    assert moved, "shuffle never permuted an 8-field document"


def test_single_field_document_survives_every_control():
    """A 1-field doc cannot be shuffled and has a doc-mean equal to itself.
    It must not crash -- such docs are simply uninformative for the paired test."""
    d = _doc(n_fields=1)
    for v in ("fused_attr", "ctrl_docmean", "ctrl_shuffled", "ctrl_random"):
        out = s8.build_control_features(d, 19, v, np.random.default_rng(0))
        assert out.shape == (1, 12)


def test_random_control_matches_the_attr_block_scale():
    d = _doc(n_fields=400, dim=3, seed=11)
    d["attr"][19] = (np.random.default_rng(2).standard_normal((400, 3)) * 10).astype(np.float32)
    out = s8.build_control_features(d, 19, "ctrl_random", np.random.default_rng(0))
    assert np.allclose(out[:, 3:].std(axis=0), 10, rtol=0.25)


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError):
        s8.build_control_features(_doc(), 19, "nope", np.random.default_rng(0))


# --- statistics ------------------------------------------------------------

def test_holm_is_step_down_and_monotone():
    adj = s8.holm_bonferroni({"a": 0.01, "b": 0.02, "c": 0.04})
    assert adj["a"] == pytest.approx(0.03)   # 3 * 0.01
    assert adj["b"] == pytest.approx(0.04)   # 2 * 0.02
    assert adj["c"] == pytest.approx(0.04)   # 1 * 0.04, raised to stay monotone
    assert adj["a"] <= adj["b"] <= adj["c"]


def test_holm_caps_at_one_and_skips_missing_pvalues():
    adj = s8.holm_bonferroni({"a": 0.9, "b": 0.95, "c": None})
    assert "c" not in adj
    assert all(v <= 1.0 for v in adj.values())
    assert s8.holm_bonferroni({}) == {}


def test_bootstrap_ci_brackets_a_real_positive_shift():
    n = 120
    a = list(np.linspace(0.70, 0.80, n))
    b = [x + 0.02 for x in a]
    ci = s8.bootstrap_delta_ci(a, b, n_boot=500, seed=0)
    assert ci["mean"] == pytest.approx(0.02, abs=1e-6)
    assert ci["ci_low"] > 0
    assert ci["frac_boot_positive"] == 1.0
    assert ci["n_pairs"] == n


def test_bootstrap_ci_straddles_zero_when_there_is_no_effect():
    rng = np.random.default_rng(0)
    a = list(rng.random(200))
    b = list(rng.random(200))
    ci = s8.bootstrap_delta_ci(a, b, n_boot=500, seed=1)
    assert ci["ci_low"] < 0 < ci["ci_high"]


def test_bootstrap_ci_ignores_unpaired_documents():
    a = [0.5, None, 0.6, 0.7, 0.8, 0.9]
    b = [0.6, 0.9, None, 0.8, 0.9, 1.0]
    ci = s8.bootstrap_delta_ci(a, b, n_boot=200, seed=0)
    assert ci is None or ci["n_pairs"] == 4


def test_bootstrap_declines_on_too_few_pairs():
    assert s8.bootstrap_delta_ci([0.5, 0.6], [0.6, 0.7], n_boot=100) is None
    assert s8.bootstrap_delta_ci([0.5] * 10, [0.6] * 10, n_boot=0) is None


def test_average_per_doc_requires_every_seed_to_be_valid():
    merged = s8.average_per_doc([[0.6, 0.8, None], [0.8, None, 0.5]])
    assert merged[0] == pytest.approx(0.7)
    assert merged[1] is None   # invalid in the second seed
    assert merged[2] is None   # invalid in the first seed

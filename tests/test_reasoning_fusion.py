"""Tests for the reasoning-fusion LODO harness (scripts/06_reasoning_fusion_lodo.py).

Loaded via spec (it's a script, not a package module). Validates feature
construction and the two-metric behaviour: doc-level reasoning fusion should
help the POOLED out-of-fold AUROC (global ranking, used by selective
regeneration) but be ~null on per-document AUROC (within-doc ranking), since the
reasoning vector is constant across a document's fields.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_MOD = _ROOT / "scripts" / "06_reasoning_fusion_lodo.py"
_spec = importlib.util.spec_from_file_location("reasoning_fusion_iso", _MOD)
rf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rf)

L = 18


def _make_docs(n_docs=14, n_fields=24, seed=0):
    """Synthetic docs: answer features are pure noise (no per-field signal);
    the reasoning vector encodes the document's error propensity `r`. So global
    ranking is recoverable from reasoning, within-doc ranking is not."""
    rng = np.random.default_rng(seed)
    docs = []
    for k in range(n_docs):
        r = rng.uniform(0.15, 0.85)
        y = (rng.uniform(size=n_fields) < r).astype(np.int64)
        answer = rng.normal(size=(n_fields, 8))                 # noise
        rvec = np.concatenate([[r], rng.normal(0, 0.01, 3)]).astype(np.float64)
        docs.append({
            "doc_id": f"d{k}", "y": y,
            "answer": {L: answer},
            "reasoning_mean": {L: rvec}, "reasoning_last": {L: rvec.copy()},
        })
    return docs


def test_build_features_shapes():
    d = _make_docs(n_docs=2)[0]
    assert rf.build_features(d, L, "answer").shape[1] == 8
    assert rf.build_features(d, L, "fused_mean").shape[1] == 8 + 4
    assert rf.build_features(d, L, "fused_both").shape[1] == 8 + 4 + 4


def test_pooled_oof_benefits_from_reasoning():
    docs = _make_docs()
    ans = rf.lodo_eval(docs, L, "answer")
    fused = rf.lodo_eval(docs, L, "fused_mean")
    assert ans["pooled_oof_auroc"] is not None
    assert fused["pooled_oof_auroc"] is not None
    # answer is noise -> ~chance globally; reasoning encodes doc risk -> helps.
    assert ans["pooled_oof_auroc"] < 0.65
    assert fused["pooled_oof_auroc"] > ans["pooled_oof_auroc"] + 0.10


def test_per_doc_auroc_not_helped_by_doc_level_reasoning():
    docs = _make_docs()
    ans = rf.lodo_eval(docs, L, "answer")
    fused = rf.lodo_eval(docs, L, "fused_mean")
    assert ans["per_doc_auroc_mean"] is not None
    assert fused["per_doc_auroc_mean"] is not None
    # Within-doc ranking cannot improve from a doc-constant feature.
    assert abs(fused["per_doc_auroc_mean"] - ans["per_doc_auroc_mean"]) < 0.10

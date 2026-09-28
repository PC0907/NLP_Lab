"""Tests for reasoning-trace boundary detection and pooling.

reasoning_trace.py is numpy-only and imported in isolation (via spec) so these
run on CPU without loading the torch/transformers model stack.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_MOD_PATH = _ROOT / "src" / "probe_extraction" / "extraction" / "reasoning_trace.py"
_spec = importlib.util.spec_from_file_location("reasoning_trace_iso", _MOD_PATH)
rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rt)


# ---------------------------------------------------------------------------
# find_reasoning_end_token
# ---------------------------------------------------------------------------

def test_find_end_single_marker_token():
    toks = ["<think>", " reasoning ", "here", "</think>", "{", '"a"', ":", "1", "}"]
    assert rt.find_reasoning_end_token(toks) == 4  # up to & incl. "</think>"


def test_find_end_marker_split_across_tokens():
    # </think> tokenised as several pieces — completion detected on the last.
    toks = ["<think>", "x", "</", "think", ">", "{", '"a"']
    assert rt.find_reasoning_end_token(toks) == 5


def test_find_end_no_marker_returns_zero():
    toks = ["{", '"a"', ":", "1", "}"]  # no reasoning trace (non-reasoning model)
    assert rt.find_reasoning_end_token(toks) == 0


# ---------------------------------------------------------------------------
# reasoning_pooled_vectors
# ---------------------------------------------------------------------------

def test_pool_mean_and_last():
    arr = np.arange(6 * 4, dtype=np.float16).reshape(6, 4)  # 6 tokens, dim 4
    hs = {18: arr}
    out = rt.reasoning_pooled_vectors(hs, reasoning_end=4)
    assert set(out) == {"reasoning_mean", "reasoning_last"}
    # mean over the first 4 rows
    np.testing.assert_allclose(
        out["reasoning_mean"][18].astype(np.float32),
        arr[:4].astype(np.float32).mean(axis=0), rtol=1e-3,
    )
    # last reasoning token = row index 3 (the </think> position)
    np.testing.assert_array_equal(out["reasoning_last"][18], arr[3])


def test_pool_clamps_to_available_rows():
    arr = np.ones((3, 4), dtype=np.float16)
    out = rt.reasoning_pooled_vectors({1: arr}, reasoning_end=10)  # > n rows
    assert out["reasoning_last"][1].shape == (4,)


def test_pool_empty_when_no_reasoning():
    arr = np.ones((5, 4), dtype=np.float16)
    assert rt.reasoning_pooled_vectors({1: arr}, reasoning_end=0) == {}
    assert rt.reasoning_pooled_vectors({}, reasoning_end=4) == {}

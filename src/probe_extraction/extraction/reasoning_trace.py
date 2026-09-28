"""Reasoning-trace pooling for reasoning models (DeepSeek-R1 etc.).

A reasoning model emits `<think> ... </think>` before the JSON answer. The
project's probes traditionally read activations only at the *answer* token of
each field. This module extracts a complementary signal from the *reasoning
trace*: it locates the `</think>` boundary in the generated token stream and
pools the reasoning-token activations (already captured by the model wrapper
over the full generated sequence) into compact per-layer summary vectors.

Kept dependency-light (numpy only, no torch / no package imports) so the
pooling logic is unit-testable on CPU without loading the model stack.
"""

from __future__ import annotations

import numpy as np

_THINK_END = "</think>"


def find_reasoning_end_token(per_token_strings: list[str]) -> int:
    """Number of leading generated tokens that form the reasoning trace.

    Reconstructs the surface text token-by-token and returns the count of
    tokens up to and including the one that completes `</think>`. The marker
    may be split across several BPE tokens (e.g. ["</", "think", ">"]); this
    detects completion regardless of tokenisation.

    Returns 0 if no `</think>` is present (non-reasoning model) — the caller
    then captures no reasoning activations, leaving non-reasoning pipelines
    unchanged.
    """
    acc = ""
    for i, s in enumerate(per_token_strings):
        acc += s
        if _THINK_END in acc:
            return i + 1
    return 0


def reasoning_pooled_vectors(
    hidden_states: dict[int, np.ndarray],
    reasoning_end: int,
) -> dict[str, dict[int, np.ndarray]]:
    """Pool the reasoning-trace activations into per-layer summary vectors.

    Args:
        hidden_states: {layer: (n_generated, hidden_dim)} captured over the
            FULL generated sequence (the reasoning trace occupies the leading
            positions, before the JSON answer).
        reasoning_end: number of leading tokens belonging to the reasoning
            trace (from find_reasoning_end_token).

    Returns:
        {"reasoning_mean": {layer: vec}, "reasoning_last": {layer: vec}} with
        vec of shape (hidden_dim,) float16. Empty dict if reasoning_end <= 0
        or no hidden states were captured.
    """
    if reasoning_end <= 0 or not hidden_states:
        return {}
    mean_d: dict[int, np.ndarray] = {}
    last_d: dict[int, np.ndarray] = {}
    for layer, arr in hidden_states.items():
        end = min(reasoning_end, arr.shape[0])
        if end <= 0:
            continue
        span = arr[:end]
        mean_d[layer] = span.astype(np.float32).mean(axis=0).astype(np.float16)
        last_d[layer] = arr[end - 1]  # the </think> position: a reasoning summary
    out: dict[str, dict[int, np.ndarray]] = {}
    if mean_d:
        out["reasoning_mean"] = mean_d
    if last_d:
        out["reasoning_last"] = last_d
    return out

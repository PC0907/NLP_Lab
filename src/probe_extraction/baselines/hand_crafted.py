"""Hand-crafted surface-feature baseline (deflationary control).

Trains logistic regression on simple observable features of each field --
no hidden states, no logprobs. If this matches the probe, the hidden-state
claim is weak; if it stays below, the probe's advantage is genuine.
Evaluated under LODO (it is trained) for comparability.
"""
from __future__ import annotations

import numpy as np

from probe_extraction.baselines.lodo_eval import lodo_evaluate
from probe_extraction.baselines.token_logprob import BaselineMetrics


def build_handcrafted_features(meta: list[dict]) -> np.ndarray:
    """One feature row per field from its path and (if available) value.

    NOTE: 04_evaluate.py's meta currently carries {doc_id, path_str} only.
    To enable the value-based features, extend the meta.append(...) in
    build_evaluation_dataset to also store "value": lab.get("extracted_value").
    Without that, value features fall back to 0 (path-structure-only control).
    """
    rows = []
    for m in meta:
        path_str = m.get("path_str", "")
        val = m.get("value", None)
        s = "" if val is None else str(val)
        depth = path_str.count(".")
        rows.append([
            float(len(s)),                          # char length
            float(len(s.split())),                  # word count
            float(depth),                           # nesting depth
            float(any(c.isdigit() for c in s)),     # has digit
            float(any(c.isupper() for c in s)),     # has uppercase
            float(s.count(",")),                    # comma count
            float(len(path_str)),                   # path length (field-type proxy)
        ])
    return np.asarray(rows, dtype=np.float32)


def evaluate_handcrafted(*, meta: list[dict], y: np.ndarray,
                         C: float = 1.0) -> BaselineMetrics:
    X = build_handcrafted_features(meta)
    doc_ids = [m["doc_id"] for m in meta]
    return lodo_evaluate(X=X, y=y, doc_ids=doc_ids, name="hand_crafted", C=C)
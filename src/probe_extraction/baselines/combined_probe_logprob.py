"""Combined probe + logprob baseline.

Concatenates the hidden-state activation (best probe layer) with the output-
level logprob features (mean, min), trains logistic regression under LODO.
Tests complementarity: if combined > probe-alone, internal and output signals
carry different information.
"""
from __future__ import annotations

import numpy as np

from probe_extraction.baselines.lodo_eval import lodo_evaluate
from probe_extraction.baselines.token_logprob import BaselineMetrics


def evaluate_combined(*, activations: np.ndarray, mean_logprob: np.ndarray,
                      min_logprob: np.ndarray, y: np.ndarray,
                      doc_ids: list[str], layer: int,
                      C: float = 1.0) -> BaselineMetrics:
    lp = np.stack([mean_logprob, min_logprob], axis=1).astype(np.float32)
    X = np.concatenate([activations.astype(np.float32), lp], axis=1)
    return lodo_evaluate(X=X, y=y, doc_ids=doc_ids,
                         name=f"combined_probe_logprob_layer{layer}", C=C)
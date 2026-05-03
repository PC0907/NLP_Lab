"""Token log-probability baselines.

Uses the model's own per-token log-probabilities as a signal for whether
a field is correct. The intuition: when generating fields the model is
uncertain about (or making up), it tends to use lower-probability tokens.

We compute two scalar features per field:
  - mean_logprob: average logprob across the field's tokens
  - min_logprob: minimum logprob (single most-uncertain token)

Each is then evaluated as a standalone error-detection signal: the
"prediction" is is_error = (low score), and we compute AUROC against
the same labels the probe was trained on.

Note on signs: log-probabilities are negative numbers (≤0). Higher (closer
to zero) = more confident. Lower (more negative) = less confident. So a
LOW logprob should correlate with HIGH error probability. We negate when
needed so AUROC is computed in a consistent direction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class BaselineMetrics:
    """Metrics for one baseline scoring function."""

    name: str
    n_samples: int
    n_errors: int
    auroc: float
    auprc: float


def compute_token_logprob_scores(
    *,
    token_logprobs: list[float],
    token_span: tuple[int, int],
) -> dict[str, float]:
    """Compute per-field logprob statistics.

    Args:
        token_logprobs: Per-generated-token log-probabilities for the
            entire document's generation. Length = number of generated tokens.
        token_span: (start, end) — the field's token range, end-exclusive.

    Returns:
        Dict with keys "mean_logprob" and "min_logprob". If the span is
        empty or out of range, returns NaN values (caller should skip).
    """
    start, end = token_span
    n = len(token_logprobs)
    start = max(0, min(start, n))
    end = max(start, min(end, n))

    if end <= start:
        return {"mean_logprob": float("nan"), "min_logprob": float("nan")}

    span_lp = token_logprobs[start:end]
    return {
        "mean_logprob": float(np.mean(span_lp)),
        "min_logprob": float(np.min(span_lp)),
    }


def evaluate_baseline(
    *,
    scores: np.ndarray,
    y: np.ndarray,
    name: str,
    score_higher_is_error: bool,
) -> BaselineMetrics:
    """Compute AUROC/AUPRC for a single scalar baseline.

    Args:
        scores: (n,) raw scalar scores (e.g., mean logprob).
        y: (n,) binary labels (1 = error).
        name: Identifier for the baseline.
        score_higher_is_error: Whether higher score → more likely error.
            For logprobs (lower confidence = more error-prone), set False
            so we negate before scoring.

    Returns:
        BaselineMetrics with AUROC, AUPRC, sample counts.
    """
    # Filter out NaNs (fields with empty token spans).
    mask = ~np.isnan(scores)
    scores = scores[mask]
    y = y[mask]

    n = len(y)
    n_errors = int(y.sum())

    if n_errors == 0 or n_errors == n:
        logger.warning(
            "Baseline %s: only one class in evaluation set; AUROC undefined.",
            name,
        )
        return BaselineMetrics(
            name=name, n_samples=n, n_errors=n_errors,
            auroc=float("nan"), auprc=float("nan"),
        )

    # Convert to "prediction of error": higher = more likely error.
    if score_higher_is_error:
        pred = scores
    else:
        pred = -scores

    auroc = float(roc_auc_score(y, pred))
    auprc = float(average_precision_score(y, pred))

    return BaselineMetrics(
        name=name, n_samples=n, n_errors=n_errors,
        auroc=auroc, auprc=auprc,
    )
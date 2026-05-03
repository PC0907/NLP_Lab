"""Black-box baselines for per-field error detection.

These compare against probe-based signals using only data the model
already exposes: token log-probabilities, agreement across samples, etc.
No hidden-state access required.
"""

from probe_extraction.baselines.token_logprob import (
    BaselineMetrics,
    compute_token_logprob_scores,
    evaluate_baseline,
)

__all__ = [
    "BaselineMetrics",
    "compute_token_logprob_scores",
    "evaluate_baseline",
]
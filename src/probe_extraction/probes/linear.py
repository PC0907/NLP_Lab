"""Linear probe (logistic regression) on LLM hidden states.

Wraps sklearn LogisticRegression with the metadata the project needs:
which layer it was trained on, which fields it was trained on, and
evaluation metrics (AUROC, calibration, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


@dataclass
class ProbeMetrics:
    """Evaluation metrics for a single probe."""

    layer: int
    n_train: int
    n_test: int
    n_test_errors: int
    auroc: float
    auprc: float
    brier: float
    threshold_at_50pct_recall: float | None
    accuracy_at_default_threshold: float
    per_fold_auroc: list[float] = field(default_factory=list)


@dataclass
class LinearProbe:
    """Trained linear probe + metadata.

    Stored on disk as a pickle. Light enough to keep many around.
    """

    layer: int
    weights: np.ndarray            # (hidden_dim,)
    bias: float
    classes: np.ndarray             # [0, 1]
    n_train: int
    metrics: ProbeMetrics
    config: dict[str, Any] = field(default_factory=dict)

    def score(self, activations: np.ndarray) -> np.ndarray:
        """Return probability of error for each row of activations.

        Args:
            activations: (n_samples, hidden_dim) — must match training dim.

        Returns:
            (n_samples,) array of P(error=1) in [0, 1].
        """
        z = activations @ self.weights + self.bias
        # Numerically stable sigmoid.
        return np.where(
            z >= 0,
            1.0 / (1.0 + np.exp(-z)),
            np.exp(z) / (1.0 + np.exp(z)),
        )


def train_probe(
    *,
    X: np.ndarray,
    y: np.ndarray,
    layer: int,
    C: float = 1.0,
    max_iter: int = 1000,
    class_weight: str | None = "balanced",
    cv_folds: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
) -> LinearProbe:
    """Train a linear probe with cross-validated AUROC reporting.

    Strategy:
      1. Run K-fold CV on the entire dataset to get robust AUROC estimates.
      2. Hold out a final test split, train on the rest, report metrics.
      3. Refit on the full dataset for the final saved probe (more data
         = better probe; CV/test give us the honest performance estimate).

    Args:
        X: (n, hidden_dim) activation vectors.
        y: (n,) binary labels (0 = correct, 1 = error).
        layer: Which layer these activations came from (recorded in metadata).
        C: Inverse regularization strength.
        max_iter: sklearn convergence cap.
        class_weight: "balanced" recommended; class imbalance is real.
        cv_folds: K for cross-validation.
        test_size: Fraction held out for final test metrics.
        random_state: For reproducibility.

    Returns:
        LinearProbe with weights and metrics populated.
    """
    n_samples, hidden_dim = X.shape
    n_pos = int(y.sum())
    n_neg = n_samples - n_pos

    if n_pos < cv_folds or n_neg < cv_folds:
        # Not enough of one class for stratified K-fold. Fall back to
        # a simpler single train/test split.
        logger.warning(
            "Layer %d: only %d positive / %d negative; skipping CV, "
            "using single train/test split only.",
            layer, n_pos, n_neg,
        )
        per_fold_auroc: list[float] = []
    else:
        # K-fold CV.
        skf = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=random_state,
        )
        per_fold_auroc = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            clf = LogisticRegression(
                C=C, max_iter=max_iter, class_weight=class_weight,
                random_state=random_state,
            )
            clf.fit(X[train_idx], y[train_idx])
            val_proba = clf.predict_proba(X[val_idx])[:, 1]
            try:
                fold_auroc = float(roc_auc_score(y[val_idx], val_proba))
            except ValueError:
                # Single-class fold (shouldn't happen with stratified, but defensive).
                continue
            per_fold_auroc.append(fold_auroc)
        logger.info(
            "Layer %d CV AUROC: %.3f ± %.3f over %d folds",
            layer,
            float(np.mean(per_fold_auroc)) if per_fold_auroc else float("nan"),
            float(np.std(per_fold_auroc)) if per_fold_auroc else 0.0,
            len(per_fold_auroc),
        )

    # Single train/test split for additional metrics.
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n_samples)
    n_test = max(1, int(round(test_size * n_samples)))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    clf_eval = LogisticRegression(
        C=C, max_iter=max_iter, class_weight=class_weight,
        random_state=random_state,
    )
    clf_eval.fit(X[train_idx], y[train_idx])
    test_proba = clf_eval.predict_proba(X[test_idx])[:, 1]
    test_pred = clf_eval.predict(X[test_idx])

    n_test_errors = int(y[test_idx].sum())
    if n_test_errors == 0 or n_test_errors == len(test_idx):
        logger.warning(
            "Layer %d: test split has only one class; AUROC undefined.",
            layer,
        )
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = float(roc_auc_score(y[test_idx], test_proba))
        auprc = float(average_precision_score(y[test_idx], test_proba))

    brier = float(brier_score_loss(y[test_idx], test_proba))
    accuracy = float((test_pred == y[test_idx]).mean())

    # Threshold at which we'd recover at least 50% of errors.
    threshold_at_50 = _threshold_for_recall(
        y[test_idx], test_proba, target_recall=0.5,
    )

    metrics = ProbeMetrics(
        layer=layer,
        n_train=len(train_idx),
        n_test=len(test_idx),
        n_test_errors=n_test_errors,
        auroc=auroc,
        auprc=auprc,
        brier=brier,
        threshold_at_50pct_recall=threshold_at_50,
        accuracy_at_default_threshold=accuracy,
        per_fold_auroc=per_fold_auroc,
    )

    # Final probe: refit on ALL data for max signal.
    clf_final = LogisticRegression(
        C=C, max_iter=max_iter, class_weight=class_weight,
        random_state=random_state,
    )
    clf_final.fit(X, y)

    return LinearProbe(
        layer=layer,
        weights=clf_final.coef_.flatten().astype(np.float32),
        bias=float(clf_final.intercept_[0]),
        classes=clf_final.classes_,
        n_train=n_samples,
        metrics=metrics,
        config={
            "C": C, "max_iter": max_iter, "class_weight": class_weight,
            "cv_folds": cv_folds, "test_size": test_size,
            "random_state": random_state,
        },
    )


def _threshold_for_recall(
    y_true: np.ndarray, y_proba: np.ndarray, target_recall: float,
) -> float | None:
    """Smallest threshold such that recall on positive class >= target_recall.

    Returns None if no threshold achieves the target (e.g., zero positives).
    """
    pos_proba = y_proba[y_true == 1]
    if len(pos_proba) == 0:
        return None
    sorted_pos = np.sort(pos_proba)[::-1]  # descending
    n_needed = int(np.ceil(target_recall * len(sorted_pos)))
    if n_needed == 0:
        return float(sorted_pos[0])
    if n_needed > len(sorted_pos):
        return float(sorted_pos[-1])
    return float(sorted_pos[n_needed - 1])
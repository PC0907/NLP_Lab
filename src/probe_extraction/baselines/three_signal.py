"""Three-signal stacking regression (Task 1).

Combines THREE scalar trust signals per field:
  1. probe P(error)        -- the probe's scalar output (NOT its activation vector)
  2. logprob               -- output-level confidence (min-logprob by default)
  3. hand-crafted score    -- a scalar from surface features (length/complexity)

into ONE logistic regression, under leave-one-document-out. Because each signal
is reduced to a single standardized scalar, the regression yields ONE
interpretable coefficient per signal -- so we can read off how much each
contributes. (Contrast: concatenating the raw 2560-dim activation would give
2560 coefficients that swamp the 2 scalar signals and aren't interpretable.)

Standardization (z-score, fit on train fold only) is essential: the three
signals live on different scales (P(error) in [0,1], logprob ~ -20..0,
hand-crafted score unbounded), so without it coefficient magnitude would reflect
scale, not importance.

Returns metrics AND the per-signal coefficients (mean over folds).

Place at: src/probe_extraction/baselines/three_signal.py
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)


@dataclass
class ThreeSignalResult:
    name: str
    n_samples: int
    n_errors: int
    auroc: float
    auroc_std: float
    auprc: float
    signal_names: list[str]
    # mean (over folds) standardized coefficient per signal -> comparable importances
    coefficients: dict[str, float]
    coefficients_std: dict[str, float]


def _standardize(train: np.ndarray, test: np.ndarray):
    """Z-score using TRAIN statistics only (no leakage)."""
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)   # guard constant columns
    return (train - mu) / sd, (test - mu) / sd


def evaluate_three_signal(
    *,
    probe_score: np.ndarray,     # (n,) the probe's P(error) per field
    logprob: np.ndarray,         # (n,) min- or mean-logprob per field
    handcrafted: np.ndarray,     # (n,) a scalar hand-crafted score per field
    y: np.ndarray,
    doc_ids: list[str],
    signal_names: tuple[str, str, str] = ("probe", "logprob", "handcrafted"),
    C: float = 1.0,
    max_iter: int = 1000,
    class_weight: str | None = "balanced",
    random_state: int = 42,
) -> ThreeSignalResult:
    """LODO stacking regression over the 3 scalar signals, with standardized,
    interpretable per-signal coefficients."""
    X = np.stack([probe_score, logprob, handcrafted], axis=1).astype(np.float64)
    doc_ids_arr = np.asarray(doc_ids)
    unique_docs = list(dict.fromkeys(doc_ids))

    # Drop rows with NaN in any signal (e.g. missing logprob).
    row_ok = ~np.isnan(X).any(axis=1)
    if not row_ok.all():
        logger.warning("three_signal: dropping %d/%d rows with NaN signals.",
                        int((~row_ok).sum()), len(y))
    X, y2, doc_ids_arr = X[row_ok], y[row_ok], doc_ids_arr[row_ok]

    fold_aurocs: list[float] = []
    fold_coefs: list[np.ndarray] = []
    oof = np.full(len(y2), np.nan, dtype=np.float64)

    for doc in unique_docs:
        test_mask = doc_ids_arr == doc
        train_mask = ~test_mask
        if test_mask.sum() == 0:
            continue
        y_tr, y_te = y2[train_mask], y2[test_mask]
        if y_tr.sum() in (0, len(y_tr)):
            continue
        Xtr, Xte = _standardize(X[train_mask], X[test_mask])
        clf = LogisticRegression(C=C, max_iter=max_iter,
                                 class_weight=class_weight,
                                 random_state=random_state)
        clf.fit(Xtr, y_tr)
        oof[test_mask] = clf.predict_proba(Xte)[:, 1]
        fold_coefs.append(clf.coef_.ravel().copy())
        if y_te.sum() not in (0, len(y_te)):
            fold_aurocs.append(float(roc_auc_score(y_te, oof[test_mask])))

    n, n_err = len(y2), int(y2.sum())
    if not fold_aurocs:
        logger.warning("three_signal: no valid folds.")
        return ThreeSignalResult(
            name="three_signal", n_samples=n, n_errors=n_err,
            auroc=float("nan"), auroc_std=float("nan"), auprc=float("nan"),
            signal_names=list(signal_names),
            coefficients={}, coefficients_std={})

    coefs = np.stack(fold_coefs, axis=0)        # (folds, 3)
    coef_mean = coefs.mean(axis=0)
    coef_std = coefs.std(axis=0)
    valid = ~np.isnan(oof)
    auprc = (float(average_precision_score(y2[valid], oof[valid]))
             if valid.sum() and y2[valid].sum() not in (0, int(valid.sum()))
             else float("nan"))

    result = ThreeSignalResult(
        name="three_signal", n_samples=n, n_errors=n_err,
        auroc=float(np.mean(fold_aurocs)), auroc_std=float(np.std(fold_aurocs)),
        auprc=auprc, signal_names=list(signal_names),
        coefficients={s: float(c) for s, c in zip(signal_names, coef_mean)},
        coefficients_std={s: float(c) for s, c in zip(signal_names, coef_std)},
    )
    logger.info("three_signal LODO AUROC=%.4f±%.4f, AUPRC=%.4f over %d folds",
                result.auroc, result.auroc_std, result.auprc, len(fold_aurocs))
    logger.info("standardized coefficients (importance): %s",
                {k: round(v, 3) for k, v in result.coefficients.items()})
    return result
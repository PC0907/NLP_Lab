"""Three-signal stacking regression (Task 1) — LEAK-FREE version.

Combines THREE scalar trust signals per field:
  1. probe P(error)     -- but computed OUT-OF-FOLD (see below), not from a probe
                           that trained on the field being scored
  2. logprob            -- output-level confidence (min-logprob by default)
  3. hand-crafted score -- a scalar from surface features (length/complexity)

into ONE logistic regression under leave-one-document-out, yielding one
interpretable standardized coefficient per signal.

WHY OUT-OF-FOLD MATTERS (the bug this fixes):
A probe trained on the full dataset and then scored on that same data gives an
in-sample (overfit) prediction -- the full-set probe AUROC is ~1.000. If that
leaked probe score is used as a feature, the stacking regression inherits the
leak and reports a fake ~1.000. To be honest, the probe feature for a held-out
document must come from a probe that NEVER trained on that document. So here we
train the probe INSIDE the LODO loop: for each held-out doc, fit a fresh probe
on the other docs' activations, score the held-out doc, and use THAT as the
probe feature. This matches the probe's honest CV AUROC (~0.92), not the
optimistic 1.000.

Standardization (z-score, train-fold stats only) makes the three coefficients
comparable as importances rather than reflecting raw feature scale.

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
    coefficients: dict[str, float]
    coefficients_std: dict[str, float]


def _standardize(train: np.ndarray, test: np.ndarray):
    """Z-score using TRAIN statistics only (no leakage)."""
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (train - mu) / sd, (test - mu) / sd


def evaluate_three_signal(
    *,
    activations: np.ndarray,     # (n, hidden) raw probe-layer activations
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
    """LODO stacking over 3 scalar signals, with the probe feature computed
    OUT-OF-FOLD (probe trained per-fold on the other docs). Leak-free."""
    activations = activations.astype(np.float64)
    logprob = logprob.astype(np.float64)
    handcrafted = handcrafted.astype(np.float64)
    doc_ids_arr = np.asarray(doc_ids)
    unique_docs = list(dict.fromkeys(doc_ids))

    # Drop rows with NaN in the scalar signals (e.g. missing logprob).
    row_ok = ~(np.isnan(logprob) | np.isnan(handcrafted))
    if not row_ok.all():
        logger.warning("three_signal: dropping %d/%d rows with NaN scalar signals.",
                        int((~row_ok).sum()), len(y))
    activations, logprob, handcrafted = activations[row_ok], logprob[row_ok], handcrafted[row_ok]
    y2 = y[row_ok]
    doc_ids_arr = doc_ids_arr[row_ok]

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

        # --- OUT-OF-FOLD probe feature ---
        # Train a probe on the OTHER docs' activations, score the held-out doc.
        # This is the leak-free probe P(error) for the held-out fields.
        probe = LogisticRegression(C=C, max_iter=max_iter,
                                   class_weight=class_weight,
                                   random_state=random_state)
        probe.fit(activations[train_mask], y_tr)
        probe_tr = probe.predict_proba(activations[train_mask])[:, 1]
        probe_te = probe.predict_proba(activations[test_mask])[:, 1]

        # Assemble the 3 scalar signals for train/test.
        Xtr = np.stack([probe_tr, logprob[train_mask], handcrafted[train_mask]], axis=1)
        Xte = np.stack([probe_te, logprob[test_mask], handcrafted[test_mask]], axis=1)

        # Standardize (train stats only) so coefficients are comparable.
        Xtr_s, Xte_s = _standardize(Xtr, Xte)

        # --- meta-regression over the 3 standardized scalar signals ---
        meta = LogisticRegression(C=C, max_iter=max_iter,
                                  class_weight=class_weight,
                                  random_state=random_state)
        meta.fit(Xtr_s, y_tr)
        oof[test_mask] = meta.predict_proba(Xte_s)[:, 1]
        fold_coefs.append(meta.coef_.ravel().copy())
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

    coefs = np.stack(fold_coefs, axis=0)
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
    logger.info("three_signal LODO AUROC=%.4f±%.4f, AUPRC=%.4f over %d folds (probe feature OUT-OF-FOLD)",
                result.auroc, result.auroc_std, result.auprc, len(fold_aurocs))
    logger.info("standardized coefficients (importance): %s",
                {k: round(v, 3) for k, v in result.coefficients.items()})
    return result
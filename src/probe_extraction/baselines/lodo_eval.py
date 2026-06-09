"""Leave-one-document-out evaluation for TRAINED baselines.

Scalar baselines (token_logprob) need no training -> full-set AUROC via
evaluate_baseline. Trained baselines (hand-crafted features, combined
probe+logprob) CAN overfit, so to be comparable to the probe's LODO AUROC
they must be evaluated under the same document-held-out folds. This module
provides that fold loop, operating on the in-memory arrays 04_evaluate.py
already assembles (X, y, and per-field doc_ids from meta).
"""
from __future__ import annotations

import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from probe_extraction.baselines.token_logprob import BaselineMetrics

logger = logging.getLogger(__name__)


def lodo_evaluate(
    *,
    X: np.ndarray,
    y: np.ndarray,
    doc_ids: list[str],
    name: str,
    C: float = 1.0,
    max_iter: int = 1000,
    class_weight: str | None = "balanced",
    random_state: int = 42,
) -> BaselineMetrics:
    """Train-and-test a logistic regression under leave-one-document-out CV.

    For each unique document: hold it out, train on the others' fields, score
    the held-out doc, collect AUROC. Aggregate (mean) across folds. AUPRC on
    pooled out-of-fold predictions. Mirrors scripts/05_lodo_cv.py so the AUROC
    is directly comparable to the probe's LODO number.
    """
    doc_ids_arr = np.asarray(doc_ids)
    unique_docs = list(dict.fromkeys(doc_ids))

    # Drop feature columns that are NaN anywhere (e.g. missing logprobs).
    col_ok = ~np.isnan(X).any(axis=0)
    if not col_ok.all():
        logger.warning("%s: dropping %d/%d NaN feature columns.",
                       name, int((~col_ok).sum()), X.shape[1])
    Xc = X[:, col_ok] if col_ok.any() else X

    fold_aurocs: list[float] = []
    oof = np.full(len(y), np.nan, dtype=np.float64)

    for doc in unique_docs:
        test_mask = doc_ids_arr == doc
        train_mask = ~test_mask
        y_tr, y_te = y[train_mask], y[test_mask]
        if y_tr.sum() in (0, len(y_tr)):
            continue
        clf = LogisticRegression(C=C, max_iter=max_iter,
                                 class_weight=class_weight, random_state=random_state)
        clf.fit(Xc[train_mask], y_tr)
        s = clf.predict_proba(Xc[test_mask])[:, 1]
        oof[test_mask] = s
        if y_te.sum() not in (0, len(y_te)):
            fold_aurocs.append(float(roc_auc_score(y_te, s)))

    n = len(y)
    n_errors = int(y.sum())
    if not fold_aurocs:
        logger.warning("%s: no valid LODO folds.", name)
        return BaselineMetrics(name=name, n_samples=n, n_errors=n_errors,
                               auroc=float("nan"), auprc=float("nan"))

    mean_auroc = float(np.mean(fold_aurocs))
    std_auroc = float(np.std(fold_aurocs))
    valid = ~np.isnan(oof)
    if valid.sum() and y[valid].sum() not in (0, int(valid.sum())):
        auprc = float(average_precision_score(y[valid], oof[valid]))
    else:
        auprc = float("nan")

    logger.info("%s LODO AUROC: %.4f ± %.4f over %d folds",
                name, mean_auroc, std_auroc, len(fold_aurocs))
    return BaselineMetrics(name=name, n_samples=n, n_errors=n_errors,
                           auroc=mean_auroc, auprc=auprc)
"""Nested grouped K-fold probe evaluation (for many-record benchmarks like SOB).

05b_nested_lodo.py holds out one document per outer AND inner fold. That fits
ExtractBench (28 docs) but not SOB (~1000 short records): ~1M fits per layer,
and most single-record folds have no errors, so AUROC is undefined and the fold
is skipped.

This script keeps everything else identical and only changes the fold scheme:
  outer: StratifiedGroupKFold(k_outer), grouped by doc_id
  inner: StratifiedGroupKFold(k_inner) on the outer-train records only,
         picks the layer with the best pooled inner out-of-fold AUROC
  then:  refit the selected layer on all outer-train records, score outer-test

Features, labels and the probe come from 05b (load_layer_matrix, _fit_score),
so numbers are comparable to the ExtractBench nested LODO.

Outputs under {artifacts}/probes/:
  nested_groupkfold.json   pooled OOF AUROC (headline), per-fold AUROC,
                           selected layers, per-layer fixed-layer pooled AUROC
  oof_scores.npz           doc_ids, y, oof probe score, outer fold index
                           (input for selective regeneration later)

Usage:
    python scripts/05c_nested_groupkfold.py --config configs/exp_qwen35_4b_sob_1k.yaml \
        --layers 14 16 18 20 22
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from probe_extraction.config import load_config
from probe_extraction.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# Reuse 05b's loader and probe fit (module name starts with a digit, so load by path).
_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("nested_lodo_05b", _HERE / "05b_nested_lodo.py")
_m05b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m05b)
load_layer_matrix = _m05b.load_layer_matrix
_fit_score = _m05b._fit_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nested grouped K-fold probe evaluation.")
    p.add_argument("--config", required=True)
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Candidate layers (default: config activations.layers).")
    p.add_argument("--k-outer", type=int, default=5)
    p.add_argument("--k-inner", type=int, default=5)
    p.add_argument("--seed", type=int, default=None,
                   help="Fold seed (default: config experiment.seed).")
    return p.parse_args()


def _pooled_auroc(y: np.ndarray, s: np.ndarray) -> float | None:
    ok = ~np.isnan(s)
    if ok.sum() == 0 or len(set(y[ok].tolist())) < 2:
        return None
    return float(roc_auc_score(y[ok], s[ok]))


def inner_select_layer(X, y, groups, train_idx, layers, C, k_inner, seed):
    """Pick the layer with the best pooled inner-OOF AUROC, using train_idx only."""
    y_tr, g_tr = y[train_idx], groups[train_idx]
    cv = StratifiedGroupKFold(n_splits=k_inner, shuffle=True, random_state=seed)
    splits = list(cv.split(np.zeros(len(train_idx)), y_tr, g_tr))

    scores_by_layer = {}
    for L in layers:
        XL = X[L][train_idx]
        oof = np.full(len(train_idx), np.nan)
        for itr, ite in splits:
            if len(set(y_tr[itr].tolist())) < 2:
                continue
            oof[ite] = _fit_score(XL[itr], y_tr[itr], XL[ite], C)
        scores_by_layer[L] = _pooled_auroc(y_tr, oof)

    valid = {L: a for L, a in scores_by_layer.items() if a is not None}
    if not valid:
        return None, scores_by_layer
    best = max(valid, key=valid.get)
    return best, scores_by_layer


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_dir=cfg.logging.log_dir,
                  log_name="05c_nested_groupkfold", log_to_file=cfg.logging.log_to_file)

    layers = args.layers or cfg.activations.layers
    seed = args.seed if args.seed is not None else cfg.experiment.seed
    C = cfg.probe.C
    labels_dir = cfg.artifacts_path / "labels"
    activations_dir = cfg.artifacts_path / "activations"
    out_dir = cfg.artifacts_path / "probes"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading activations for layers %s ...", layers)
    X, y, doc_ids = load_layer_matrix(labels_dir, activations_dir, layers,
                                      include=None, exclude=None)
    y = np.asarray(y).astype(int)
    doc_ids = np.asarray(doc_ids)
    n_docs = len(set(doc_ids.tolist()))
    logger.info("Loaded %d fields across %d records (%d errors, %.1f%%).",
                len(y), n_docs, int(y.sum()), 100.0 * y.mean())

    outer = StratifiedGroupKFold(n_splits=args.k_outer, shuffle=True, random_state=seed)
    outer_splits = list(outer.split(np.zeros(len(y)), y, doc_ids))

    oof = np.full(len(y), np.nan)
    fold_of = np.full(len(y), -1, dtype=int)
    per_fold = []

    for k, (tr, te) in enumerate(outer_splits):
        L_sel, inner_scores = inner_select_layer(X, y, doc_ids, tr, layers, C,
                                                 args.k_inner, seed)
        if L_sel is None:
            logger.warning("Outer fold %d: no valid inner layer, skipped.", k)
            continue
        s = _fit_score(X[L_sel][tr], y[tr], X[L_sel][te], C)
        oof[te] = s
        fold_of[te] = k
        fold_auroc = _pooled_auroc(y[te], np.asarray(s, dtype=float))
        per_fold.append({
            "fold": k,
            "selected_layer": int(L_sel),
            "inner_auroc_by_layer": {str(L): a for L, a in inner_scores.items()},
            "test_auroc": fold_auroc,
            "n_test_fields": int(len(te)),
            "n_test_errors": int(y[te].sum()),
        })
        logger.info("Outer fold %d: layer %d (inner %.4f) -> test AUROC %s on %d fields",
                    k, L_sel, inner_scores[L_sel],
                    f"{fold_auroc:.4f}" if fold_auroc is not None else "n/a", len(te))

    # Fixed-layer pooled OOF per layer, same outer folds (for the per-layer figure).
    # Not a nested number: the layer is not selected here.
    fixed_layer = {}
    for L in layers:
        oof_L = np.full(len(y), np.nan)
        for tr, te in outer_splits:
            if len(set(y[tr].tolist())) < 2:
                continue
            oof_L[te] = _fit_score(X[L][tr], y[tr], X[L][te], C)
        fixed_layer[str(L)] = _pooled_auroc(y, oof_L)
        logger.info("Fixed layer %d: pooled OOF AUROC %s", L, fixed_layer[str(L)])

    fold_aurocs = [f["test_auroc"] for f in per_fold if f["test_auroc"] is not None]
    result = {
        "config": args.config,
        "layers": [int(L) for L in layers],
        "k_outer": args.k_outer,
        "k_inner": args.k_inner,
        "seed": int(seed),
        "C": C,
        "n_fields": int(len(y)),
        "n_records": int(n_docs),
        "n_errors": int(y.sum()),
        "nested_pooled_oof_auroc": _pooled_auroc(y, oof),
        "nested_fold_mean_auroc": float(np.mean(fold_aurocs)) if fold_aurocs else None,
        "selected_layer_counts": dict(Counter(str(f["selected_layer"]) for f in per_fold)),
        "per_fold": per_fold,
        "fixed_layer_pooled_oof_auroc": fixed_layer,
    }
    with (out_dir / "nested_groupkfold.json").open("w") as fh:
        json.dump(result, fh, indent=2)
    np.savez(out_dir / "oof_scores.npz", doc_ids=doc_ids, y=y, oof=oof, fold=fold_of)

    logger.info("=" * 70)
    logger.info("Nested pooled OOF AUROC: %s", result["nested_pooled_oof_auroc"])
    logger.info("Fold-mean AUROC: %s", result["nested_fold_mean_auroc"])
    logger.info("Selected layers: %s", result["selected_layer_counts"])
    logger.info("Wrote %s and oof_scores.npz", out_dir / "nested_groupkfold.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
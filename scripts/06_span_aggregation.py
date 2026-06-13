#!/usr/bin/env python3
"""Compare per-field activation aggregations under LODO: last / mean / span-max.

Requires an extraction run with `position: "all_tokens"`, so each field's
activation is stored as a (span_len, hidden_dim) array per layer. From that one
extraction we can compute every aggregation downstream (no re-extraction):

  - last  : last token's vector            -> probe score
  - mean  : mean over the span's vectors   -> probe score
  - max   : score EVERY token, take the max (span-max; Obeso et al. 2025)

For 1-token fields, all three are identical by construction. The difference (if
any) comes from multi-token fields, especially the long tail.

This runs LODO directly (trains a fresh logistic-regression probe per fold on
the chosen aggregation) so the three are compared on equal footing. It does NOT
reuse a previously-trained last-token probe, because applying a last-token-
trained probe to mid-span tokens would be off-distribution; span-max is only
fair if the probe is trained consistently with the aggregation.

Usage:
    python scripts/06_span_aggregation.py --config configs/exp_qwen35_4b_pooled.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from probe_extraction.config import load_config
from probe_extraction.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Aggregation comparison under LODO")
    p.add_argument("--config", required=True)
    p.add_argument("--layers", type=int, nargs="*", default=None,
                   help="Layers to evaluate (default: all in config).")
    p.add_argument("--C", type=float, default=1.0)
    return p.parse_args()


def _aggregate(arr2d: np.ndarray, mode: str, probe=None) -> np.ndarray | float:
    """Given a field's (span_len, hidden_dim) activation, return either a
    feature vector (last/mean) or, for max, a per-token feature matrix to be
    scored by the probe (handled by the caller)."""
    if arr2d.ndim == 1:
        arr2d = arr2d[None, :]  # tolerate accidental 1D (single-token)
    if mode == "last":
        return arr2d[-1].astype(np.float32)
    if mode == "mean":
        return arr2d.astype(np.float32).mean(axis=0)
    raise AssertionError(mode)


def build_matrix(labels_dir, activations_dir, layer):
    """Load per-field all-token activations for one layer.

    Returns lists aligned by field:
      tokens[i] = (span_len_i, hidden_dim) float32 array
      y[i], doc_ids[i]
    """
    tokens, y, doc_ids = [], [], []
    for lp in sorted(Path(labels_dir).glob("*.json")):
        if lp.name.startswith("_"):
            continue
        doc_id = lp.stem
        data = json.load(lp.open())
        npz = Path(activations_dir) / f"{doc_id}.npz"
        if not npz.exists():
            continue
        with np.load(npz) as acts:
            for fld in data.get("labels", []):
                if not fld.get("extracted_present", True):
                    continue
                key = f"{fld['path_str']}__layer{layer}"
                if key not in acts:
                    continue
                a = acts[key].astype(np.float32)
                if a.ndim == 1:
                    a = a[None, :]
                tokens.append(a)
                y.append(int(fld.get("is_error", 0)))
                doc_ids.append(doc_id)
    return tokens, np.array(y), doc_ids


def lodo_score(tokens, y, doc_ids, mode, C):
    """LODO for a given aggregation mode.

    For last/mean: each field -> one feature vector; standard LODO.
    For max: train a token-level probe (every token is a training row with its
    field's label), then at test time score every token of a field and take the
    max as the field score. This mirrors the paper's span-max usage.
    """
    docs = list(dict.fromkeys(doc_ids))
    doc_ids = np.array(doc_ids)
    fold_aurocs = []
    oof = np.full(len(y), np.nan)

    for d in docs:
        te = doc_ids == d
        tr = ~te
        if y[tr].sum() in (0, tr.sum()):
            continue

        if mode in ("last", "mean"):
            Xtr = np.stack([_aggregate(tokens[i], mode) for i in np.where(tr)[0]])
            Xte = np.stack([_aggregate(tokens[i], mode) for i in np.where(te)[0]])
            clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced")
            clf.fit(Xtr, y[tr])
            s = clf.predict_proba(Xte)[:, 1]

        elif mode == "max":
            # token-level training rows
            Xtr_rows, ytr_rows = [], []
            for i in np.where(tr)[0]:
                for row in tokens[i]:
                    Xtr_rows.append(row); ytr_rows.append(y[i])
            clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced")
            clf.fit(np.stack(Xtr_rows), np.array(ytr_rows))
            # test: score every token, take max per field
            s = []
            for i in np.where(te)[0]:
                tok_scores = clf.predict_proba(tokens[i])[:, 1]
                s.append(float(tok_scores.max()))
            s = np.array(s)
        else:
            raise AssertionError(mode)

        oof[te] = s
        yte = y[te]
        if yte.sum() not in (0, len(yte)):
            fold_aurocs.append(roc_auc_score(yte, s))

    if not fold_aurocs:
        return float("nan"), float("nan"), 0
    valid = ~np.isnan(oof)
    auprc = (average_precision_score(y[valid], oof[valid])
             if y[valid].sum() not in (0, valid.sum()) else float("nan"))
    return float(np.mean(fold_aurocs)), float(np.std(fold_aurocs)), len(fold_aurocs)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_dir=cfg.logging.log_dir,
                  log_name="06_span_aggregation", log_to_file=cfg.logging.log_to_file)
    labels_dir = cfg.artifacts_path / "labels"
    activations_dir = cfg.artifacts_path / "activations"
    layers = args.layers or cfg.activations.layers

    logger.info("Aggregation comparison (LODO): last vs mean vs span-max")
    logger.info("%-7s %-8s %-8s %-8s", "layer", "last", "mean", "max")
    for layer in layers:
        tokens, y, doc_ids = build_matrix(labels_dir, activations_dir, layer)
        if len(y) == 0:
            logger.warning("Layer %d: no data (did you extract with position=all_tokens?)", layer)
            continue
        results = {}
        for mode in ("last", "mean", "max"):
            auroc, std, nf = lodo_score(tokens, y, doc_ids, mode, args.C)
            results[mode] = auroc
        logger.info("%-7d %-8.3f %-8.3f %-8.3f",
                    layer, results["last"], results["mean"], results["max"])

    # also report span-length context
    span_lens = [t.shape[0] for t in tokens] if 'tokens' in dir() else []
    if span_lens:
        logger.info("fields=%d  single-token=%d (%.0f%%)  median_span=%d",
                    len(span_lens), sum(1 for x in span_lens if x <= 1),
                    100*sum(1 for x in span_lens if x <= 1)/len(span_lens),
                    int(np.median(span_lens)))


if __name__ == "__main__":
    import sys
    sys.exit(main())
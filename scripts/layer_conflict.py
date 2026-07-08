#!/usr/bin/env python3
"""Inter-layer conflict signal (REDD-inspired).

HYPOTHESIS: when per-layer probes DISAGREE about a field, that field is more
likely to be an error (REDD, arXiv:2511.02711). We test whether cross-layer
disagreement is (a) informative on its own, and (b) adds anything over the
single best layer -- the latter being the claim that matters for the
safe-override gate.

METHOD (all leakage-free under LODO):
  1. For each candidate layer L, run LODO: for each held-out doc, train a probe
     on the other docs at layer L, score the held-out doc. This gives every
     field an OUT-OF-FOLD P(error) at layer L -- a probe that never saw that
     field's document.
  2. Each field now has a vector of per-layer OOF predictions [p_L1, p_L4, ...].
  3. CONFLICT = spread of that vector (std across layers; also max-min).
  4. Three tests, pooled over all OOF fields:
       (a) conflict-alone   : AUROC(spread vs is_error)      -- is disagreement informative?
       (b) conflict-added   : [best-layer p, spread] stacked vs best-layer p alone
       (c) layer-ensemble   : mean of per-layer p vs best-layer p alone
  Report is honest about NULLs: given the 3-signal result showed the probe
  already saturates, conflict may add ~nothing (layers are the same residual
  stream, highly correlated). A null here is a legitimate finding.

Reuses load_layer_matrix from 05b so data loading is identical to the pipeline.
CPU only.

Usage:
    python scripts/06_layer_conflict.py --config configs/exp_qwen35_4b_pooled_alltokens.yaml \\
        --exclude-domains finance/10kq --out-name layer_conflict.json
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

# Reuse the EXACT loader from 05b so fields/keys/reduction match the pipeline.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "nested_lodo_mod", str(Path(__file__).parent / "05b_nested_lodo.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
load_layer_matrix = _mod.load_layer_matrix

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--layers", type=int, nargs="*", default=None)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--include-domains", nargs="*", default=None)
    p.add_argument("--exclude-domains", nargs="*", default=None)
    p.add_argument("--include-docs-file", default=None)
    p.add_argument("--out-name", default="layer_conflict.json")
    return p.parse_args()


def _fit_score(Xtr, ytr, Xte, C):
    clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def oof_predictions_for_layer(XL, y, doc_ids, docs, C):
    """LODO out-of-fold P(error) for one layer: each field scored by a probe
    trained on all OTHER documents. Returns an array aligned with y (NaN where
    a fold was single-class and could not be scored)."""
    oof = np.full(len(y), np.nan)
    for d in docs:
        te = (doc_ids == d)
        tr = ~te
        if y[tr].sum() == 0 or y[tr].sum() == tr.sum():
            continue  # train degenerate
        oof[te] = _fit_score(XL[tr], y[tr], XL[te], C)
    return oof


def safe_auroc(y, s):
    m = ~np.isnan(s)
    if m.sum() == 0 or y[m].sum() in (0, m.sum()):
        return float("nan")
    return roc_auc_score(y[m], s[m])


def safe_auprc(y, s):
    m = ~np.isnan(s)
    if m.sum() == 0 or y[m].sum() in (0, m.sum()):
        return float("nan")
    return average_precision_score(y[m], s[m])


def main():
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_dir=cfg.logging.log_dir,
                  log_name="06_layer_conflict", log_to_file=cfg.logging.log_to_file)

    layers = args.layers or cfg.activations.layers
    labels_dir = cfg.artifacts_path / "labels"
    activations_dir = cfg.artifacts_path / "activations"

    include_docs = None
    if args.include_docs_file:
        include_docs = {l.strip() for l in Path(args.include_docs_file).read_text().splitlines() if l.strip()}

    logger.info("Loading activations for layers %s ...", layers)
    X, y, doc_ids = load_layer_matrix(
        labels_dir, activations_dir, layers,
        include=args.include_domains, exclude=args.exclude_domains,
        include_docs=include_docs)
    docs = list(dict.fromkeys(doc_ids.tolist()))
    logger.info("Loaded %d fields across %d docs (%d errors, %.1f%%).",
                len(y), len(docs), int(y.sum()), 100 * y.mean())

    # 1. per-layer OOF predictions
    logger.info("Computing per-layer OOF predictions (LODO) ...")
    P = {}  # layer -> oof array
    per_layer_auroc = {}
    for L in layers:
        P[L] = oof_predictions_for_layer(X[L], y, doc_ids, docs, args.C)
        per_layer_auroc[L] = safe_auroc(y, P[L])
        logger.info("  layer %d: pooled-OOF AUROC = %.4f", L, per_layer_auroc[L])

    # stack into (n_fields, n_layers), keep only fields scored at ALL layers
    Pmat = np.column_stack([P[L] for L in layers])
    valid = ~np.isnan(Pmat).any(axis=1)
    Pv = Pmat[valid]
    yv = y[valid]
    docs_v = doc_ids[valid]
    logger.info("Fields scored at all layers: %d / %d", valid.sum(), len(y))

    # best single layer (by pooled-OOF AUROC) -> the baseline to beat
    best_layer = max(per_layer_auroc, key=lambda L: (per_layer_auroc[L] if not np.isnan(per_layer_auroc[L]) else -1))
    best_idx = layers.index(best_layer)
    best_p = Pv[:, best_idx]
    best_auroc = safe_auroc(yv, best_p)
    logger.info("Best single layer = %d (pooled-OOF AUROC %.4f)", best_layer, best_auroc)

    # 2. conflict signal = spread across layers
    spread_std = Pv.std(axis=1)
    spread_range = Pv.max(axis=1) - Pv.min(axis=1)

    # --- test (a): conflict alone ---
    a_std = safe_auroc(yv, spread_std)
    a_rng = safe_auroc(yv, spread_range)

    # --- test (c): layer ensemble (mean) ---
    ens = Pv.mean(axis=1)
    c_auroc = safe_auroc(yv, ens)
    c_auprc = safe_auprc(yv, ens)

    # --- test (b): conflict as ADDED feature, under LODO on the meta-features ---
    # meta-features per field: [best_layer_p, spread_std]. Train a tiny LODO
    # logistic reg to combine, so this is leakage-free too.
    def meta_lodo(feats):
        oof = np.full(len(yv), np.nan)
        for d in list(dict.fromkeys(docs_v.tolist())):
            te = (docs_v == d); tr = ~te
            if yv[tr].sum() in (0, tr.sum()):
                continue
            oof[te] = _fit_score(feats[tr], yv[tr], feats[te], args.C)
        return oof

    single_feat = best_p.reshape(-1, 1)
    added_feat = np.column_stack([best_p, spread_std])
    b_single = safe_auroc(yv, meta_lodo(single_feat))   # sanity: ~= best_auroc
    b_added = safe_auroc(yv, meta_lodo(added_feat))

    # ---- report ----
    logger.info("=" * 64)
    logger.info("LAYER-CONFLICT ANALYSIS")
    logger.info("-" * 64)
    logger.info("(a) conflict ALONE:  AUROC(std)=%.4f  AUROC(range)=%.4f", a_std, a_rng)
    logger.info("    -> is layer-disagreement informative on its own?")
    logger.info("(b) conflict ADDED:  single-layer=%.4f  +conflict=%.4f  delta=%+.4f",
                b_single, b_added, b_added - b_single)
    logger.info("    -> does disagreement add over the best single layer? (the REDD claim)")
    logger.info("(c) layer ENSEMBLE:  best-single=%.4f  mean-ensemble=%.4f  delta=%+.4f",
                best_auroc, c_auroc, c_auroc - best_auroc)
    logger.info("    -> does averaging layers beat the best single layer?")
    logger.info("-" * 64)
    if abs(b_added - b_single) < 0.005:
        logger.info("READ: conflict adds < 0.005 -> effectively NULL. Layers are")
        logger.info("highly correlated (same residual stream); the single-layer probe")
        logger.info("already captures the signal. This is a legitimate negative result,")
        logger.info("consistent with the 3-signal finding that the probe saturates.")
    else:
        logger.info("READ: conflict shifts AUROC by %+.4f -- non-trivial; worth pursuing", b_added - b_single)
    logger.info("=" * 64)

    out = cfg.artifacts_path / "results" / args.out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_fields": int(valid.sum()), "n_errors": int(yv.sum()),
        "per_layer_auroc": {str(L): per_layer_auroc[L] for L in layers},
        "best_layer": int(best_layer), "best_layer_auroc": best_auroc,
        "conflict_alone_auroc_std": a_std,
        "conflict_alone_auroc_range": a_rng,
        "added_single_auroc": b_single, "added_with_conflict_auroc": b_added,
        "added_delta": b_added - b_single,
        "ensemble_auroc": c_auroc, "ensemble_auprc": c_auprc,
        "ensemble_delta_vs_best": c_auroc - best_auroc,
        "layers": layers,
    }, indent=2))
    logger.info("Saved to %s", out)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
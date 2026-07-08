#!/usr/bin/env python3
"""Inter-layer conflict signal -- CORRECTED added-feature test.

The v1 script compared two SCALARS (probe prob + conflict) through a second
logistic regression, which degraded the single-layer signal (0.898 -> 0.774)
and made the delta uninterpretable. This version does the RIGHT test:

  Take the best layer's FULL activation vector (the features the probe actually
  uses). Compare, under identical LODO:
     baseline : probe on [best-layer activation vector]
     +conflict: probe on [best-layer activation vector | conflict scalar]
  where the conflict scalar = std across layers of the per-layer OOF probs.

  If +conflict beats baseline, layer-disagreement adds real signal on top of the
  probe. If not (expected), it's a clean null consistent with the 3-signal
  saturation result.

Reuses 05b's loader. CPU only.

Usage:
  python scripts/layer_conflict_v2.py --config configs/exp_qwen35_4b_pooled_alltokens.yaml \\
      --exclude-domains finance/10kq --out-name layer_conflict_v2.json
"""
from __future__ import annotations
import argparse, json, logging
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from probe_extraction.config import load_config
from probe_extraction.utils.logging import setup_logging

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "nested_lodo_mod", str(Path(__file__).parent / "05b_nested_lodo.py"))
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
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
    p.add_argument("--out-name", default="layer_conflict_v2.json")
    return p.parse_args()


def fit_score(Xtr, ytr, Xte, C):
    clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def lodo_auroc(Xfull, y, doc_ids, docs, C):
    """Pooled-OOF AUROC of a probe on feature matrix Xfull under LODO."""
    oof = np.full(len(y), np.nan)
    for d in docs:
        te = (doc_ids == d); tr = ~te
        if y[tr].sum() in (0, tr.sum()):
            continue
        oof[te] = fit_score(Xfull[tr], y[tr], Xfull[te], C)
    m = ~np.isnan(oof)
    return roc_auc_score(y[m], oof[m]), oof


def per_layer_oof(XL, y, doc_ids, docs, C):
    oof = np.full(len(y), np.nan)
    for d in docs:
        te = (doc_ids == d); tr = ~te
        if y[tr].sum() in (0, tr.sum()):
            continue
        oof[te] = fit_score(XL[tr], y[tr], XL[te], C)
    return oof


def main():
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_dir=cfg.logging.log_dir,
                  log_name="layer_conflict_v2", log_to_file=cfg.logging.log_to_file)
    layers = args.layers or cfg.activations.layers
    labels_dir = cfg.artifacts_path / "labels"
    activations_dir = cfg.artifacts_path / "activations"
    include_docs = None
    if args.include_docs_file:
        include_docs = {l.strip() for l in Path(args.include_docs_file).read_text().splitlines() if l.strip()}

    X, y, doc_ids = load_layer_matrix(labels_dir, activations_dir, layers,
                                      include=args.include_domains, exclude=args.exclude_domains,
                                      include_docs=include_docs)
    docs = list(dict.fromkeys(doc_ids.tolist()))
    logger.info("Loaded %d fields, %d docs, %d errors (%.1f%%)",
                len(y), len(docs), int(y.sum()), 100*y.mean())

    # per-layer OOF probs -> conflict scalar (std across layers)
    logger.info("Per-layer OOF for conflict scalar ...")
    P = np.column_stack([per_layer_oof(X[L], y, doc_ids, docs, args.C) for L in layers])
    valid = ~np.isnan(P).any(axis=1)
    conflict = np.full(len(y), np.nan)
    conflict[valid] = P[valid].std(axis=1)

    # pick best layer by its own OOF AUROC
    per_auroc = {}
    for i, L in enumerate(layers):
        m = ~np.isnan(P[:, i])
        per_auroc[L] = roc_auc_score(y[m], P[m, i])
    best_layer = max(per_auroc, key=per_auroc.get)
    logger.info("Best layer = %d (OOF AUROC %.4f)", best_layer, per_auroc[best_layer])

    # restrict to fields valid everywhere (so both models see identical rows)
    yv = y[valid]; dv = doc_ids[valid]
    docs_v = list(dict.fromkeys(dv.tolist()))
    Xbest = X[best_layer][valid]                      # full activation vector
    conf = conflict[valid].reshape(-1, 1)

    # BASELINE: probe on best-layer activation vector
    base_auroc, _ = lodo_auroc(Xbest, yv, dv, docs_v, args.C)
    # +CONFLICT: same vector with the conflict scalar appended as one feature
    Xaug = np.hstack([Xbest, conf])
    aug_auroc, _ = lodo_auroc(Xaug, yv, dv, docs_v, args.C)

    logger.info("=" * 60)
    logger.info("CORRECTED CONFLICT-AS-FEATURE TEST (apples-to-apples)")
    logger.info("  baseline (best-layer probe)      : %.4f", base_auroc)
    logger.info("  + conflict scalar appended       : %.4f", aug_auroc)
    logger.info("  delta                            : %+.4f", aug_auroc - base_auroc)
    logger.info("-" * 60)
    if abs(aug_auroc - base_auroc) < 0.005:
        logger.info("READ: NULL. Layer-disagreement adds nothing over the probe.")
        logger.info("Consistent with the 3-signal result: the single mid-layer")
        logger.info("probe already saturates the achievable signal.")
    elif aug_auroc > base_auroc:
        logger.info("READ: conflict adds %+.4f -- a real (small) improvement.", aug_auroc-base_auroc)
    else:
        logger.info("READ: conflict HURTS by %+.4f (adds noise).", aug_auroc-base_auroc)
    logger.info("=" * 60)

    out = cfg.artifacts_path / "results" / args.out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "best_layer": int(best_layer),
        "baseline_auroc": base_auroc,
        "with_conflict_auroc": aug_auroc,
        "delta": aug_auroc - base_auroc,
        "n_fields": int(valid.sum()), "n_errors": int(yv.sum()),
    }, indent=2))
    logger.info("Saved to %s", out)
    return 0


if __name__ == "__main__":
    import sys; sys.exit(main())
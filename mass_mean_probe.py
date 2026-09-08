#!/usr/bin/env python3
"""Mass-mean probes vs logistic regression, under identical nested LODO.

METHODS COMPARED
----------------
All three see the same fields, the same folds and the same layers; only the way
the probe direction is obtained differs.

  logreg     Logistic regression (the pipeline's current probe).
             Direction fitted by maximum likelihood with L2 regularisation.

  mass_mean  Difference of class means: theta = mean(errors) - mean(correct),
             scored by projection onto theta. No fitting, no regularisation,
             no optimisation (Marks & Tegmark 2024). If this matches logreg,
             the signal is a simple direction in activation space rather than
             something the classifier constructs.

  mass_mean_iid  The covariance-adjusted variant: theta_adj = Sigma^-1 theta,
             where Sigma is the pooled within-class covariance estimated on the
             training fold. Equivalent to LDA up to scaling. Uses Ledoit-Wolf
             shrinkage, because with ~2,300 samples in 2,560 dimensions the
             empirical covariance is singular and cannot be inverted directly.

PROTOCOL
--------
Nested LODO, matching scripts/05b_nested_lodo.py: an outer loop holds out a
document; an inner loop selects the layer using only the remaining documents;
the selected layer is refitted on all non-test documents and scores the held-out
one. Reported as pooled out-of-fold AUROC over every held-out prediction, plus
the per-fold mean, plus the distribution of layers each method selected.

Also reports a per-layer fixed-layer LODO table for each method, which is NOT
nested and whose maximum should not be quoted as a headline -- it is there to
show the shape of each method's layer curve.

CPU only.

Usage:
  python mass_mean_probe.py \
      --domain qwen35_4b_pooled_alltokens \
      --layers 14 16 18 20 22 \
      --exclude-domains finance/10kq
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True)
    p.add_argument("--layers", type=int, nargs="+",
                   default=[14, 16, 18, 20, 22])
    p.add_argument("--exclude-domains", nargs="*", default=None)
    p.add_argument("--include-domains", nargs="*", default=None)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--out", default=None)
    return p.parse_args()


def _domain_ok(domain, include, exclude):
    if include is not None and domain not in include:
        return False
    if exclude is not None and domain in exclude:
        return False
    return True


def load_data(domain, layers, include, exclude):
    """{layer: X}, y, doc_ids. Keeps only fields present at every candidate
    layer, so all layers and all methods see an identical field set."""
    labels_dir = Path(f"artifacts/{domain}/labels")
    acts_dir = Path(f"artifacts/{domain}/activations")
    rows = []
    skipped = {}
    for lp in sorted(labels_dir.glob("*.json")):
        if lp.name.startswith("_"):
            continue
        data = json.load(lp.open())
        dom = data.get("domain")
        if not _domain_ok(dom, include, exclude):
            skipped[dom] = skipped.get(dom, 0) + 1
            continue
        doc_id = data.get("doc_id", lp.stem)
        npz = acts_dir / f"{doc_id}.npz"
        if not npz.exists():
            continue
        with np.load(npz) as acts:
            for fld in data.get("labels", []):
                if not fld.get("extracted_present", True):
                    continue
                ps = fld["path_str"]
                vecs, ok = {}, True
                for L in layers:
                    key = f"{ps}__layer{L}"
                    if key not in acts:
                        ok = False
                        break
                    v = acts[key].astype(np.float32)
                    if v.ndim > 1:
                        v = v[-1]
                    vecs[L] = v
                if not ok:
                    continue
                rows.append((doc_id, int(fld.get("is_error", 0)), vecs))
    if skipped:
        logger.info("Domain filter skipped: %s",
                    ", ".join(f"{d}={n}" for d, n in sorted(skipped.items())))
    doc_ids = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    X = {L: np.stack([r[2][L] for r in rows]) for L in layers}
    return X, y, doc_ids


# ---------------------------------------------------------------------------
# The three probe methods. Each takes (X_train, y_train, X_test) and returns
# a score per test row. Higher score = more likely to be an error.
# ---------------------------------------------------------------------------

def fit_logreg(Xtr, ytr, Xte, C):
    clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced",
                             random_state=42)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def fit_mass_mean(Xtr, ytr, Xte, C=None):
    """theta = mean(errors) - mean(correct); score = x . theta.

    AUROC is invariant to monotone transforms, so the raw projection needs no
    sigmoid or centring -- only the ordering matters."""
    mu_pos = Xtr[ytr == 1].mean(axis=0)
    mu_neg = Xtr[ytr == 0].mean(axis=0)
    theta = mu_pos - mu_neg
    return Xte @ theta


def fit_mass_mean_iid(Xtr, ytr, Xte, C=None):
    """Covariance-adjusted: theta_adj = Sigma^-1 (mu_pos - mu_neg).

    Ledoit-Wolf shrinkage is required: with ~2,300 training samples in 2,560
    dimensions the empirical covariance is singular. Shrinkage also stabilises
    the inverse, which is otherwise dominated by near-zero eigenvalues."""
    mu_pos = Xtr[ytr == 1].mean(axis=0)
    mu_neg = Xtr[ytr == 0].mean(axis=0)
    theta = mu_pos - mu_neg

    # Pooled within-class covariance: centre each class, then estimate jointly.
    Xc = np.vstack([Xtr[ytr == 1] - mu_pos, Xtr[ytr == 0] - mu_neg])
    lw = LedoitWolf(assume_centered=True).fit(Xc)
    theta_adj = lw.precision_ @ theta
    return Xte @ theta_adj


METHODS = {
    "logreg": fit_logreg,
    "mass_mean": fit_mass_mean,
    "mass_mean_iid": fit_mass_mean_iid,
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def lodo_scores(XL, y, doc_ids, docs, fit_fn, C):
    """Plain LODO at one fixed layer. Returns out-of-fold scores (NaN where a
    fold could not be scored)."""
    oof = np.full(len(y), np.nan)
    for d in docs:
        te = doc_ids == d
        tr = ~te
        if y[tr].sum() == 0 or y[tr].sum() == tr.sum():
            continue
        oof[te] = fit_fn(XL[tr], y[tr], XL[te], C)
    return oof


def safe_auroc(y, s):
    m = ~np.isnan(s)
    if m.sum() == 0 or y[m].sum() in (0, m.sum()):
        return float("nan")
    return roc_auc_score(y[m], s[m])


def per_fold_auroc(y, s, doc_ids, docs):
    vals = []
    for d in docs:
        m = (doc_ids == d) & ~np.isnan(s)
        if m.sum() < 2 or y[m].sum() in (0, m.sum()):
            continue
        vals.append(roc_auc_score(y[m], s[m]))
    return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals \
        else (float("nan"), float("nan"), 0)


def nested_lodo(X, y, doc_ids, docs, layers, fit_fn, C):
    """Outer loop holds out a document; inner loop picks the layer on the rest."""
    oof = np.full(len(y), np.nan)
    selected = []
    for d in docs:
        inner_docs = [x for x in docs if x != d]
        te = doc_ids == d
        tr = ~te
        if y[tr].sum() == 0 or y[tr].sum() == tr.sum():
            continue

        best_layer, best_score = None, None
        for L in layers:
            # inner LODO over the training documents only
            inner = np.full(len(y), np.nan)
            for d2 in inner_docs:
                te2 = doc_ids == d2
                tr2 = tr & ~te2
                if y[tr2].sum() == 0 or y[tr2].sum() == tr2.sum():
                    continue
                inner[te2] = fit_fn(X[L][tr2], y[tr2], X[L][te2], C)
            a = safe_auroc(y, inner)
            if np.isnan(a):
                continue
            if best_score is None or a > best_score:
                best_score, best_layer = a, L

        if best_layer is None:
            continue
        selected.append(best_layer)
        oof[te] = fit_fn(X[best_layer][tr], y[tr], X[best_layer][te], C)

    dist = {str(L): selected.count(L) for L in sorted(set(selected))}
    return oof, dist


def main():
    args = parse_args()
    X, y, doc_ids = load_data(args.domain, args.layers,
                              args.include_domains, args.exclude_domains)
    docs = list(dict.fromkeys(doc_ids.tolist()))
    logger.info("Loaded %d fields across %d documents (%d errors, %.1f%%)",
                len(y), len(docs), int(y.sum()), 100 * y.mean())
    logger.info("Candidate layers: %s", args.layers)

    results = {}

    # ---- fixed-layer LODO curves (NOT nested; shape only) -----------------
    logger.info("=" * 72)
    logger.info("FIXED-LAYER LODO (pooled-OOF AUROC) -- shape of each curve.")
    logger.info("NOT nested: do not quote the maximum as a headline.")
    header = "  layer  " + "  ".join(f"{m:>14s}" for m in METHODS)
    logger.info(header)
    per_layer = {m: {} for m in METHODS}
    for L in args.layers:
        cells = []
        for name, fn in METHODS.items():
            s = lodo_scores(X[L], y, doc_ids, docs, fn, args.C)
            a = safe_auroc(y, s)
            per_layer[name][str(L)] = a
            cells.append(f"{a:14.4f}")
        logger.info("  %-6d %s", L, "  ".join(cells))
    results["fixed_layer_lodo"] = per_layer

    # ---- nested LODO (the comparable numbers) -----------------------------
    logger.info("=" * 72)
    logger.info("NESTED LODO -- layer selected on inner documents only.")
    nested = {}
    for name, fn in METHODS.items():
        oof, dist = nested_lodo(X, y, doc_ids, docs, args.layers, fn, args.C)
        pooled = safe_auroc(y, oof)
        pf_mean, pf_std, pf_n = per_fold_auroc(y, oof, doc_ids, docs)
        nested[name] = {
            "pooled_oof_auroc": pooled,
            "per_fold_auroc_mean": pf_mean,
            "per_fold_auroc_std": pf_std,
            "n_valid_folds": pf_n,
            "layers_selected": dist,
        }
        logger.info("  %-14s pooled-OOF %.4f | per-fold %.4f +/- %.4f (n=%d)",
                    name, pooled, pf_mean, pf_std, pf_n)
        logger.info("  %-14s layers: %s", "", dist)
    results["nested_lodo"] = nested

    # ---- reading ----------------------------------------------------------
    lg = nested["logreg"]["pooled_oof_auroc"]
    mm = nested["mass_mean"]["pooled_oof_auroc"]
    mi = nested["mass_mean_iid"]["pooled_oof_auroc"]
    logger.info("=" * 72)
    logger.info("READING")
    logger.info("  mass_mean - logreg      = %+.4f", mm - lg)
    logger.info("  mass_mean_iid - logreg  = %+.4f", mi - lg)
    logger.info("")
    logger.info("  If difference-of-means lands close to logistic regression,")
    logger.info("  the error signal is a simple DIRECTION in activation space,")
    logger.info("  not something the classifier constructs -- a fourth")
    logger.info("  independent line of evidence for the saturation result.")
    logger.info("  If it lands well below, the fitting is doing real work and")
    logger.info("  the signal is not a clean linear separation.")
    logger.info("=" * 72)

    out = Path(args.out) if args.out else Path(
        f"artifacts/{args.domain}/results/mass_mean_comparison.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "domain": args.domain,
        "candidate_layers": args.layers,
        "n_fields": int(len(y)),
        "n_docs": len(docs),
        "error_rate": float(y.mean()),
        **results,
    }, indent=2))
    logger.info("Saved to %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
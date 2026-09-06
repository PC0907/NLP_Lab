#!/usr/bin/env python3
"""Fully nested safe-override regeneration (cached LODO scores).

WHAT THIS FIXES
---------------
scripts/safe_override_rescore.py holds out the THRESHOLD (tau is chosen
leave-one-document-out) but NOT the LAYER, which is a fixed command-line
argument picked from analyses computed over every document -- including the
ones subsequently scored. This script nests both.

STRUCTURE
---------
Precompute, once per candidate layer, a leave-one-document-out score for every
field: the probe scoring a field was trained on all documents except that
field's own. Then, for each held-out document D:

    for each candidate layer L:
        rank (L, tau) on the OTHER documents using the cached L-scores
    apply the winning (L, tau) to D, with the probe RETRAINED on the
    inner documents only

APPROXIMATION (state this in the paper)
---------------------------------------
The exact nested form would recompute inner scores per outer fold, so that no
probe involved in selecting (L, tau) for D had ever seen D. That is ~3,800
probe fits and runs for hours. Caching reduces it to ~140 fits: an inner
document's cached score comes from a probe trained on every document but its
own, which for outer fold D includes D. So the SELECTION is marginally
informed by D; the SCORING that produces the reported number is not, because
the outer probe is retrained on inner documents only. With 28 documents the
influence of any single one on the inner ranking is small, and this trade is
standard in nested-CV implementations.

INNER OBJECTIVE
---------------
  --objective auroc : pick L by detection AUROC on inner docs (as 05b does),
                      then pick tau for regeneration net.
  --objective net   : pick (L, tau) jointly by regeneration net. Optimises the
                      reported quantity, but searches a 2-D grid on ~27
                      documents and can overfit the inner folds.
Both are reported by default, because on this data they disagree: layer 16
wins on AUROC, layer 18 on gating net.

Reuses rescore_regen.py for the matcher, so fixed/broke outcomes match the
existing script exactly. CPU only.

Usage:
  python nested_safe_override.py \
      --domain qwen35_4b_pooled_alltokens \
      --cache artifacts/qwen35_4b_pooled_alltokens/results/regen_cache_pooled_v2.json \
      --layers 14 16 18 20 22 --objective both
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "scripts")
from importlib import import_module
_rs = import_module("rescore_regen")
strict_ok = _rs.strict_ok
lenient_ok = _rs.lenient_ok
is_scorable_lenient = _rs.is_scorable_lenient
load_labels = _rs.load_labels

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--layers", type=int, nargs="+", default=[14, 16, 18, 20, 22])
    p.add_argument("--objective", choices=["net", "auroc", "both"], default="both")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--out", default=None)
    return p.parse_args()


def load_activations(domain: str, layers: list[int]):
    """{(doc,path): {layer: vec}}, {(doc,path): y}, [docs].
    Keeps only fields present at ALL candidate layers, so every layer is
    compared on an identical field set."""
    labels_dir = Path(f"artifacts/{domain}/labels")
    acts_dir = Path(f"artifacts/{domain}/activations")
    feats, ys = {}, {}
    for lp in sorted(labels_dir.glob("*.json")):
        if lp.name.startswith("_"):
            continue
        data = json.load(lp.open())
        doc_id = data.get("doc_id", lp.stem)
        npz_path = acts_dir / f"{doc_id}.npz"
        if not npz_path.exists():
            continue
        with np.load(npz_path) as acts:
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
                feats[(doc_id, ps)] = vecs
                ys[(doc_id, ps)] = int(fld.get("is_error", 0))
    return feats, ys, sorted({k[0] for k in feats})


def build_rows(cache, labels, mode):
    ok = strict_ok if mode == "strict" else lenient_ok
    rows = []
    for key, entry in cache.items():
        doc_id, path = key.split("::", 1)
        lab = labels.get((doc_id, path))
        if lab is None:
            continue
        gold = lab.get("gold_value")
        if mode == "lenient" and not is_scorable_lenient(path, gold):
            continue
        was_err = int(lab.get("is_error", 0)) == 1
        now_ok = ok(gold, entry["new_value"])
        delta = +1 if (was_err and now_ok) else (-1 if (not was_err and not now_ok) else 0)
        rows.append({"doc_id": doc_id, "path": path, "delta": delta})
    return rows


def fit_predict(X_tr, y_tr, X_te, C):
    if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
        return None
    clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced",
                             random_state=42)
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]


def cached_lodo_scores(feats, ys, docs, layers, C):
    """Per layer, a leave-one-document-out score for every field.
    ~len(layers) * len(docs) fits in total, rather than that many per fold."""
    keys = list(feats.keys())
    key_doc = np.array([k[0] for k in keys])
    y_all = np.array([ys[k] for k in keys])
    scores = {}
    for L in layers:
        t0 = time.time()
        X = np.stack([feats[k][L] for k in keys])
        s = np.full(len(keys), np.nan)
        for d in docs:
            te = key_doc == d
            tr = ~te
            p = fit_predict(X[tr], y_all[tr], X[te], C)
            if p is not None:
                s[te] = p
        scores[L] = dict(zip(keys, s))
        logger.info("  layer %-3d cached (%.0fs)", L, time.time() - t0)
    return scores


def gate(rows, pscores, tau):
    fixed = broke = n_over = 0
    for r in rows:
        s = pscores.get((r["doc_id"], r["path"]))
        if s is None or (isinstance(s, float) and np.isnan(s)) or s < tau:
            continue
        n_over += 1
        if r["delta"] == +1:
            fixed += 1
        elif r["delta"] == -1:
            broke += 1
    return n_over, fixed, broke, fixed - broke


def nested_eval(rows, feats, ys, docs, layers, taus, C, objective, cached):
    keys = list(feats.keys())
    key_doc = np.array([k[0] for k in keys])
    y_all = np.array([ys[k] for k in keys])
    tot_fixed = tot_broke = 0
    chosen = []

    for d in docs:
        inner_rows = [r for r in rows if r["doc_id"] != d]
        test_rows = [r for r in rows if r["doc_id"] == d]
        if not test_rows:
            continue

        best = None  # (criterion, layer, tau)
        for L in layers:
            sc = cached[L]
            inner = {k: v for k, v in sc.items()
                     if k[0] != d and not np.isnan(v)}
            if not inner:
                continue

            best_tau, best_net = None, None
            for t in taus:
                _, _, _, net = gate(inner_rows, inner, t)
                if best_net is None or net > best_net:
                    best_net, best_tau = net, t

            if objective == "auroc":
                yv = np.array([ys[k] for k in inner])
                sv = np.array([inner[k] for k in inner])
                if yv.sum() in (0, len(yv)):
                    continue
                crit = roc_auc_score(yv, sv)
            else:
                crit = best_net

            if best is None or crit > best[0]:
                best = (crit, L, best_tau)

        if best is None:
            continue
        _, L_sel, tau_sel = best
        chosen.append((L_sel, tau_sel))

        # Outer scoring: retrain at the selected layer on inner documents only.
        X = np.stack([feats[k][L_sel] for k in keys])
        te = key_doc == d
        p = fit_predict(X[~te], y_all[~te], X[te], C)
        if p is None:
            continue
        test_scores = {k: v for k, v in zip([k for k, m in zip(keys, te) if m], p)}
        _, f, b, _ = gate(test_rows, test_scores, tau_sel)
        tot_fixed += f
        tot_broke += b

    return {
        "held_out_fixed": tot_fixed,
        "held_out_broke": tot_broke,
        "held_out_net": tot_fixed - tot_broke,
        "n_folds": len(chosen),
        "layer_dist": {str(l): sum(1 for c in chosen if c[0] == l)
                       for l in sorted({c[0] for c in chosen})},
        "tau_dist": {f"{t:.2f}": sum(1 for c in chosen if abs(c[1] - t) < 1e-9)
                     for t in sorted({c[1] for c in chosen})},
    }


def main():
    args = parse_args()
    cache = json.load(open(args.cache))
    labels = load_labels(args.domain)

    logger.info("Loading activations for layers %s ...", args.layers)
    feats, ys, docs = load_activations(args.domain, args.layers)
    logger.info("  %d fields across %d documents (%d errors, %.1f%%)",
                len(feats), len(docs), sum(ys.values()),
                100 * sum(ys.values()) / max(len(ys), 1))

    logger.info("Caching per-layer LODO scores (%d layers x %d docs fits) ...",
                len(args.layers), len(docs))
    cached = cached_lodo_scores(feats, ys, docs, args.layers, args.C)

    taus = [round(x, 2) for x in np.linspace(0.0, 0.95, 20)]
    objectives = ["auroc", "net"] if args.objective == "both" else [args.objective]

    results = {}
    for mode in ("strict", "lenient"):
        rows = build_rows(cache, labels, mode)
        logger.info("=" * 72)
        logger.info("MODE: %s  (%d candidates)", mode.upper(), len(rows))
        for obj in objectives:
            r = nested_eval(rows, feats, ys, docs, args.layers, taus,
                            args.C, obj, cached)
            logger.info("  [%s] held-out net = %+d  (fixed %d, broke %d), %d folds",
                        obj, r["held_out_net"], r["held_out_fixed"],
                        r["held_out_broke"], r["n_folds"])
            logger.info("       layers: %s", r["layer_dist"])
            logger.info("       taus  : %s", r["tau_dist"])
            results[f"{mode}_{obj}"] = r

    logger.info("=" * 72)
    logger.info("Layer AND threshold both selected on inner documents.")
    logger.info("Fixed-layer comparison: L18 -> +44 lenient / +17 strict;")
    logger.info("                        L16 -> +34 lenient / +12 strict.")

    out = Path(args.out) if args.out else Path(
        f"artifacts/{args.domain}/results/nested_safe_override.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "cache": args.cache,
        "candidate_layers": args.layers,
        "note": "Inner LODO scores cached per layer; see module docstring for "
                "the approximation this introduces.",
        "results": results,
    }, indent=2))
    logger.info("Saved to %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
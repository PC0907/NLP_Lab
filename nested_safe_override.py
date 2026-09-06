#!/usr/bin/env python3
"""Fully nested safe-override regeneration.

WHAT THIS FIXES
---------------
scripts/safe_override_rescore.py holds out the THRESHOLD (tau is picked
leave-one-document-out) but NOT the LAYER: the layer is a fixed command-line
argument, chosen from analyses computed over all documents including the ones
subsequently scored. One hyperparameter is nested, the other is not.

This script nests both. For each held-out document D:
    for each candidate layer L:
        train a probe on the OTHER documents' activations at layer L
        score the other documents, sweep tau, record the best (L, tau)
    apply the winning (L, tau) to D and accumulate fixed / broke / net

The probe is RETRAINED per fold. Loading artifacts/.../probe_layer*.pkl would
leak, because those probes saw every document including D.

INNER OBJECTIVE
---------------
Two choices, both reported, because they can disagree -- on this data layer 16
wins on detection AUROC while layer 18 wins on gating net:
  --objective net    : pick (L, tau) maximising regeneration net on the inner
                       documents. Coherent (optimises what is reported) but
                       searches a 2-D grid on ~27 documents, so it can overfit
                       the inner folds.
  --objective auroc  : pick L maximising detection AUROC on the inner
                       documents (as 05b does), then pick tau for net.
                       Smaller search, closer to the existing protocol.

Reuses rescore_regen.py for the matcher, so fixed/broke outcomes are identical
to the existing script. CPU only.

Usage:
  python nested_safe_override.py \
      --domain qwen35_4b_pooled_alltokens \
      --cache artifacts/qwen35_4b_pooled_alltokens/results/regen_cache_pooled_v2.json \
      --layers 14 16 18 20 22 \
      --objective both
"""
from __future__ import annotations

import argparse
import json
import sys
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
    p.add_argument("--domain", required=True,
                   help="artifacts/<domain>, e.g. qwen35_4b_pooled_alltokens")
    p.add_argument("--cache", required=True)
    p.add_argument("--layers", type=int, nargs="+", default=[14, 16, 18, 20, 22],
                   help="Candidate layers for the inner selection.")
    p.add_argument("--objective", choices=["net", "auroc", "both"], default="both")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--out", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading: activations + labels, per layer, aligned by (doc_id, path)
# ---------------------------------------------------------------------------

def load_activations(domain: str, layers: list[int]):
    """Return {(doc_id, path): {layer: vec}}, y dict, and doc list.

    Only fields present at ALL candidate layers are kept, so every layer is
    compared on an identical field set.
    """
    labels_dir = Path(f"artifacts/{domain}/labels")
    acts_dir = Path(f"artifacts/{domain}/activations")

    feats: dict[tuple[str, str], dict[int, np.ndarray]] = {}
    ys: dict[tuple[str, str], int] = {}

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
                vecs = {}
                ok = True
                for L in layers:
                    key = f"{ps}__layer{L}"
                    if key not in acts:
                        ok = False
                        break
                    v = acts[key].astype(np.float32)
                    if v.ndim > 1:          # all-tokens storage -> last token
                        v = v[-1]
                    vecs[L] = v
                if not ok:
                    continue
                feats[(doc_id, ps)] = vecs
                ys[(doc_id, ps)] = int(fld.get("is_error", 0))
    docs = sorted({k[0] for k in feats})
    return feats, ys, docs


def build_rows(cache, labels, mode):
    """One row per cached candidate: doc, path, and the delta if overridden."""
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
        if was_err and now_ok:
            delta = +1
        elif (not was_err) and (not now_ok):
            delta = -1
        else:
            delta = 0
        rows.append({"doc_id": doc_id, "path": path, "delta": delta})
    return rows


# ---------------------------------------------------------------------------
# Probe training / scoring
# ---------------------------------------------------------------------------

def train_score(feats, ys, layer, train_keys, score_keys, C):
    """Fit a probe at `layer` on train_keys, return {key: P(error)} for
    score_keys. Returns None if the training set is single-class."""
    Xtr = np.stack([feats[k][layer] for k in train_keys])
    ytr = np.array([ys[k] for k in train_keys])
    if ytr.sum() == 0 or ytr.sum() == len(ytr):
        return None
    clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced",
                             random_state=42)
    clf.fit(Xtr, ytr)
    Xte = np.stack([feats[k][layer] for k in score_keys])
    p = clf.predict_proba(Xte)[:, 1]
    return dict(zip(score_keys, p))


def gate(rows, pscores, tau):
    """Apply the override gate at tau. Rows lacking a score are not overridden."""
    fixed = broke = n_over = 0
    for r in rows:
        s = pscores.get((r["doc_id"], r["path"]))
        if s is None or s < tau:
            continue
        n_over += 1
        if r["delta"] == +1:
            fixed += 1
        elif r["delta"] == -1:
            broke += 1
    return n_over, fixed, broke, fixed - broke


# ---------------------------------------------------------------------------
# Nested evaluation
# ---------------------------------------------------------------------------

def nested_eval(rows, feats, ys, docs, layers, taus, C, objective):
    """Outer loop over documents; inner loop selects (layer, tau) on the rest."""
    all_keys = list(feats.keys())
    tot_fixed = tot_broke = 0
    chosen: list[tuple[int, float]] = []

    for d in docs:
        inner_docs = [x for x in docs if x != d]
        inner_rows = [r for r in rows if r["doc_id"] != d]
        test_rows = [r for r in rows if r["doc_id"] == d]
        if not test_rows:
            continue

        train_keys = [k for k in all_keys if k[0] != d]
        test_keys = [k for k in all_keys if k[0] == d]
        if not test_keys or not train_keys:
            continue

        best = None  # (score, layer, tau)

        for L in layers:
            # --- inner: score the inner documents out-of-fold ---------------
            inner_scores: dict[tuple[str, str], float] = {}
            inner_auroc_parts = []
            for d2 in inner_docs:
                tr = [k for k in train_keys if k[0] != d2]
                te = [k for k in train_keys if k[0] == d2]
                if not te:
                    continue
                sc = train_score(feats, ys, L, tr, te, C)
                if sc is None:
                    continue
                inner_scores.update(sc)
            if not inner_scores:
                continue

            if objective == "auroc":
                yv = np.array([ys[k] for k in inner_scores])
                sv = np.array([inner_scores[k] for k in inner_scores])
                if yv.sum() in (0, len(yv)):
                    continue
                layer_score = roc_auc_score(yv, sv)
                # tau still chosen for net, given this layer
                best_tau, best_net = None, None
                for t in taus:
                    _, _, _, net = gate(inner_rows, inner_scores, t)
                    if best_net is None or net > best_net:
                        best_net, best_tau = net, t
                cand = (layer_score, L, best_tau)
            else:  # objective == "net"
                best_tau, best_net = None, None
                for t in taus:
                    _, _, _, net = gate(inner_rows, inner_scores, t)
                    if best_net is None or net > best_net:
                        best_net, best_tau = net, t
                cand = (best_net, L, best_tau)

            if best is None or cand[0] > best[0]:
                best = cand

        if best is None:
            continue
        _, L_sel, tau_sel = best
        chosen.append((L_sel, tau_sel))

        # --- outer: retrain at the selected layer on ALL inner docs ---------
        test_scores = train_score(feats, ys, L_sel, train_keys, test_keys, C)
        if test_scores is None:
            continue
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

    taus = [round(x, 2) for x in np.linspace(0.0, 0.95, 20)]
    objectives = ["auroc", "net"] if args.objective == "both" else [args.objective]

    results = {}
    for mode in ("strict", "lenient"):
        rows = build_rows(cache, labels, mode)
        logger.info("=" * 72)
        logger.info("MODE: %s  (%d candidates)", mode.upper(), len(rows))
        for obj in objectives:
            logger.info("-" * 72)
            logger.info("Inner objective: %s", obj)
            r = nested_eval(rows, feats, ys, docs, args.layers, taus, args.C, obj)
            logger.info("  held-out net = %+d  (fixed %d, broke %d) over %d folds",
                        r["held_out_net"], r["held_out_fixed"],
                        r["held_out_broke"], r["n_folds"])
            logger.info("  layers selected: %s", r["layer_dist"])
            logger.info("  taus selected  : %s", r["tau_dist"])
            results[f"{mode}_{obj}"] = r

    logger.info("=" * 72)
    logger.info("Both layer AND threshold were selected on inner documents only.")
    logger.info("Compare against the fixed-layer script, where only tau was held out.")

    out = Path(args.out) if args.out else (
        Path(f"artifacts/{args.domain}/results/nested_safe_override.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "cache": args.cache,
        "candidate_layers": args.layers,
        "results": results,
    }, indent=2))
    logger.info("Saved to %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
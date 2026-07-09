"""Stage 6 (reasoning-trace paper): answer-token vs reasoning-fused probe, LODO.

THE HEADLINE EXPERIMENT. For a reasoning model (DeepSeek-R1) we compare, under
leave-one-document-out (LODO) cross-validation, per-field error-detection probes
built from different feature sets:

  - answer        : the field's answer-token activation (the existing approach,
                    and the partner branch's approach).
  - fused_mean    : answer + mean-pooled reasoning-trace vector (doc-level).
  - fused_last    : answer + the </think> reasoning-summary vector (doc-level).
  - fused_both    : answer + both reasoning vectors.

The question: does an explicit reasoning-trace representation add error-relevant
signal beyond what is already linearly accessible at the answer token? A
positive, LODO-robust gap is the paper's central claim.

Reads the reserved reasoning keys written by 01_extract.save_activations
(__reasoning_mean__layerN / __reasoning_last__layerN).

Usage:
    python scripts/06_reasoning_fusion_lodo.py --config CFG --layers 16 19 23 26 --jobs -1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from probe_extraction.config import load_config
from probe_extraction.utils.logging import setup_logging

logger = logging.getLogger(__name__)

VARIANTS = ("answer", "fused_mean", "fused_last", "fused_both")

# LODO retrains one probe per held-out doc. 200 lbfgs iters converge on these
# standardized features (max_iter=1000 just burned time hitting the cap); the
# fold loop is embarrassingly parallel, so we fan it across all cores.
_MAX_ITER = 200


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reasoning-fused vs answer-only probe (LODO).")
    p.add_argument("--config", required=True)
    p.add_argument("--layers", type=int, nargs="*", default=None,
                   help="Layers to evaluate (default: all in config).")
    p.add_argument("--jobs", type=int, default=-1,
                   help="Parallel workers for the LODO fold loop (default: all cores).")
    p.add_argument("--out-name", type=str, default="reasoning_fusion_lodo.json")
    return p.parse_args()


def load_fusion_docs(activations_dir: Path, labels_dir: Path, layers: list[int]):
    docs = []
    n_skip_no_reasoning = 0
    for labels_path in sorted(labels_dir.glob("*.json")):
        if labels_path.name.startswith("_"):
            continue
        doc_id = labels_path.stem
        data = json.load(labels_path.open())
        act_path = activations_dir / f"{doc_id}.npz"
        if not act_path.exists():
            continue
        with np.load(act_path) as npz:
            keys = set(npz.keys())
            reasoning_mean = {L: npz[f"__reasoning_mean__layer{L}"].astype(np.float32)
                              for L in layers if f"__reasoning_mean__layer{L}" in keys}
            reasoning_last = {L: npz[f"__reasoning_last__layer{L}"].astype(np.float32)
                              for L in layers if f"__reasoning_last__layer{L}" in keys}
            if not reasoning_mean or not reasoning_last:
                n_skip_no_reasoning += 1
                continue

            per_layer_X = {L: [] for L in layers}
            y_list = []
            for fld in data.get("labels", []):
                if not fld.get("extracted_present", False):
                    continue
                ps = fld["path_str"]
                fld_keys = [f"{ps}__layer{L}" for L in layers]
                if not all(k in keys for k in fld_keys):
                    continue
                for L in layers:
                    per_layer_X[L].append(npz[f"{ps}__layer{L}"].astype(np.float32))
                y_list.append(int(fld["is_error"]))

            if not y_list:
                continue
            answer = {L: np.stack(per_layer_X[L], axis=0) for L in layers}
            docs.append({
                "doc_id": doc_id,
                "y": np.array(y_list, dtype=np.int64),
                "answer": answer,
                "reasoning_mean": reasoning_mean,
                "reasoning_last": reasoning_last,
            })
    if n_skip_no_reasoning:
        logger.info("Skipped %d docs with no reasoning-trace vectors.", n_skip_no_reasoning)
    return docs


def build_features(doc: dict, layer: int, variant: str) -> np.ndarray:
    ans = doc["answer"][layer]
    n = ans.shape[0]
    if variant == "answer":
        return ans
    rmean = np.tile(doc["reasoning_mean"][layer], (n, 1))
    rlast = np.tile(doc["reasoning_last"][layer], (n, 1))
    if variant == "fused_mean":
        return np.concatenate([ans, rmean], axis=1)
    if variant == "fused_last":
        return np.concatenate([ans, rlast], axis=1)
    if variant == "fused_both":
        return np.concatenate([ans, rmean, rlast], axis=1)
    raise ValueError(f"Unknown variant: {variant}")


def _standardize(train: np.ndarray, test: np.ndarray):
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (train - mu) / sd, (test - mu) / sd


def _fit_fold(full_X: np.ndarray, full_y: np.ndarray, lo: int, hi: int, C: float):
    """Fit one LODO fold: train on all rows except [lo:hi), score that slice."""
    mask = np.ones(full_y.shape[0], dtype=bool)
    mask[lo:hi] = False
    y_tr = full_y[mask]
    if y_tr.sum() in (0, len(y_tr)):
        return None
    X_tr = full_X[mask]
    X_te = full_X[lo:hi]
    y_te = full_y[lo:hi]
    X_tr_s, X_te_s = _standardize(X_tr.astype(np.float64), X_te.astype(np.float64))
    clf = LogisticRegression(C=C, max_iter=_MAX_ITER, class_weight="balanced")
    clf.fit(X_tr_s, y_tr)
    proba = clf.predict_proba(X_te_s)[:, 1]
    fold_auroc = (float(roc_auc_score(y_te, proba))
                  if y_te.sum() not in (0, len(y_te)) else None)
    return y_te, proba, fold_auroc


def lodo_eval(docs: list[dict], layer: int, variant: str, *, C: float = 1.0,
              n_jobs: int = -1) -> dict:
    """Leave-one-document-out evaluation for one (layer, variant).

    per_doc_auroc_mean : mean within-doc AUROC (doc-level reasoning ~ null here).
    pooled_oof_auroc   : global field ranking across docs (reasoning CAN help).
    The per-document folds run in parallel; the shared feature matrix is built
    once and memmapped to workers by joblib.
    """
    feats = [build_features(d, layer, variant).astype(np.float64) for d in docs]
    full_X = np.concatenate(feats, axis=0)
    full_y = np.concatenate([d["y"] for d in docs], axis=0)
    bounds = np.concatenate([[0], np.cumsum([len(d["y"]) for d in docs])])

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_fit_fold)(full_X, full_y, int(bounds[i]), int(bounds[i + 1]), C)
        for i in range(len(docs))
    )

    fold_aurocs = []
    oof_y, oof_p = [], []
    for r in results:
        if r is None:
            continue
        y_te, proba, fold_auroc = r
        oof_y.append(y_te)
        oof_p.append(proba)
        if fold_auroc is not None:
            fold_aurocs.append(fold_auroc)

    res = {"layer": layer, "variant": variant, "n_valid_folds": len(fold_aurocs),
           "per_doc_auroc_mean": None, "per_doc_auroc_std": None,
           "pooled_oof_auroc": None}
    if fold_aurocs:
        res["per_doc_auroc_mean"] = float(np.mean(fold_aurocs))
        res["per_doc_auroc_std"] = float(np.std(fold_aurocs))
    if oof_y:
        ally = np.concatenate(oof_y)
        allp = np.concatenate(oof_p)
        if ally.sum() not in (0, len(ally)):
            res["pooled_oof_auroc"] = float(roc_auc_score(ally, allp))
    return res


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_dir=cfg.logging.log_dir,
                  log_name="06_reasoning_fusion_lodo", log_to_file=cfg.logging.log_to_file)

    layers = args.layers or cfg.activations.layers
    activations_dir = cfg.artifacts_path / "activations"
    labels_dir = cfg.artifacts_path / "labels"
    results_dir = cfg.artifacts_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    docs = load_fusion_docs(activations_dir, labels_dir, layers)
    if len(docs) < 2:
        logger.error("Need >=2 docs with reasoning vectors; found %d.", len(docs))
        return 1
    n_fields = sum(len(d["y"]) for d in docs)
    n_err = sum(int(d["y"].sum()) for d in docs)
    logger.info("Loaded %d docs, %d fields, %d errors (%.1f%%).",
                len(docs), n_fields, n_err, 100 * n_err / max(n_fields, 1))

    results = {v: {} for v in VARIANTS}
    for L in layers:
        for v in VARIANTS:
            logger.info("LODO: layer %d, variant %s ...", L, v)
            results[v][str(L)] = lodo_eval(docs, L, v, C=cfg.probe.C, n_jobs=args.jobs)

    def best(variant, metric):
        rows = [(int(L), r[metric]) for L, r in results[variant].items()
                if r.get(metric) is not None]
        return max(rows, key=lambda x: x[1]) if rows else (None, None)

    summary = {}
    for metric in ("per_doc_auroc_mean", "pooled_oof_auroc"):
        logger.info("=" * 70)
        logger.info("REASONING-FUSION LODO metric: %s (best layer per variant)", metric)
        summary[metric] = {}
        for v in VARIANTS:
            bl, ba = best(v, metric)
            summary[metric][v] = {"best_layer": bl, "best_auroc": ba}
            logger.info("  %-12s best layer %s : %s", v, bl,
                        f"{ba:.4f}" if ba is not None else "n/a")
        ans = summary[metric]["answer"]["best_auroc"]
        if ans is not None:
            for v in ("fused_mean", "fused_last", "fused_both"):
                bv = summary[metric][v]["best_auroc"]
                if bv is not None:
                    logger.info("  delta(%s - answer) = %+.4f", v, bv - ans)
    logger.info("=" * 70)

    out = {"layers": layers, "n_docs": len(docs), "n_fields": n_fields,
           "n_errors": n_err, "per_layer": results, "summary": summary}
    out_path = results_dir / args.out_name
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("Saved -> %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

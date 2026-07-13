"""Stage 7 (reasoning-trace paper): FIELD-LOCALIZED reasoning attribution, LODO.

The document-level reasoning fusion (Stage 6) was ~null because a doc-level
vector is constant across a document's fields and cannot change within-document
ranking. This stage fixes the granularity: for each extracted field it locates
where that field's VALUE is mentioned in the <think>...</think> trace and pools
exactly those tokens, giving a reasoning feature that VARIES across fields.

Two signals per field (see probe_extraction.extraction.reasoning_attribution):
  - attr_vec : pooled hidden states of the value-mentioning reasoning tokens.
  - scalars  : interpretable "was this value reasoned about, how often, where"
               features. A value absent from the reasoning trace is a
               hallucination red-flag -- a cheap, human-readable trust signal.

Variants compared under leave-one-document-out (LODO):
  - answer         : answer-token activation only (baseline / partner branch).
  - fused_attr     : answer + field-localized reasoning vector.
  - fused_scalars  : answer + interpretable mention scalars.
  - fused_both     : answer + reasoning vector + scalars.
  - scalars_only   : the mention scalars ALONE (interpretable baseline).

For every variant we report per_doc_auroc (within-doc detection) and pooled_oof
(global ranking), plus a PAIRED Wilcoxon significance test vs `answer` on the
per-document AUROCs -- the small, consistent gaps the marginal std hides show up
in a paired test.

Requires a Stage-1 run with REASONING_TOKEN_LAYERS set (so the per-token
reasoning states + token-string sidecars exist).

Usage:
    python scripts/07_reasoning_attribution_lodo.py --config CFG --layers 16 19 23 26 --jobs -1
"""

from __future__ import annotations

import argparse
import importlib.util
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

# Import the numpy-only attribution module by path (avoids importing the
# extraction package __init__, which pulls in torch).
_RA_PATH = Path(__file__).resolve().parents[1] / "src" / "probe_extraction" / "extraction" / "reasoning_attribution.py"
_spec = importlib.util.spec_from_file_location("reasoning_attribution", _RA_PATH)
ra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ra)

VARIANTS = ("answer", "fused_attr", "fused_scalars", "fused_both", "scalars_only")
_MAX_ITER = 200


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Field-localized reasoning attribution (LODO).")
    p.add_argument("--config", required=True)
    p.add_argument("--layers", type=int, nargs="*", default=None)
    p.add_argument("--jobs", type=int, default=-1)
    p.add_argument("--out-name", type=str, default="reasoning_attribution_lodo.json")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading + per-field attribution
# ---------------------------------------------------------------------------

def load_attribution_docs(activations_dir: Path, labels_dir: Path,
                          extractions_dir: Path, layers: list[int]):
    """Per document build: answer activations per field/layer, the
    field-localized reasoning attr_vec per field/layer, the scalar mention
    features per field, and the error labels y. Docs lacking per-token
    reasoning states are skipped (so all variants share the same doc set)."""
    docs = []
    n_skip = 0
    for labels_path in sorted(labels_dir.glob("*.json")):
        if labels_path.name.startswith("_"):
            continue
        doc_id = labels_path.stem
        data = json.load(labels_path.open())
        act_path = activations_dir / f"{doc_id}.npz"
        rtok_path = activations_dir / f"{doc_id}.rtokens.json"
        if not act_path.exists() or not rtok_path.exists():
            n_skip += 1
            continue

        token_strings = json.load(rtok_path.open())
        # Values come from the extraction metadata (path_str -> extracted value).
        values: dict[str, object] = {}
        ext_path = extractions_dir / f"{doc_id}.json"
        if ext_path.exists():
            for fld in json.load(ext_path.open()).get("fields", []):
                values[fld["path_str"]] = fld.get("value")

        with np.load(act_path) as npz:
            keys = set(npz.keys())
            rtok_states = {L: npz[f"__reasoning_tokens__layer{L}"]
                           for L in layers if f"__reasoning_tokens__layer{L}" in keys}
            if not rtok_states or not token_strings:
                n_skip += 1
                continue

            ans_rows = {L: [] for L in layers}
            attr_rows = {L: [] for L in layers}
            scalar_rows = []
            y_list = []
            for fld in data.get("labels", []):
                if not fld.get("extracted_present", False):
                    continue
                ps = fld["path_str"]
                if not all(f"{ps}__layer{L}" in keys for L in layers):
                    continue
                out = ra.attribute_field(token_strings, rtok_states, values.get(ps))
                for L in layers:
                    ans_rows[L].append(npz[f"{ps}__layer{L}"].astype(np.float32))
                    attr_rows[L].append(out["attr_vec"][L].astype(np.float32))
                scalar_rows.append(ra.features_to_array(out["features"]))
                y_list.append(int(fld["is_error"]))

            if not y_list:
                continue
            docs.append({
                "doc_id": doc_id,
                "y": np.array(y_list, dtype=np.int64),
                "answer": {L: np.stack(ans_rows[L], 0) for L in layers},
                "attr": {L: np.stack(attr_rows[L], 0) for L in layers},
                "scalars": np.stack(scalar_rows, 0).astype(np.float32),
            })
    if n_skip:
        logger.info("Skipped %d docs lacking per-token reasoning states.", n_skip)
    return docs


def build_features(doc: dict, layer: int, variant: str) -> np.ndarray:
    ans = doc["answer"][layer]
    if variant == "answer":
        return ans
    if variant == "scalars_only":
        return doc["scalars"]
    if variant == "fused_attr":
        return np.concatenate([ans, doc["attr"][layer]], axis=1)
    if variant == "fused_scalars":
        return np.concatenate([ans, doc["scalars"]], axis=1)
    if variant == "fused_both":
        return np.concatenate([ans, doc["attr"][layer], doc["scalars"]], axis=1)
    raise ValueError(f"Unknown variant: {variant}")


def _standardize(train: np.ndarray, test: np.ndarray):
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (train - mu) / sd, (test - mu) / sd


def _fit_fold(full_X, full_y, lo, hi, C):
    mask = np.ones(full_y.shape[0], dtype=bool)
    mask[lo:hi] = False
    y_tr = full_y[mask]
    if y_tr.sum() in (0, len(y_tr)):
        return None
    X_tr_s, X_te_s = _standardize(full_X[mask].astype(np.float64),
                                  full_X[lo:hi].astype(np.float64))
    y_te = full_y[lo:hi]
    clf = LogisticRegression(C=C, max_iter=_MAX_ITER, class_weight="balanced")
    clf.fit(X_tr_s, y_tr)
    proba = clf.predict_proba(X_te_s)[:, 1]
    fold_auroc = (float(roc_auc_score(y_te, proba))
                  if y_te.sum() not in (0, len(y_te)) else None)
    return y_te, proba, fold_auroc


def lodo_eval(docs, layer, variant, *, C=1.0, n_jobs=-1):
    """Returns metrics + a per-doc AUROC list aligned to `docs` (None where the
    fold was degenerate / single-class) so variants can be paired by document."""
    feats = [build_features(d, layer, variant).astype(np.float64) for d in docs]
    full_X = np.concatenate(feats, axis=0)
    full_y = np.concatenate([d["y"] for d in docs], axis=0)
    bounds = np.concatenate([[0], np.cumsum([len(d["y"]) for d in docs])])

    out = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_fit_fold)(full_X, full_y, int(bounds[i]), int(bounds[i + 1]), C)
        for i in range(len(docs))
    )

    per_doc = []
    fold_aurocs, oof_y, oof_p = [], [], []
    for r in out:
        if r is None:
            per_doc.append(None)
            continue
        y_te, proba, fa = r
        oof_y.append(y_te)
        oof_p.append(proba)
        per_doc.append(fa)
        if fa is not None:
            fold_aurocs.append(fa)

    res = {"layer": layer, "variant": variant, "n_valid_folds": len(fold_aurocs),
           "per_doc_auroc_mean": float(np.mean(fold_aurocs)) if fold_aurocs else None,
           "per_doc_auroc_std": float(np.std(fold_aurocs)) if fold_aurocs else None,
           "pooled_oof_auroc": None, "_per_doc": per_doc}
    if oof_y:
        ally, allp = np.concatenate(oof_y), np.concatenate(oof_p)
        if ally.sum() not in (0, len(ally)):
            res["pooled_oof_auroc"] = float(roc_auc_score(ally, allp))
    return res


def paired_test(answer_per_doc, variant_per_doc):
    """Paired Wilcoxon signed-rank on per-doc AUROCs (docs valid in both).
    Returns (mean_delta, p_value, n_pairs). p_value None if scipy unavailable
    or too few non-zero pairs."""
    pairs = [(a, b) for a, b in zip(answer_per_doc, variant_per_doc)
             if a is not None and b is not None]
    if len(pairs) < 5:
        return (None, None, len(pairs))
    a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
    mean_delta = float((b - a).mean())
    try:
        from scipy.stats import wilcoxon
        if np.allclose(a, b):
            return (mean_delta, 1.0, len(pairs))
        stat, p = wilcoxon(b, a, zero_method="wilcox", alternative="two-sided")
        return (mean_delta, float(p), len(pairs))
    except Exception as e:  # pragma: no cover
        logger.warning("Wilcoxon unavailable (%s); reporting mean delta only.", e)
        return (mean_delta, None, len(pairs))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_dir=cfg.logging.log_dir,
                  log_name="07_reasoning_attribution_lodo", log_to_file=cfg.logging.log_to_file)

    layers = args.layers or cfg.activations.layers
    activations_dir = cfg.artifacts_path / "activations"
    labels_dir = cfg.artifacts_path / "labels"
    extractions_dir = cfg.artifacts_path / "extractions"
    results_dir = cfg.artifacts_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    docs = load_attribution_docs(activations_dir, labels_dir, extractions_dir, layers)
    if len(docs) < 2:
        logger.error("Need >=2 docs with per-token reasoning states; found %d. "
                     "Did Stage 1 run with REASONING_TOKEN_LAYERS set?", len(docs))
        return 1

    n_fields = sum(len(d["y"]) for d in docs)
    n_err = sum(int(d["y"].sum()) for d in docs)
    # How often was a field's value actually found in the reasoning trace?
    mentioned = np.concatenate([d["scalars"][:, ra.FEATURE_NAMES.index("mentioned")] for d in docs])
    logger.info("Loaded %d docs, %d fields, %d errors (%.1f%%). Value mentioned in "
                "trace: %.1f%% of fields.", len(docs), n_fields, n_err,
                100 * n_err / max(n_fields, 1), 100 * float(mentioned.mean()))

    results = {v: {} for v in VARIANTS}
    for L in layers:
        for v in VARIANTS:
            logger.info("LODO: layer %d, variant %s ...", L, v)
            results[v][str(L)] = lodo_eval(docs, L, v, C=cfg.probe.C, n_jobs=args.jobs)

    # Significance vs answer, at each variant's best per_doc layer.
    def best_layer(variant, metric):
        rows = [(int(L), r[metric]) for L, r in results[variant].items()
                if r.get(metric) is not None]
        return max(rows, key=lambda x: x[1])[0] if rows else None

    summary = {}
    for metric in ("per_doc_auroc_mean", "pooled_oof_auroc"):
        logger.info("=" * 70)
        logger.info("ATTRIBUTION LODO metric: %s (best layer per variant)", metric)
        summary[metric] = {}
        for v in VARIANTS:
            bl = best_layer(v, metric)
            ba = results[v][str(bl)][metric] if bl is not None else None
            summary[metric][v] = {"best_layer": bl, "best_auroc": ba}
            logger.info("  %-14s best layer %s : %s", v, bl,
                        f"{ba:.4f}" if ba is not None else "n/a")

    # Paired significance (per_doc), each variant vs answer at the answer-best layer.
    ans_layer = best_layer("answer", "per_doc_auroc_mean")
    sig = {}
    if ans_layer is not None:
        ans_pd = results["answer"][str(ans_layer)]["_per_doc"]
        logger.info("=" * 70)
        logger.info("PAIRED SIGNIFICANCE vs answer (per-doc AUROC, layer %d):", ans_layer)
        for v in VARIANTS:
            if v == "answer":
                continue
            # Compare at the SAME layer as answer for a fair paired test.
            v_pd = results[v][str(ans_layer)]["_per_doc"]
            md, p, n = paired_test(ans_pd, v_pd)
            sig[v] = {"layer": ans_layer, "mean_delta": md, "p_value": p, "n_pairs": n}
            logger.info("  %-14s Δ=%+.4f  p=%s  (n=%d)", v,
                        md if md is not None else float("nan"),
                        f"{p:.4g}" if p is not None else "n/a", n)

    # Strip bulky per-doc lists before saving.
    for v in VARIANTS:
        for L in results[v]:
            results[v][L].pop("_per_doc", None)

    out = {"layers": layers, "n_docs": len(docs), "n_fields": n_fields,
           "n_errors": n_err, "pct_value_mentioned": float(mentioned.mean()),
           "per_layer": results, "summary": summary, "significance_vs_answer": sig}
    (results_dir / args.out_name).write_text(json.dumps(out, indent=2))
    logger.info("Saved -> %s", results_dir / args.out_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())

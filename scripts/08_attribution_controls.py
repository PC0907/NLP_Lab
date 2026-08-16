"""Stage 8 (reasoning-trace paper): CONTROLS for the field-localized attribution gain.

Stage 7 showed `answer + field-localized reasoning vector` beats `answer` alone
under LODO (paired Wilcoxon p=0.044 at layer 19 on the 300-doc SOB run). The
first question any reviewer asks is: *is that gain actually about localization,
or would ANY extra 3584 dimensions have done the same?* This stage answers it
with three controls that hold everything constant except the thing we claim
matters.

Every control keeps the answer-token block, the layer, and the feature
dimensionality IDENTICAL to `fused_attr`. Only the content of the reasoning
block changes:

  fused_attr        the claim: attr_vec is the pooled hidden states of the
                    reasoning tokens where THIS field's value is mentioned.

  ctrl_docmean      attr_vec replaced by the DOCUMENT MEAN of its fields'
                    attr_vecs -> same tokens, same layer, same dimensionality,
                    but constant within a document. This is Stage 6's doc-level
                    pooling rebuilt from the attribution tokens themselves, so
                    it is the apples-to-apples version of the null. If the
                    localization story is right, this should land at ~answer.

  ctrl_shuffled     attr_vec permuted across the fields WITHIN each document ->
                    the exact same set of vectors is present in the document,
                    just attached to the wrong fields. Kills the field<->reason
                    correspondence and nothing else. Averaged over --shuffle-reps
                    seeds. This is the strongest control: if fused_attr beats
                    ctrl_shuffled, the gain is specifically about attributing
                    the right reasoning to the right field.

  ctrl_random       attr_vec replaced by Gaussian noise matched to the attr
                    block's per-dimension scale -> pure dimensionality control
                    (does the probe just benefit from a wider feature space?).

Reported per control: paired Wilcoxon on per-document AUROCs, mean delta, a
document-level bootstrap CI on that delta, and a Holm-Bonferroni-corrected
p-value over the whole family of tests (the multiple-comparison caveat Update 07
flagged as open).

Usage:
    python scripts/08_attribution_controls.py --config CFG --layers 19 --jobs -1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np

from probe_extraction.config import load_config
from probe_extraction.utils.logging import setup_logging

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]


def _load_by_path(name: str, rel: str):
    """Import a module by file path (Stage 7 starts with a digit, and the
    attribution core must bypass the extraction package's torch import)."""
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ra = _load_by_path("reasoning_attribution",
                   "src/probe_extraction/extraction/reasoning_attribution.py")
s7 = _load_by_path("stage07", "scripts/07_reasoning_attribution_lodo.py")

# The claim + its controls. `answer` is the baseline all deltas are measured from.
VARIANTS = ("answer", "fused_attr", "ctrl_docmean", "ctrl_shuffled", "ctrl_random")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Controls for field-localized attribution.")
    p.add_argument("--config", required=True)
    p.add_argument("--layers", type=int, nargs="*", default=[19],
                   help="Layers to run controls at (default: 19, the peak layer).")
    p.add_argument("--jobs", type=int, default=-1)
    p.add_argument("--shuffle-reps", type=int, default=3,
                   help="Seeds to average the within-document shuffle control over.")
    p.add_argument("--bootstrap", type=int, default=2000,
                   help="Document-level bootstrap resamples for the delta CI (0 = off).")
    p.add_argument("--out-name", type=str, default="attribution_controls.json")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Control feature construction
# ---------------------------------------------------------------------------

def build_control_features(doc: dict, layer: int, variant: str,
                           rng: np.random.Generator) -> np.ndarray:
    """Answer block + a reasoning block whose CONTENT depends on the variant.
    Shape is identical across fused_* / ctrl_* so dimensionality is controlled."""
    ans = doc["answer"][layer]
    if variant == "answer":
        return ans

    attr = doc["attr"][layer]
    if variant == "fused_attr":
        block = attr
    elif variant == "ctrl_docmean":
        # Same vectors, collapsed to one per document -> constant within doc.
        block = np.repeat(attr.mean(axis=0, keepdims=True), attr.shape[0], axis=0)
    elif variant == "ctrl_shuffled":
        # Same vectors, wrong fields. A 1-field doc cannot be shuffled; it
        # contributes an identical row, which is correct (nothing to break).
        block = attr[rng.permutation(attr.shape[0])]
    elif variant == "ctrl_random":
        # Match the attr block's per-dimension scale so the control is not
        # trivially discarded by standardization.
        sd = attr.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)
        block = rng.standard_normal(attr.shape).astype(np.float32) * sd
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return np.concatenate([ans, block], axis=1)


def lodo_eval_variant(docs, layer, variant, *, C=1.0, n_jobs=-1, seed=0):
    """LODO over a control variant. Mirrors Stage 7's lodo_eval but builds
    features through build_control_features so the controls stay in one place."""
    rng = np.random.default_rng(seed)
    feats = [build_control_features(d, layer, variant, rng).astype(np.float64)
             for d in docs]
    full_X = np.concatenate(feats, axis=0)
    full_y = np.concatenate([d["y"] for d in docs], axis=0)
    bounds = np.concatenate([[0], np.cumsum([len(d["y"]) for d in docs])])

    from joblib import Parallel, delayed
    out = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(s7._fit_fold)(full_X, full_y, int(bounds[i]), int(bounds[i + 1]), C)
        for i in range(len(docs))
    )

    per_doc, fold_aurocs, oof_y, oof_p = [], [], [], []
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

    from sklearn.metrics import roc_auc_score
    pooled = None
    if oof_y:
        ally, allp = np.concatenate(oof_y), np.concatenate(oof_p)
        if ally.sum() not in (0, len(ally)):
            pooled = float(roc_auc_score(ally, allp))

    return {"layer": layer, "variant": variant, "n_valid_folds": len(fold_aurocs),
            "per_doc_auroc_mean": float(np.mean(fold_aurocs)) if fold_aurocs else None,
            "per_doc_auroc_std": float(np.std(fold_aurocs)) if fold_aurocs else None,
            "pooled_oof_auroc": pooled, "_per_doc": per_doc}


def average_per_doc(runs: list[list]) -> list:
    """Average aligned per-doc AUROC lists across shuffle seeds. A document is
    kept only if every seed produced a valid fold for it (so the paired test
    against `answer` stays on a clean, common document set)."""
    n = len(runs[0])
    out = []
    for i in range(n):
        vals = [r[i] for r in runs]
        out.append(float(np.mean(vals)) if all(v is not None for v in vals) else None)
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def bootstrap_delta_ci(a_per_doc, b_per_doc, n_boot: int, seed: int = 0):
    """Percentile CI for the mean paired delta, resampling DOCUMENTS (the unit
    of independence) with replacement."""
    pairs = [(a, b) for a, b in zip(a_per_doc, b_per_doc)
             if a is not None and b is not None]
    if n_boot <= 0 or len(pairs) < 5:
        return None
    d = np.array([b - a for a, b in pairs])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return {"mean": float(d.mean()),
            "ci_low": float(np.percentile(means, 2.5)),
            "ci_high": float(np.percentile(means, 97.5)),
            "frac_boot_positive": float((means > 0).mean()),
            "n_pairs": len(pairs)}


def mention_analysis(docs) -> dict:
    """The interpretable, model-agnostic headline: is a field whose value never
    appears in the reasoning trace more likely to be WRONG?

    This is the cheapest trust signal in the whole project -- it needs no probe,
    no hidden states, just string matching against the trace -- so its effect
    size is worth reporting on its own, whatever the probe does.
    """
    y = np.concatenate([d["y"] for d in docs])
    mentioned = np.concatenate(
        [d["scalars"][:, ra.FEATURE_NAMES.index("mentioned")] for d in docs]).astype(bool)
    full = np.concatenate(
        [d["scalars"][:, ra.FEATURE_NAMES.index("match_full")] for d in docs]).astype(bool)

    def rate(mask):
        return {"n": int(mask.sum()),
                "n_errors": int(y[mask].sum()),
                "error_rate": float(y[mask].mean()) if mask.any() else None}

    groups = {
        "mentioned_full": rate(mentioned & full),
        "mentioned_partial": rate(mentioned & ~full),
        "not_mentioned": rate(~mentioned),
        "mentioned_any": rate(mentioned),
        "all": rate(np.ones_like(y, dtype=bool)),
    }

    er_yes = groups["mentioned_any"]["error_rate"]
    er_no = groups["not_mentioned"]["error_rate"]
    out = {"groups": groups, "risk_ratio": None, "p_value": None, "test": None}
    if er_yes and er_no is not None and groups["not_mentioned"]["n"] > 0:
        out["risk_ratio"] = (er_no / er_yes) if er_yes > 0 else None
        table = [[int(y[~mentioned].sum()), int((~y.astype(bool))[~mentioned].sum())],
                 [int(y[mentioned].sum()), int((~y.astype(bool))[mentioned].sum())]]
        out["contingency_not_mentioned_vs_mentioned"] = table
        try:
            from scipy.stats import fisher_exact
            _, p = fisher_exact(table)
            out["p_value"], out["test"] = float(p), "fisher_exact"
        except Exception as e:  # pragma: no cover
            logger.warning("Fisher exact unavailable (%s).", e)
    return out


def holm_bonferroni(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down correction over a family of tests. Returns adjusted
    p-values (monotone, capped at 1.0). Entries with p=None are skipped."""
    items = [(k, v) for k, v in pvals.items() if v is not None]
    if not items:
        return {}
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        val = min(1.0, (m - i) * p)
        running = max(running, val)  # enforce monotonicity
        adj[k] = running
    return adj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_dir=cfg.logging.log_dir,
                  log_name="08_attribution_controls", log_to_file=cfg.logging.log_to_file)

    layers = args.layers
    activations_dir = cfg.artifacts_path / "activations"
    labels_dir = cfg.artifacts_path / "labels"
    extractions_dir = cfg.artifacts_path / "extractions"
    results_dir = cfg.artifacts_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    docs = s7.load_attribution_docs(activations_dir, labels_dir, extractions_dir, layers)
    if len(docs) < 2:
        logger.error("Need >=2 docs with per-token reasoning states; found %d.", len(docs))
        return 1

    n_fields = sum(len(d["y"]) for d in docs)
    n_err = sum(int(d["y"].sum()) for d in docs)
    n_multi = sum(1 for d in docs if len(d["y"]) > 1)
    logger.info("Loaded %d docs (%d with >1 field), %d fields, %d errors (%.1f%%).",
                len(docs), n_multi, n_fields, n_err, 100 * n_err / max(n_fields, 1))
    logger.info("Controls at layers %s | shuffle reps %d | bootstrap %d",
                layers, args.shuffle_reps, args.bootstrap)

    # Interpretable, probe-free signal first -- it stands on its own.
    mention = mention_analysis(docs)
    logger.info("-" * 70)
    logger.info("MENTION ANALYSIS (is an unmentioned value more often wrong?):")
    for name, g in mention["groups"].items():
        logger.info("  %-18s n=%5d  errors=%5d  error rate %s", name, g["n"], g["n_errors"],
                    f"{g['error_rate']:.1%}" if g["error_rate"] is not None else "n/a")
    logger.info("  risk ratio (not mentioned / mentioned): %s   p=%s",
                f"{mention['risk_ratio']:.2f}" if mention["risk_ratio"] else "n/a",
                f"{mention['p_value']:.4g}" if mention["p_value"] is not None else "n/a")

    all_results, all_sig = {}, {}
    for L in layers:
        logger.info("=" * 70)
        logger.info("LAYER %d", L)
        res = {}
        for v in VARIANTS:
            if v == "ctrl_shuffled":
                logger.info("  LODO: %s (%d seeds) ...", v, args.shuffle_reps)
                runs = [lodo_eval_variant(docs, L, v, C=cfg.probe.C,
                                          n_jobs=args.jobs, seed=1000 + s)
                        for s in range(args.shuffle_reps)]
                merged = dict(runs[0])
                merged["_per_doc"] = average_per_doc([r["_per_doc"] for r in runs])
                valid = [x for x in merged["_per_doc"] if x is not None]
                merged["per_doc_auroc_mean"] = float(np.mean(valid)) if valid else None
                merged["per_doc_auroc_std"] = float(np.std(valid)) if valid else None
                merged["pooled_oof_auroc"] = float(np.mean(
                    [r["pooled_oof_auroc"] for r in runs
                     if r["pooled_oof_auroc"] is not None])) if any(
                    r["pooled_oof_auroc"] is not None for r in runs) else None
                merged["n_seeds"] = args.shuffle_reps
                res[v] = merged
            else:
                logger.info("  LODO: %s ...", v)
                res[v] = lodo_eval_variant(docs, L, v, C=cfg.probe.C,
                                           n_jobs=args.jobs, seed=7)
            r = res[v]
            logger.info("    per-doc %.4f | pooled %s", r["per_doc_auroc_mean"],
                        f"{r['pooled_oof_auroc']:.4f}"
                        if r["pooled_oof_auroc"] is not None else "n/a")

        # --- The comparisons that carry the argument ---------------------
        # 1. fused_attr vs answer          -> does attribution help at all?
        # 2. fused_attr vs each control    -> is it the LOCALIZATION that helps?
        comparisons = {
            "fused_attr_vs_answer": ("answer", "fused_attr"),
            "fused_attr_vs_docmean": ("ctrl_docmean", "fused_attr"),
            "fused_attr_vs_shuffled": ("ctrl_shuffled", "fused_attr"),
            "fused_attr_vs_random": ("ctrl_random", "fused_attr"),
            # Controls vs answer: each of these SHOULD be ~null. Reporting them
            # keeps us honest -- a control that also beats answer would mean the
            # gain is not about localization after all.
            "docmean_vs_answer": ("answer", "ctrl_docmean"),
            "shuffled_vs_answer": ("answer", "ctrl_shuffled"),
            "random_vs_answer": ("answer", "ctrl_random"),
        }
        sig = {}
        for name, (base, test) in comparisons.items():
            md, p, n = s7.paired_test(res[base]["_per_doc"], res[test]["_per_doc"])
            sig[name] = {"baseline": base, "variant": test, "mean_delta": md,
                         "p_value": p, "n_pairs": n,
                         "bootstrap": bootstrap_delta_ci(res[base]["_per_doc"],
                                                         res[test]["_per_doc"],
                                                         args.bootstrap)}

        adj = holm_bonferroni({k: v["p_value"] for k, v in sig.items()})
        for k in sig:
            sig[k]["p_holm"] = adj.get(k)

        logger.info("-" * 70)
        logger.info("PAIRED TESTS (per-doc AUROC, layer %d, Holm-corrected over %d tests):",
                    L, len(sig))
        for k, s in sig.items():
            b = s["bootstrap"]
            ci = (f"[{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]" if b else "n/a")
            logger.info("  %-26s Δ=%+.4f  95%% CI %s  p=%s  p_holm=%s  (n=%d)",
                        k, s["mean_delta"] if s["mean_delta"] is not None else float("nan"),
                        ci,
                        f"{s['p_value']:.4g}" if s["p_value"] is not None else "n/a",
                        f"{s['p_holm']:.4g}" if s.get("p_holm") is not None else "n/a",
                        s["n_pairs"])

        for v in res:
            res[v].pop("_per_doc", None)
        all_results[str(L)] = res
        all_sig[str(L)] = sig

    out = {"layers": layers, "n_docs": len(docs), "n_docs_multifield": n_multi,
           "n_fields": n_fields, "n_errors": n_err,
           "shuffle_reps": args.shuffle_reps, "bootstrap": args.bootstrap,
           "mention_analysis": mention,
           "per_layer": all_results, "significance": all_sig}
    (results_dir / args.out_name).write_text(json.dumps(out, indent=2))
    logger.info("Saved -> %s", results_dir / args.out_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Stage 9 (reasoning-trace paper): what the trust signal is WORTH.

Stages 3-8 answer "can we detect wrong fields?" in AUROC. AUROC is not the
project's deliverable -- the deliverable is the cost-quality tradeoff of
SELECTIVE REGENERATION: spend a regeneration budget on the fields most likely to
be wrong, and end up with a cleaner record than blanket regeneration at a
fraction of the cost. This stage turns the LODO out-of-fold probe scores into
that curve, for the probe and for the free log-prob baselines it has to beat.

Scored signals (all evaluated on the SAME fields, same LODO folds):
  probe_answer     answer-token probe (the partner branch's signal)
  probe_fused      answer + field-localized reasoning attribution (this track)
  mean_logprob     the model's own average confidence over the field's tokens
  min_logprob      the model's least-confident token in the field
  random           a shuffled score (the floor)
  oracle           perfect ranking (the ceiling)

Two budget regimes, because they answer different questions:
  global   rank ALL fields in the corpus and regenerate the top b -- the
           deployment view when you have one budget for a batch of documents.
  per_doc  within each document regenerate its top b fraction -- the view that
           matches our per-document AUROC gain, and the realistic one when
           documents are processed one at a time.

Outputs, at each budget b:
  errors_caught      recall of true errors inside the flagged set
  precision          share of flagged fields that really were wrong
  selective_risk     error rate among the fields we DIDN'T flag (the residual
                     risk a downstream consumer inherits) -- summarized as AURC
  final_error_rate   simulated post-regeneration error rate under a repair model

The repair model is explicit, and stated as an assumption rather than hidden:
a regenerated field that was WRONG becomes right with probability `repair_rate`;
a regenerated field that was RIGHT is broken with probability `damage_rate`.
`repair_rate=1, damage_rate=0` is the parameter-free upper bound (it reduces to
errors_caught), and we sweep the rest. We also report the BREAK-EVEN repair
rate: how good a regenerator has to be for spending the budget to be worth it at
all. That number is what makes the AUROC gain actionable.

Requires a completed Stage 2 (labels) and a Stage 1 run with
REASONING_TOKEN_LAYERS set.

Usage:
    python scripts/09_selective_regeneration_sob.py --config CFG --layer 19 --jobs -1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from probe_extraction.baselines.token_logprob import compute_token_logprob_scores
from probe_extraction.config import load_config
from probe_extraction.utils.logging import setup_logging

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]


def _load_by_path(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s7 = _load_by_path("stage07", "scripts/07_reasoning_attribution_lodo.py")

# Includes 0.0 so the risk-coverage curve spans the full coverage range and AURC
# is integrated over [0, 1] rather than [0, 0.95].
DEFAULT_BUDGETS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00]

# np.trapz was removed in NumPy 2.0; the cluster venv and this laptop are on
# different sides of that line.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Selective-regeneration cost-quality curves.")
    p.add_argument("--config", required=True)
    p.add_argument("--layer", type=int, default=19,
                   help="Probe layer to score with (default 19, the peak layer).")
    p.add_argument("--fused-variant", type=str, default="fused_decomposed",
                   choices=["fused_attr", "fused_both", "fused_decomposed"],
                   help="Which attribution probe to carry forward as `probe_fused`. "
                        "Defaults to fused_decomposed, the strongest variant on "
                        "the 974-doc corpus (+0.0364 over answer-only).")
    p.add_argument("--jobs", type=int, default=-1)
    p.add_argument("--budgets", type=float, nargs="*", default=DEFAULT_BUDGETS)
    p.add_argument("--repair-rate", type=float, nargs="*", default=[0.5, 0.7, 1.0],
                   help="P(a regenerated WRONG field becomes right).")
    p.add_argument("--damage-rate", type=float, default=0.05,
                   help="P(a regenerated RIGHT field gets broken).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-name", type=str, default="selective_regeneration.json")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def lodo_oof_scores(docs, layer, variant, *, C=1.0, n_jobs=-1):
    """LODO out-of-fold P(error) per field, returned as a list aligned to
    `docs` (one array per document, None where the fold was degenerate)."""
    from joblib import Parallel, delayed

    feats = [s7.build_features(d, layer, variant).astype(np.float64) for d in docs]
    full_X = np.concatenate(feats, axis=0)
    full_y = np.concatenate([d["y"] for d in docs], axis=0)
    bounds = np.concatenate([[0], np.cumsum([len(d["y"]) for d in docs])])

    out = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(s7._fit_fold)(full_X, full_y, int(bounds[i]), int(bounds[i + 1]), C)
        for i in range(len(docs))
    )
    return [None if r is None else r[1] for r in out]


def logprob_scores(docs, extractions_dir: Path):
    """Per-field token-logprob baselines, joined to the LODO rows by path_str.

    Log-probs are negative and HIGHER means more confident, so we negate to get
    a score where higher means "more likely to be an error" -- the same
    direction as the probe's P(error). Fields whose span is missing get NaN and
    are dropped from the common evaluation set.
    """
    mean_out, min_out = [], []
    for d in docs:
        ext_path = extractions_dir / f"{d['doc_id']}.json"
        spans, lps = {}, []
        if ext_path.exists():
            ext = json.load(ext_path.open())
            lps = ext.get("token_logprobs") or []
            spans = {f["path_str"]: tuple(f["token_span"]) for f in ext.get("fields", [])}
        m_row, n_row = [], []
        for ps in d["path_strs"]:
            if not lps or ps not in spans:
                m_row.append(np.nan)
                n_row.append(np.nan)
                continue
            s = compute_token_logprob_scores(token_logprobs=lps, token_span=spans[ps])
            m_row.append(-s["mean_logprob"])
            n_row.append(-s["min_logprob"])
        mean_out.append(np.array(m_row, dtype=np.float64))
        min_out.append(np.array(n_row, dtype=np.float64))
    return mean_out, min_out


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------

def _flag_global(scores: np.ndarray, budget: float, rng) -> np.ndarray:
    """Top-b fraction of ALL fields. Ties are broken randomly rather than by
    array order, so a constant/degenerate score scores as chance, not as a
    lucky artifact of document ordering."""
    n = len(scores)
    k = int(round(budget * n))
    flagged = np.zeros(n, dtype=bool)
    if k <= 0:
        return flagged
    order = np.lexsort((rng.random(n), -scores))
    flagged[order[:k]] = True
    return flagged


def _flag_per_doc(scores: np.ndarray, doc_ids: np.ndarray, budget: float,
                  rng) -> np.ndarray:
    """Within each document, flag its top-b fraction (at least one field once
    the budget is non-zero -- you cannot regenerate a fraction of a field)."""
    flagged = np.zeros(len(scores), dtype=bool)
    for d in np.unique(doc_ids):
        idx = np.flatnonzero(doc_ids == d)
        k = int(np.ceil(budget * len(idx))) if budget > 0 else 0
        k = min(k, len(idx))
        if k <= 0:
            continue
        sub = scores[idx]
        order = np.lexsort((rng.random(len(sub)), -sub))
        flagged[idx[order[:k]]] = True
    return flagged


def curve(scores, y, doc_ids, budgets, regime, *, repair_rates, damage_rate, rng):
    n = len(y)
    total_err = int(y.sum())
    rows = []
    for b in budgets:
        flagged = (_flag_global(scores, b, rng) if regime == "global"
                   else _flag_per_doc(scores, doc_ids, b, rng))
        k = int(flagged.sum())
        caught = int(y[flagged].sum())
        retained = ~flagged
        n_ret = int(retained.sum())
        row = {
            "budget": b,
            "n_flagged": k,
            "actual_frac_flagged": k / n,
            "errors_caught": caught,
            "recall": caught / total_err if total_err else None,
            "precision": caught / k if k else None,
            "selective_risk": float(y[retained].mean()) if n_ret else 0.0,
            "final_error_rate": {},
            "break_even_repair_rate": None,
        }
        # Simulated post-regeneration error rate (expected value; the repair
        # model is stated in the module docstring).
        for pi in repair_rates:
            residual = (total_err - caught) + caught * (1 - pi) + (k - caught) * damage_rate
            row["final_error_rate"][f"{pi:g}"] = residual / n
        # How good must the regenerator be for this spend to break even?
        # caught*pi >= (k - caught)*damage_rate
        if caught > 0:
            row["break_even_repair_rate"] = min(1.0, (k - caught) * damage_rate / caught)
        rows.append(row)

    # AURC over the risk-coverage curve (coverage = 1 - budget); lower is better.
    cov = np.array([1.0 - r["actual_frac_flagged"] for r in rows])
    risk = np.array([r["selective_risk"] for r in rows])
    order = np.argsort(cov)
    aurc = float(_trapezoid(risk[order], cov[order])) if len(order) > 1 else None
    return {"rows": rows, "aurc": aurc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_dir=cfg.logging.log_dir,
                  log_name="09_selective_regeneration", log_to_file=cfg.logging.log_to_file)

    L = args.layer
    activations_dir = cfg.artifacts_path / "activations"
    labels_dir = cfg.artifacts_path / "labels"
    extractions_dir = cfg.artifacts_path / "extractions"
    results_dir = cfg.artifacts_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    docs = s7.load_attribution_docs(activations_dir, labels_dir, extractions_dir, [L])
    if len(docs) < 2:
        logger.error("Need >=2 docs with per-token reasoning states; found %d.", len(docs))
        return 1

    logger.info("Scoring %d docs at layer %d (fused variant: %s) ...",
                len(docs), L, args.fused_variant)
    oof = {
        "probe_answer": lodo_oof_scores(docs, L, "answer", C=cfg.probe.C, n_jobs=args.jobs),
        "probe_fused": lodo_oof_scores(docs, L, args.fused_variant,
                                       C=cfg.probe.C, n_jobs=args.jobs),
    }
    lp_mean, lp_min = logprob_scores(docs, extractions_dir)

    # Build the COMMON evaluation set: keep only fields where every signal is
    # defined, so the comparison is strictly like-for-like.
    rng = np.random.default_rng(args.seed)
    y_parts, doc_parts, sig_parts = [], [], {k: [] for k in oof}
    lpm_parts, lpn_parts = [], []
    n_drop = 0
    for i, d in enumerate(docs):
        if any(oof[k][i] is None for k in oof):
            n_drop += len(d["y"])
            continue
        ok = np.isfinite(lp_mean[i]) & np.isfinite(lp_min[i])
        n_drop += int((~ok).sum())
        if not ok.any():
            continue
        y_parts.append(d["y"][ok])
        doc_parts.append(np.full(int(ok.sum()), i))
        for k in oof:
            sig_parts[k].append(oof[k][i][ok])
        lpm_parts.append(lp_mean[i][ok])
        lpn_parts.append(lp_min[i][ok])

    y = np.concatenate(y_parts)
    doc_ids = np.concatenate(doc_parts)
    signals = {k: np.concatenate(v) for k, v in sig_parts.items()}
    signals["mean_logprob"] = np.concatenate(lpm_parts)
    signals["min_logprob"] = np.concatenate(lpn_parts)
    signals["random"] = rng.random(len(y))
    signals["oracle"] = y.astype(np.float64) + rng.random(len(y)) * 1e-6

    n_docs_eval = len(np.unique(doc_ids))
    logger.info("Evaluation set: %d fields in %d docs, %d errors (%.1f%%). "
                "Dropped %d fields lacking a signal.",
                len(y), n_docs_eval, int(y.sum()), 100 * y.mean(), n_drop)

    aurocs = {k: (float(roc_auc_score(y, v)) if y.sum() not in (0, len(y)) else None)
              for k, v in signals.items()}
    logger.info("-" * 70)
    logger.info("Pooled AUROC on the common set:")
    for k, v in aurocs.items():
        logger.info("  %-14s %s", k, f"{v:.4f}" if v is not None else "n/a")

    curves = {}
    for regime in ("global", "per_doc"):
        curves[regime] = {}
        logger.info("=" * 70)
        logger.info("SELECTIVE REGENERATION -- %s budget", regime)
        for k, v in signals.items():
            curves[regime][k] = curve(
                v, y, doc_ids, args.budgets, regime,
                repair_rates=args.repair_rate, damage_rate=args.damage_rate,
                rng=np.random.default_rng(args.seed))
        # Headline table: errors caught per budget, signal by signal.
        logger.info("  errors caught (recall) by regeneration budget:")
        header = "    %-14s" % "signal" + "".join(f"{b:>8.0%}" for b in args.budgets)
        logger.info(header)
        for k in signals:
            cells = "".join(
                f"{r['recall']:>8.1%}" if r["recall"] is not None else f"{'n/a':>8}"
                for r in curves[regime][k]["rows"])
            logger.info("    %-14s%s", k, cells)
        logger.info("  AURC (lower is better): %s",
                    "  ".join(f"{k}={curves[regime][k]['aurc']:.4f}"
                              for k in signals
                              if curves[regime][k]["aurc"] is not None))

    # The one number for the abstract: gain over the best free baseline at a
    # realistic 20% budget, in the per-document regime where our AUROC gain lives.
    def at_budget(regime, signal, b=0.20):
        for r in curves[regime][signal]["rows"]:
            if abs(r["budget"] - b) < 1e-9:
                return r
        return None

    headline = {}
    for regime in ("global", "per_doc"):
        r_fused = at_budget(regime, "probe_fused")
        r_ans = at_budget(regime, "probe_answer")
        best_base = max(("mean_logprob", "min_logprob"),
                        key=lambda s: (at_budget(regime, s) or {}).get("recall") or 0.0)
        r_base = at_budget(regime, best_base)
        if r_fused and r_base:
            headline[regime] = {
                "budget": 0.20,
                "probe_fused_recall": r_fused["recall"],
                "probe_answer_recall": r_ans["recall"] if r_ans else None,
                "best_logprob_baseline": best_base,
                "baseline_recall": r_base["recall"],
                "gain_vs_baseline": (r_fused["recall"] or 0) - (r_base["recall"] or 0),
                "gain_vs_answer": ((r_fused["recall"] or 0) - (r_ans["recall"] or 0)
                                   if r_ans else None),
                "break_even_repair_rate": r_fused["break_even_repair_rate"],
            }
    logger.info("=" * 70)
    logger.info("HEADLINE (%.0f%% regeneration budget):", 20)
    for regime, h in headline.items():
        logger.info("  %-8s probe_fused catches %.1f%% of errors vs %.1f%% for %s "
                    "(+%.1f pts); answer-only %.1f%%; break-even repair rate %.2f",
                    regime, 100 * h["probe_fused_recall"], 100 * h["baseline_recall"],
                    h["best_logprob_baseline"], 100 * h["gain_vs_baseline"],
                    100 * (h["probe_answer_recall"] or 0),
                    h["break_even_repair_rate"] or 0.0)

    out = {
        "layer": L, "fused_variant": args.fused_variant,
        "n_fields": int(len(y)), "n_docs": int(n_docs_eval),
        "n_errors": int(y.sum()), "error_rate": float(y.mean()),
        "n_fields_dropped": int(n_drop),
        "repair_rates": args.repair_rate, "damage_rate": args.damage_rate,
        "budgets": args.budgets,
        "pooled_auroc": aurocs, "curves": curves, "headline": headline,
    }
    (results_dir / args.out_name).write_text(json.dumps(out, indent=2))
    logger.info("Saved -> %s", results_dir / args.out_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())

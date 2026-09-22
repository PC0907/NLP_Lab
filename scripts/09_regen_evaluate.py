"""Stage 9: what real selective regeneration actually BUYS (CPU).

Stage 7 produced the cost-quality curve under an assumption: a re-asked wrong
field becomes right with probability 0.7, a re-asked right field breaks with
probability 0.05. Those two numbers were invented. Everything downstream of them
-- the post-regeneration error rate, the break-even analysis -- inherited that
invention.

This stage removes it. Stage 8 actually re-asked the model. Here we spend a
budget the way the paper says we would, swap in what the model actually said,
re-label the result against gold with the SAME matcher Stage 2 used, and count
what really happened.

The accounting, per flagged field:
    repaired     it was wrong, and after the swap it is right
    damaged      it was right, and after the swap it is wrong
    unchanged    the label did not move (this includes the case where the
                 resample returned the identical value -- reported separately,
                 because "temperature changed nothing" is a different story from
                 "the model changed its answer and was still wrong")
    unavailable  the resample had no value at that path, so nothing was swapped
    lost         the swap changed the record's shape and that leaf no longer
                 exists to be labeled

HOW THE BUDGET CURVE IS COMPUTED. Naively, every (signal, regime, budget) would
need its own joint re-labeling of every document -- hundreds of thousands of
matcher runs. Instead we measure each field's swap ONCE, in isolation, recording
its net effect on that document's error count over the scored field set
(`net_delta`, which already includes collateral damage to neighbouring fields).
Any budget is then the sum of the net deltas of the fields it flags.

That is an ADDITIVITY assumption -- swapping two fields at once might not equal
swapping each alone -- so it is not taken on faith: `--checkpoint-budgets` re-does
selected budgets jointly (all flagged fields swapped together, one re-labeling)
and reports the discrepancy. Budget 1.0 is always checked, since regenerating
everything is the most interaction-heavy case there is.

THREE REGENERATION STRATEGIES, all free from the same Stage 8 run:
    first        the first usable resample -- the honest k=1 deployment cost
    vote         plurality value across the usable resamples (self-consistency)
    vote_strict  the same, but only where the resamples AGREE (>=2 and a strict
                 majority); otherwise the field is left alone. A resample that
                 disagrees with itself is a poor reason to overwrite an answer
                 that may well have been right.

CONTROLS reported alongside:
    * the full resample's own error rate, with no probe involved -- if plain
      resampling were already better, selective anything would be pointless
    * budget 1.0 = regenerate everything, the blanket-regeneration baseline
    * random and log-prob flagging at every matched budget

Requires Stage 7 (for artifacts/results/oof_field_scores.json) and Stage 8.

Usage:
    python scripts/09_regen_evaluate.py --config CFG --jobs -1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# Labeling walks deeply nested records; Stage 2 raises the limit for the same
# reason. Set it here too rather than relying on an import side-effect.
sys.setrecursionlimit(20000)

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from probe_extraction.config import load_config  # noqa: E402
from probe_extraction.regen import (  # noqa: E402
    errors_over_scored,
    make_labeler,
    measure_document,
    usable_samples,
)
from probe_extraction.utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]


def _load_by_path(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Stage 7 owns the flagging rules; reusing them (rather than reimplementing)
# is what makes the measured curve comparable to the simulated one.
selective = _load_by_path("stage07", "scripts/07_selective_regeneration_sob.py")
# Stage 6 owns the multiple-comparison correction, so the regeneration results
# are corrected exactly the way the AUROC results were.
controls = _load_by_path("stage06", "scripts/06_attribution_controls.py")


def _stage02():
    """Stage 2 owns the matcher settings and the benchmark loader; taking them
    from there means the before and after labels cannot be produced under
    different rules.

    Imported lazily because Stage 2 pulls in Stage 1, which imports the model
    stack. This stage is CPU-only analysis and its pure functions should be
    importable (and testable) without torch installed.
    """
    return _load_by_path("stage02", "scripts/02_label.py")

STRATEGIES = ("first", "vote", "vote_strict")
SIGNALS = ("probe_fused", "probe_answer", "min_logprob", "mean_logprob",
           "random", "oracle")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure real selective regeneration.")
    p.add_argument("--config", required=True)
    p.add_argument("--scores-name", default="oof_field_scores.json",
                   help="Stage 7's per-field OOF score dump, under results/.")
    p.add_argument("--strategies", nargs="*", default=list(STRATEGIES),
                   choices=list(STRATEGIES))
    p.add_argument("--signals", nargs="*", default=list(SIGNALS),
                   choices=list(SIGNALS))
    p.add_argument("--budgets", type=float, nargs="*", default=selective.DEFAULT_BUDGETS)
    p.add_argument("--checkpoint-budgets", type=float, nargs="*",
                   default=[0.10, 0.20, 1.00],
                   help="Budgets to re-verify with a JOINT re-labeling, testing "
                        "the additivity the fast path assumes.")
    p.add_argument("--checkpoint-signal", default="probe_fused",
                   help="Signal whose flags the joint check uses.")
    p.add_argument("--allow-partial", action="store_true",
                   help="Proceed even if some scored documents have no Stage 8 "
                        "regeneration. Off by default: a partial run would "
                        "silently report a number computed on a subset.")
    p.add_argument("--limit-docs", type=int, default=None,
                   help="Cap on documents (debug aid).")
    p.add_argument("--bootstrap", type=int, default=2000,
                   help="Document-level bootstrap replicates for the confidence "
                        "intervals (0 disables).")
    p.add_argument("--significance-budgets", type=float, nargs="*",
                   default=[0.10, 0.20],
                   help="Budgets at which to test whether the improvement is real.")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-name", default="regen_evaluation.json")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Curves over the measured outcomes
# ---------------------------------------------------------------------------

def measured_curve(scores: np.ndarray, y: np.ndarray, doc_ids: np.ndarray,
                   net_delta: np.ndarray, self_out: np.ndarray,
                   budgets: Sequence[float], regime: str, *, rng) -> dict:
    """Risk-coverage plus the MEASURED post-regeneration error rate."""
    n = len(y)
    base_errors = int(y.sum())
    rows = []
    for b in budgets:
        flagged = (selective._flag_global(scores, b, rng) if regime == "global"
                   else selective._flag_per_doc(scores, doc_ids, b, rng))
        k = int(flagged.sum())
        caught = int(y[flagged].sum())
        retained = ~flagged
        counts = {o: int((self_out[flagged] == o).sum())
                  for o in ("repaired", "damaged", "unchanged", "unavailable", "lost")}
        final_errors = base_errors + int(net_delta[flagged].sum())
        # Denominators for the measured rates: only fields where a regenerated
        # value actually existed can be repaired or damaged.
        avail = flagged & (self_out != "unavailable")
        n_avail_err = int(y[avail].sum())
        n_avail_ok = int((~y[avail].astype(bool)).sum())
        rows.append({
            "budget": b,
            "n_flagged": k,
            "actual_frac_flagged": k / n if n else 0.0,
            "errors_caught": caught,
            "recall": caught / base_errors if base_errors else None,
            "precision": caught / k if k else None,
            "selective_risk": float(y[retained].mean()) if retained.any() else 0.0,
            "outcomes": counts,
            "n_available": int(avail.sum()),
            "measured_repair_rate": (counts["repaired"] / n_avail_err
                                     if n_avail_err else None),
            "measured_damage_rate": (counts["damaged"] / n_avail_ok
                                     if n_avail_ok else None),
            "final_errors": final_errors,
            "final_error_rate": final_errors / n if n else None,
            "error_rate_reduction": (base_errors - final_errors) / n if n else None,
        })
    cov = np.array([1.0 - r["actual_frac_flagged"] for r in rows])
    risk = np.array([r["selective_risk"] for r in rows])
    order = np.argsort(cov)
    aurc = float(selective._trapezoid(risk[order], cov[order])) if len(order) > 1 else None
    return {"rows": rows, "aurc": aurc}


# ---------------------------------------------------------------------------
# Is the improvement real?
# ---------------------------------------------------------------------------

def _boot_summary(samples: np.ndarray) -> dict:
    """Percentile CI and a two-sided bootstrap p-value for one quantity."""
    a = np.asarray(samples, dtype=np.float64)
    p = 2.0 * min(float((a <= 0).mean()), float((a >= 0).mean()))
    return {"mean": float(a.mean()),
            "ci_low": float(np.percentile(a, 2.5)),
            "ci_high": float(np.percentile(a, 97.5)),
            "p_value": float(min(1.0, p))}


def bootstrap_significance(y, doc_ids, net_delta, signals, sig_names,
                           budget, regime, *, n_boot, seed, ref="probe_fused"):
    """Document-level bootstrap over the MEASURED outcomes.

    The error-rate reduction is a few dozen fields out of five thousand, so a
    point estimate alone says nothing about whether it would survive a different
    sample of documents. Documents are the unit of independence -- fields within
    one document share a trace, a schema and a gold record -- so whole documents
    are resampled with replacement, never individual fields.

    Every comparison is PAIRED: within one replicate, all signals are evaluated
    on the same resampled corpus, and the difference is taken there. That
    removes the between-replicate variance that would otherwise swamp a
    difference this small.

    Flags are recomputed inside each replicate for the global regime, because a
    global budget is defined over whatever corpus you have. Per-document
    flagging does not depend on the rest of the corpus, so it is computed once
    and reused -- the same masks, not an approximation of them.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(doc_ids)
    rows_by_doc = [np.flatnonzero(doc_ids == d) for d in uniq]
    n_docs = len(rows_by_doc)
    # One fixed tie-break vector, so ties resolve the same way in every
    # replicate and the comparison stays paired.
    tie = rng.random(len(y))

    per_doc_flag = {}
    if regime == "per_doc":
        for s in sig_names:
            per_doc_flag[s] = selective._flag_per_doc(signals[s], doc_ids, budget,
                                               np.random.default_rng(seed))

    reduction, rates = [], {s: [] for s in sig_names}
    diffs = {s: [] for s in sig_names if s != ref}
    for _ in range(n_boot):
        pick = rng.integers(0, n_docs, n_docs)
        idx = np.concatenate([rows_by_doc[p] for p in pick])
        yb, ndb = y[idx], net_delta[idx]
        n_b = len(idx)
        base_errors = int(yb.sum())
        r = {}
        for s in sig_names:
            if regime == "per_doc":
                fl = per_doc_flag[s][idx]
            else:
                k = int(round(budget * n_b))
                fl = np.zeros(n_b, dtype=bool)
                if k > 0:
                    order = np.lexsort((tie[idx], -signals[s][idx]))
                    fl[order[:k]] = True
            r[s] = (base_errors + int(ndb[fl].sum())) / n_b
            rates[s].append(r[s])
        reduction.append(base_errors / n_b - r[ref])
        for s in diffs:
            # Positive means the reference signal left FEWER errors behind.
            diffs[s].append(r[s] - r[ref])

    tests = {"reduction_vs_baseline": _boot_summary(reduction)}
    for s in diffs:
        tests[f"{ref}_vs_{s}"] = _boot_summary(diffs[s])
    holm = controls.holm_bonferroni({k: v["p_value"] for k, v in tests.items()})
    for k, v in tests.items():
        v["p_holm"] = holm.get(k)

    return {
        "budget": budget, "regime": regime, "n_boot": n_boot, "reference": ref,
        "per_signal_error_rate": {
            s: {"mean": float(np.mean(v)),
                "ci_low": float(np.percentile(v, 2.5)),
                "ci_high": float(np.percentile(v, 97.5))}
            for s, v in rates.items()},
        "tests": tests,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_scored_fields(scores_path: Path, labels_dir: Path):
    """Join Stage 7's scored rows to the JSON path of each field.

    The score dump identifies fields by path_str; swapping needs the structured
    path, which lives in the Stage 2 label file. Labels are also the source of
    truth for `y`, and a mismatch against the dump means the two stages saw
    different labels -- that is a hard error, not something to average over.
    """
    dump = json.loads(scores_path.read_text())
    by_doc: dict[str, list] = {}
    for row in dump["fields"]:
        by_doc.setdefault(row["doc_id"], []).append(row)

    scored, order, n_no_path, n_y_mismatch = {}, [], 0, 0
    for doc_id, rows in by_doc.items():
        lab_path = labels_dir / f"{doc_id}.json"
        if not lab_path.exists():
            n_no_path += len(rows)
            continue
        labels = json.loads(lab_path.read_text())["labels"]
        paths = {l["path_str"]: l["path"] for l in labels}
        ys = {l["path_str"]: int(l["is_error"]) for l in labels}
        keep = []
        for r in rows:
            ps = r["path_str"]
            if ps not in paths:
                n_no_path += 1
                continue
            if ys[ps] != r["y"]:
                n_y_mismatch += 1
                continue
            keep.append((ps, paths[ps], r["y"], r["scores"]))
        if keep:
            scored[doc_id] = keep
            order.append(doc_id)
    return dump, scored, order, n_no_path, n_y_mismatch


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_dir=cfg.logging.log_dir,
                  log_name="09_regen_evaluate", log_to_file=cfg.logging.log_to_file)

    artifacts = cfg.artifacts_path
    results_dir = artifacts / "results"
    labels_dir = artifacts / "labels"
    extractions_dir = artifacts / "extractions"
    regen_dir = artifacts / "regen"

    scores_path = results_dir / args.scores_name
    if not scores_path.exists():
        logger.error("No OOF score dump at %s. Run Stage 7 first -- it writes "
                     "the per-field scores this stage ranks by.", scores_path)
        return 1
    if not regen_dir.is_dir():
        logger.error("No regenerations at %s -- run Stage 8 first.", regen_dir)
        return 1

    dump, scored, doc_order, n_no_path, n_y_mismatch = _load_scored_fields(
        scores_path, labels_dir)
    if n_no_path:
        logger.warning("%d scored fields had no path in the label files "
                       "(skipped).", n_no_path)
    if n_y_mismatch:
        logger.error("%d scored fields disagree with the current labels. The "
                     "score dump and the labels came from different Stage 2 "
                     "runs; re-run Stage 7 against these labels.", n_y_mismatch)
        return 1

    # Restrict to documents Stage 8 actually regenerated.
    missing = [d for d in doc_order if not (regen_dir / f"{d}.json").exists()]
    if missing:
        msg = ("%d of %d scored documents have no regeneration." %
               (len(missing), len(doc_order)))
        if not args.allow_partial:
            logger.error("%s Refusing to report a number computed on a subset. "
                         "Finish Stage 8 (it has --resume), or pass "
                         "--allow-partial deliberately. First few: %s",
                         msg, ", ".join(missing[:5]))
            return 1
        logger.warning("%s Proceeding under --allow-partial.", msg)
    doc_order = [d for d in doc_order if d not in set(missing)]
    if args.limit_docs:
        doc_order = doc_order[:args.limit_docs]
    if not doc_order:
        logger.error("No documents left to evaluate.")
        return 1

    # Gold + schema, loaded exactly the way Stage 2 loaded them.
    s2 = _stage02()
    benchmark = s2.load_benchmark(cfg)
    gold_by_id, schema_by_id, domain_by_id = {}, {}, {}
    for doc in benchmark:
        gold_by_id[doc.doc_id] = doc.gold
        schema_by_id[doc.doc_id] = doc.schema
        domain_by_id[doc.doc_id] = doc.domain

    mode = getattr(cfg.labeling, "match_mode", "strict")
    if mode not in s2._MODE_PARAMS:
        logger.warning("Unknown match_mode %r; falling back to 'strict'.", mode)
        mode = "strict"
    mode_params = s2._MODE_PARAMS[mode]
    logger.info("Labeling mode: %s (same settings as Stage 2).", mode)

    no_gold = [d for d in doc_order if d not in gold_by_id]
    if no_gold:
        logger.error(
            "%d of %d documents have no gold annotation in the loaded benchmark "
            "(e.g. %s). Without gold the re-labeling would mark everything wrong "
            "and the measured repair rate would be meaningless. Check "
            "data.benchmark_path / data.max_documents.",
            len(no_gold), len(doc_order), ", ".join(no_gold[:3]))
        return 1

    # ---- the records: the originals and what the model said the second time --
    originals, sample_sets = {}, {}
    usability = {"total": 0, "truncated": 0, "parse_error": 0, "no_json": 0,
                 "usable": 0, "docs_with_no_usable_sample": 0}
    for doc_id in doc_order:
        ext = json.loads((extractions_dir / f"{doc_id}.json").read_text())
        originals[doc_id] = ext.get("parsed_json")
        samples, st = usable_samples(
            json.loads((regen_dir / f"{doc_id}.json").read_text()))
        for k, v in st.items():
            usability[k] += v
        if not samples:
            usability["docs_with_no_usable_sample"] += 1
        sample_sets[doc_id] = samples

    logger.info("Resamples: %d generated, %d usable (%d truncated, %d parse "
                "errors, %d without JSON). %d documents have no usable sample.",
                usability["total"], usability["usable"], usability["truncated"],
                usability["parse_error"], usability["no_json"],
                usability["docs_with_no_usable_sample"])
    if usability["usable"] == 0:
        logger.error("No usable resamples at all -- nothing to measure.")
        return 1

    # ---- pooled arrays, in a fixed order shared by every strategy/signal ----
    rows_meta = []           # (doc_id, path_str)
    y_list, doc_idx_list = [], []
    sig_lists: dict[str, list] = {s: [] for s in dump["signals"]}
    for di, doc_id in enumerate(doc_order):
        for ps, _p, y0, sc in scored[doc_id]:
            rows_meta.append((doc_id, ps))
            y_list.append(y0)
            doc_idx_list.append(di)
            for s in dump["signals"]:
                sig_lists[s].append(sc[s])
    y = np.array(y_list, dtype=np.int64)
    doc_ids = np.array(doc_idx_list, dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    signals = {s: np.asarray(v, dtype=np.float64) for s, v in sig_lists.items()}
    signals["random"] = rng.random(len(y))
    signals["oracle"] = y.astype(np.float64) + rng.random(len(y)) * 1e-6
    use_signals = [s for s in args.signals if s in signals]

    n = len(y)
    base_errors = int(y.sum())
    logger.info("Evaluating %d fields in %d documents; %d are errors (%.1f%%).",
                n, len(doc_order), base_errors, 100 * y.mean())

    # ---- SELF-CHECK --------------------------------------------------------
    # Re-label each ORIGINAL record and require it to reproduce the labels
    # Stage 2 stored. Everything this stage reports is a difference between a
    # before-label and an after-label; if the two are not produced under
    # identical conditions -- same gold, same schema, same matcher settings --
    # then every repair and damage count is an artifact of that difference, and
    # it would not announce itself. One labeling per document buys the guard.
    n_mismatch, examples = 0, []
    for doc_id in doc_order:
        labeler = make_labeler(
            doc_id=doc_id, domain=domain_by_id.get(doc_id, ""),
            schema=schema_by_id.get(doc_id, {}), gold=gold_by_id[doc_id],
            fuzzy_threshold=cfg.labeling.fuzzy_threshold,
            number_tolerance=cfg.labeling.number_tolerance,
            mode_params=mode_params)
        got = labeler(originals[doc_id])
        for ps, _p, y0, _sc in scored[doc_id]:
            if got.get(ps) != y0:
                n_mismatch += 1
                if len(examples) < 5:
                    examples.append(
                        f"{doc_id}:{ps} stored={y0} recomputed={got.get(ps)}")
    if n_mismatch:
        logger.error(
            "Self-check failed: re-labeling the ORIGINAL extractions gets %d of "
            "%d scored labels wrong, so the before/after comparison would not "
            "be like-for-like. Gold, schema or matcher settings differ from the "
            "Stage 2 run that produced the labels. Examples: %s",
            n_mismatch, n, "; ".join(examples))
        return 1
    logger.info("Self-check passed: re-labeling the originals reproduces all "
                "%d stored labels, so before and after are like-for-like.", n)

    # ---- flag masks for the joint checkpoints (score-only, so computable now)
    row_index = {k: i for i, k in enumerate(rows_meta)}
    joint_by_doc: dict[str, dict[str, list[str]]] = {d: {} for d in doc_order}
    ck_names = []
    if args.checkpoint_signal in signals:
        for b in args.checkpoint_budgets:
            name = f"{args.checkpoint_signal}@{b:g}"
            ck_names.append(name)
            mask = selective._flag_global(signals[args.checkpoint_signal], b,
                                   np.random.default_rng(args.seed))
            for i in np.flatnonzero(mask):
                d, ps = rows_meta[i]
                joint_by_doc[d].setdefault(name, []).append(ps)

    # ---- the measurement pass ---------------------------------------------
    from joblib import Parallel, delayed
    from tqdm import tqdm

    per_strategy: dict[str, Any] = {}
    for strategy in args.strategies:
        logger.info("=" * 70)
        logger.info("STRATEGY: %s", strategy)
        jobs = [
            delayed(measure_document)(
                doc_id, domain_by_id.get(doc_id, ""), originals[doc_id],
                gold_by_id.get(doc_id, {}), schema_by_id.get(doc_id, {}),
                [(ps, p, y0) for ps, p, y0, _sc in scored[doc_id]],
                sample_sets[doc_id], strategy, joint_by_doc.get(doc_id),
                cfg.labeling.fuzzy_threshold, cfg.labeling.number_tolerance,
                mode_params)
            for doc_id in doc_order
        ]
        if args.jobs == 1:
            results = [j[0](*j[1], **j[2]) for j in tqdm(jobs, desc=strategy)]
        else:
            results = Parallel(n_jobs=args.jobs, prefer="processes")(
                tqdm(jobs, desc=strategy))
        by_doc = dict(results)

        net_delta = np.zeros(n, dtype=np.int64)
        self_out = np.empty(n, dtype=object)
        status = np.empty(n, dtype=object)
        for doc_id, res in by_doc.items():
            for rec in res["fields"]:
                i = row_index[(doc_id, rec["path_str"])]
                net_delta[i] = rec["net_delta"]
                self_out[i] = rec["self"]
                status[i] = rec["status"]

        status_counts = {s: int((status == s).sum()) for s in
                         ("swapped", "identical", "unavailable", "set_failed",
                          "label_failed")}
        outcome_counts = {o: int((self_out == o).sum()) for o in
                          ("repaired", "damaged", "unchanged", "unavailable", "lost")}
        avail = self_out != "unavailable"
        n_avail_err = int(y[avail].sum())
        n_avail_ok = int(avail.sum()) - n_avail_err
        measured = {
            "repair_rate": outcome_counts["repaired"] / n_avail_err if n_avail_err else None,
            "damage_rate": outcome_counts["damaged"] / n_avail_ok if n_avail_ok else None,
            "n_available_errors": n_avail_err,
            "n_available_correct": n_avail_ok,
        }
        logger.info("Field availability: %s", status_counts)
        if status_counts["label_failed"]:
            kinds: dict[str, int] = {}
            for res in by_doc.values():
                for rec in res["fields"]:
                    if rec["status"] == "label_failed":
                        kinds[rec.get("error", "?")] = kinds.get(rec.get("error", "?"), 0) + 1
            logger.warning(
                "%d of %d fields could not be re-labeled after the swap (%s). "
                "They are excluded from the repair/damage denominators, so this "
                "is lost coverage -- report it rather than ignoring it.",
                status_counts["label_failed"], n, kinds)
        logger.info("Swap outcomes (all scored fields): %s", outcome_counts)
        logger.info("MEASURED repair rate %.3f (of %d available errors); "
                    "MEASURED damage rate %.3f (of %d available correct fields). "
                    "Stage 7 assumed 0.700 / 0.050.",
                    measured["repair_rate"] or 0.0, n_avail_err,
                    measured["damage_rate"] or 0.0, n_avail_ok)

        curves = {}
        for regime in ("global", "per_doc"):
            curves[regime] = {}
            for sig in use_signals:
                curves[regime][sig] = measured_curve(
                    signals[sig], y, doc_ids, net_delta, self_out,
                    args.budgets, regime, rng=np.random.default_rng(args.seed))
            logger.info("-" * 70)
            logger.info("%s budget -- MEASURED error rate after regeneration "
                        "(baseline %.1f%%):", regime, 100 * base_errors / n)
            logger.info("    %-14s%s", "signal",
                        "".join(f"{b:>8.0%}" for b in args.budgets))
            for sig in use_signals:
                cells = "".join(
                    f"{r['final_error_rate']:>8.1%}"
                    for r in curves[regime][sig]["rows"])
                logger.info("    %-14s%s", sig, cells)

        # Joint validation of the additive shortcut.
        validation = []
        for name in ck_names:
            actual, n_skipped = 0, 0
            for res in by_doc.values():
                entry = res.get("joint", {}).get(name)
                # `errors` is explicitly None when that document's joint
                # re-labeling failed -- a missing key and a null are different
                # things here, and dict.get's default would not catch the null.
                if entry is None or entry.get("errors") is None:
                    actual += res["base_errors"]
                    if entry is not None:
                        n_skipped += 1
                else:
                    actual += entry["errors"]
            b = float(name.rsplit("@", 1)[1])
            mask = selective._flag_global(signals[args.checkpoint_signal], b,
                                   np.random.default_rng(args.seed))
            pred = base_errors + int(net_delta[mask].sum())
            validation.append({
                "checkpoint": name, "budget": b,
                "additive_prediction": pred, "joint_actual": int(actual),
                "difference": int(actual) - pred,
                "difference_rate_points": (int(actual) - pred) / n,
                "n_documents_skipped": n_skipped,
            })
        if validation:
            logger.info("-" * 70)
            logger.info("Additivity check (joint re-labeling vs the per-field sum):")
            for v in validation:
                note = (f"  [{v['n_documents_skipped']} docs unlabelable]"
                        if v["n_documents_skipped"] else "")
                logger.info("  %-22s predicted %5d errors, joint gives %5d "
                            "(%+d, %+.2f pts)%s", v["checkpoint"],
                            v["additive_prediction"], v["joint_actual"],
                            v["difference"], 100 * v["difference_rate_points"], note)

        # Control: what plain resampling is worth with no probe at all. Uses
        # the first usable sample as a whole record -- no selection, no swap.
        full_err, full_scored, full_missing, full_failed = 0, 0, 0, 0
        for doc_id in doc_order:
            samples = sample_sets[doc_id]
            before = {ps: y0 for ps, _p, y0, _sc in scored[doc_id]}
            full_scored += len(before)
            if not samples:
                full_err += sum(before.values())
                full_missing += len(before)
                continue
            labeler = make_labeler(
                doc_id=doc_id, domain=domain_by_id.get(doc_id, ""),
                schema=schema_by_id.get(doc_id, {}), gold=gold_by_id.get(doc_id, {}),
                fuzzy_threshold=cfg.labeling.fuzzy_threshold,
                number_tolerance=cfg.labeling.number_tolerance,
                mode_params=mode_params)
            try:
                after = labeler(samples[0])
            except Exception as e:
                # A whole regenerated record can nest deeper than the matcher can
                # walk. This is a CONTROL, not the measurement -- it must not take
                # the run down with it. The document keeps its original labels and
                # the loss of coverage is reported.
                logger.debug("control labeling failed for %s: %s", doc_id, e)
                full_err += sum(before.values())
                full_failed += 1
                continue
            full_err += errors_over_scored(after, before)
            full_missing += sum(1 for ps in before if ps not in after)
        control_full = {
            "error_rate": full_err / full_scored if full_scored else None,
            "n_scored_paths_missing_from_resample": full_missing,
            "n_documents_unlabelable": full_failed,
        }
        logger.info("CONTROL -- replacing the whole record with the resample "
                    "(no probe): error rate %.1f%% vs %.1f%% for the original.%s",
                    100 * (control_full["error_rate"] or 0), 100 * base_errors / n,
                    f"  [{full_failed} documents unlabelable, kept as-is]"
                    if full_failed else "")

        def at(regime, sig, b=0.20):
            for r in curves[regime][sig]["rows"]:
                if abs(r["budget"] - b) < 1e-9:
                    return r
            return None

        headline = {}
        have = set(use_signals)
        for regime in ("global", "per_doc"):
            if "probe_fused" not in have:
                break
            rf = at(regime, "probe_fused")
            ra = at(regime, "probe_answer") if "probe_answer" in have else None
            lp = [s for s in ("mean_logprob", "min_logprob") if s in have]
            # "Best" baseline = the one that leaves the FEWEST errors behind.
            base_sig = min(lp, key=lambda s: (at(regime, s) or {}).get(
                "final_error_rate") or 1.0) if lp else None
            rb = at(regime, base_sig) if base_sig else None
            if rf and rb:
                headline[regime] = {
                    "budget": 0.20,
                    "baseline_error_rate": base_errors / n,
                    "probe_fused_error_rate": rf["final_error_rate"],
                    "probe_answer_error_rate": ra["final_error_rate"] if ra else None,
                    "best_logprob_baseline": base_sig,
                    "logprob_error_rate": rb["final_error_rate"],
                    "probe_fused_repaired": rf["outcomes"]["repaired"],
                    "probe_fused_damaged": rf["outcomes"]["damaged"],
                }
        logger.info("HEADLINE at a 20%% budget (%s):", strategy)
        for regime, h in headline.items():
            logger.info("  %-8s %.1f%% -> %.1f%% with the probe (%d repaired, "
                        "%d damaged); %s reaches %.1f%%",
                        regime, 100 * h["baseline_error_rate"],
                        100 * h["probe_fused_error_rate"],
                        h["probe_fused_repaired"], h["probe_fused_damaged"],
                        h["best_logprob_baseline"], 100 * h["logprob_error_rate"])

        # Is the improvement real, or a few dozen fields of luck?
        significance = []
        if args.bootstrap > 0:
            boot_signals = [s for s in ("probe_fused", "probe_answer",
                                        "min_logprob", "mean_logprob")
                            if s in signals]
            if "probe_fused" in boot_signals:
                logger.info("-" * 70)
                logger.info("SIGNIFICANCE (%d document-level bootstrap replicates, "
                            "paired, Holm-corrected within each budget/regime):",
                            args.bootstrap)
                for regime in ("global", "per_doc"):
                    for b in args.significance_budgets:
                        sg = bootstrap_significance(
                            y, doc_ids, net_delta, signals, boot_signals,
                            b, regime, n_boot=args.bootstrap, seed=args.seed)
                        significance.append(sg)
                        red = sg["tests"]["reduction_vs_baseline"]
                        logger.info(
                            "  %-8s @%3.0f%%  error rate %.1f%% -> %.1f%%  "
                            "(reduction %+.2f pts, 95%% CI [%+.2f, %+.2f], "
                            "p=%.4g, Holm %.4g)%s",
                            regime, 100 * b, 100 * base_errors / n,
                            100 * sg["per_signal_error_rate"]["probe_fused"]["mean"],
                            100 * red["mean"], 100 * red["ci_low"],
                            100 * red["ci_high"], red["p_value"], red["p_holm"],
                            "" if red["ci_low"] > 0 else "   <-- CI includes zero")
                        for k, v in sg["tests"].items():
                            if k == "reduction_vs_baseline":
                                continue
                            logger.info(
                                "             %-28s %+.2f pts [%+.2f, %+.2f] "
                                "p=%.4g Holm %.4g%s", k.replace("probe_fused_vs_", "vs "),
                                100 * v["mean"], 100 * v["ci_low"], 100 * v["ci_high"],
                                v["p_value"], v["p_holm"],
                                "" if v["ci_low"] > 0 else "   (ns)")

        per_strategy[strategy] = {
            "significance": significance,
            "status_counts": status_counts,
            "outcome_counts": outcome_counts,
            "measured_rates": measured,
            "curves": curves,
            "additivity_check": validation,
            "control_full_resample": control_full,
            "headline": headline,
        }

    out = {
        "layer": dump.get("layer"),
        "fused_variant": dump.get("fused_variant"),
        "labeling_mode": mode,
        "n_docs": len(doc_order), "n_fields": n,
        "baseline_errors": base_errors, "baseline_error_rate": base_errors / n,
        "n_docs_without_regeneration": len(missing),
        "budgets": args.budgets,
        "resample_usability": usability,
        "strategies": per_strategy,
    }
    out_path = results_dir / args.out_name
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("=" * 70)
    logger.info("Saved -> %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

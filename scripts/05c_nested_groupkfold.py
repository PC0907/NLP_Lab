#!/usr/bin/env python3
"""Nested grouped K-fold: unbiased layer selection for many small records (SOB).

WHY NOT 05b
  05b holds out one document per outer fold and one per inner fold. That suits
  28 ExtractBench documents. On ~1,000 SOB records it needs ~N^2 probe fits per
  layer, and most single-record folds are one-class (skipped).

DESIGN
  Outer: StratifiedGroupKFold(K_out), grouped by record, stratified on field
         labels so every fold has errors. Reported on, never used for selection.
  Inner: StratifiedGroupKFold(K_in) on the outer-train records only. Each
         candidate layer gets a pooled inner-OOF AUROC; the best is selected.
  The selected layer is retrained on all outer-train records and scores the
  outer-test fold. Headline = pooled out-of-fold AUROC (same as 05b).

  Features, labels and the probe are identical to 05b: X, y, doc_ids come from
  05b.load_layer_matrix and every fit goes through 05b._fit_score.

ALSO
  * Fixed-layer grouped K-fold on the same outer splits (the naive per-layer
    numbers; their max is optimistically biased, the nested number is not).
  * Final probes per candidate layer, trained on all records, saved as
    LinearProbe pickles at <artifacts>/probes/probe_layer{L}.pkl
    (same format as 03_train_probe.py, so 07_regen_single.py can load them).
  * oof_scores.npz keyed by (doc_id, path_str), matching the regeneration
    scripts' field key.
  * Data checks: records lost between extraction and labeling, and an error dump.

Report goes to stdout; progress logging goes to stderr. CPU only.

Usage:
    python scripts/05c_nested_groupkfold.py --config configs/exp_qwen35_4b_sob_1k.yaml \
        --layers 14 16 18 20 22
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import re
import sys
import time
from collections import Counter
from importlib import import_module
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score, precision_recall_curve)
from sklearn.model_selection import StratifiedGroupKFold

from probe_extraction.config import load_config
from probe_extraction.probes.linear import LinearProbe, ProbeMetrics
from probe_extraction.utils.logging import setup_logging

sys.path.insert(0, str(Path(__file__).parent))
_nl = import_module("05b_nested_lodo")
load_layer_matrix = _nl.load_layer_matrix
_fit_score = _nl._fit_score

import logging
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Nested grouped K-fold (SOB)")
    p.add_argument("--config", required=True)
    p.add_argument("--layers", type=int, nargs="*", default=None)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=5)
    p.add_argument("--n-dump", type=int, default=30,
                   help="Number of random errors to print in the error dump.")
    p.add_argument("--skip-fixed-sweep", action="store_true",
                   help="Skip the fixed-layer sweep (saves ~25 fits).")
    p.add_argument("--skip-final-probes", action="store_true")
    p.add_argument("--list-agg", choices=["none", "last", "mean"], default="last",
                   help="List-valued fields: labels are per list, activations per "
                        "element (path.0, path.1, ...). 'last' = last token of the last "
                        "element, 'mean' = mean of each element's last token, "
                        "'none' = drop list fields (05b behaviour).")
    p.add_argument("--out-name", default="nested_groupkfold.json")
    return p.parse_args()


def report(msg=""):
    print(msg, flush=True)


# ----------------------------------------------------------------------------
# Data checks
# ----------------------------------------------------------------------------

def _short(v, n=70):
    s = json.dumps(v, ensure_ascii=False, default=str) if not isinstance(v, str) else v
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 3] + "..."


def data_checks(art: Path, n_dump: int, seed: int):
    ext_dir, lab_dir, act_dir = art / "extractions", art / "labels", art / "activations"
    report("=" * 72)
    report("CHECK 1: records lost between extraction and labeling")
    report("=" * 72)

    ext_files = {p.stem for p in ext_dir.glob("*.json") if not p.name.startswith("_")}
    lab_files = {p.stem for p in lab_dir.glob("*.json") if not p.name.startswith("_")}
    act_files = {p.stem for p in act_dir.glob("*.npz")}
    report(f"extraction files: {len(ext_files)}   label files: {len(lab_files)}   "
           f"activation files: {len(act_files)}")

    summ_path = ext_dir / "_summary.json"
    if summ_path.exists():
        s = json.load(summ_path.open())
        report(f"extraction summary: total={s.get('total_documents')} "
               f"successful={s.get('successful')} parse_failed={s.get('parse_failed')} "
               f"no_text={s.get('skipped_no_text')} "
               f"truncated(finish=length)={s.get('finish_length_truncated')}")
        bad = [d for d in s.get("per_document", []) if d.get("parse_error")]
        for d in bad:
            report(f"  FAILED {d['doc_id']}: finish={d.get('finish_reason')} "
                   f"tokens={d.get('generated_tokens')} err={_short(d['parse_error'], 90)}")
    else:
        report("no extractions/_summary.json found")

    for sid in sorted(ext_files - lab_files):
        e = json.load((ext_dir / f"{sid}.json").open())
        reason = e.get("parse_error") or "no parse_error (likely no gold for this id)"
        report(f"  extracted but not labeled: {sid} -> {_short(reason, 90)}")
    for sid in sorted(lab_files - act_files):
        report(f"  labeled but no activations: {sid}")

    report("")
    report("=" * 72)
    report("CHECK 2: error dump")
    report("=" * 72)
    labels = []
    zero_field_recs = 0
    for p in sorted(lab_dir.glob("*.json")):
        if p.name.startswith("_"):
            continue
        d = json.load(p.open())
        if not d.get("labels"):
            zero_field_recs += 1
        for l in d.get("labels", []):
            labels.append((d.get("doc_id", p.stem), l))
    n = len(labels)
    errs = [(doc, l) for doc, l in labels if l.get("is_error")]
    present = [(doc, l) for doc, l in labels if l.get("extracted_present", True)]
    err_present = [x for x in present if x[1].get("is_error")]
    report(f"all labeled fields: {n}, errors: {len(errs)} ({100*len(errs)/max(n,1):.1f}%)")
    report(f"probe-eligible (extracted_present): {len(present)}, errors: {len(err_present)} "
           f"({100*len(err_present)/max(len(present),1):.1f}%)")
    report(f"records with zero labeled fields: {zero_field_recs}")
    report(f"error types (all): {dict(Counter(l.get('error_type') for _, l in errs))}")
    report(f"comparison strategy of errors: "
           f"{dict(Counter(l.get('comparison_strategy') for _, l in errs))}")
    per_doc = Counter(doc for doc, _ in errs)
    report(f"records with >=1 error: {len(per_doc)}; top 5 by error count: "
           f"{per_doc.most_common(5)}")

    rng = random.Random(seed)
    sample = rng.sample(errs, min(n_dump, len(errs)))
    report(f"\nrandom sample of {len(sample)} errors (type | strategy | path | gold | extracted):")
    for doc, l in sample:
        report(f"  [{l.get('error_type')}|{l.get('comparison_strategy')}] {doc} :: "
               f"{_short(l.get('path_str'), 45)}")
        report(f"      gold={_short(l.get('gold_value'))}")
        report(f"      ext ={_short(l.get('extracted_value'))}")
    report("")


# ----------------------------------------------------------------------------
# Field paths aligned with 05b.load_layer_matrix
# ----------------------------------------------------------------------------

def aligned_paths(labels_dir: Path, activations_dir: Path, layers):
    """Replay 05b.load_layer_matrix's filter order (no domain/doc filters) and
    return path_str per row. Only reads npz key lists, not arrays."""
    doc_ids, paths, ys = [], [], []
    for lp in sorted(labels_dir.glob("*.json")):
        if lp.name.startswith("_"):
            continue
        doc_id = lp.stem
        data = json.load(lp.open())
        npz = activations_dir / f"{doc_id}.npz"
        if not npz.exists():
            continue
        with np.load(npz) as acts:
            keys = set(acts.files)
        for fld in data.get("labels", []):
            if not fld.get("extracted_present", True):
                continue
            ps = fld["path_str"]
            if all(f"{ps}__layer{L}" in keys for L in layers):
                doc_ids.append(doc_id)
                paths.append(ps)
                ys.append(int(fld.get("is_error", 0)))
    return np.array(doc_ids), np.array(paths), np.array(ys)


ELEM_RE = re.compile(r"^(.*)\.(\d+)__layer(\d+)$")


def load_with_lists(labels_dir: Path, activations_dir: Path, layers, agg):
    """Same filters and scalar features as 05b.load_layer_matrix, plus list
    fields whose own key is absent but whose elements path.{i} have
    activations at every layer. Returns X, y, doc_ids, paths, kind
    (kind: 0 scalar, 1 list)."""
    rows = []
    n_unmatched, unmatched_ex = 0, []
    for lp in sorted(labels_dir.glob("*.json")):
        if lp.name.startswith("_"):
            continue
        doc_id = lp.stem
        data = json.load(lp.open())
        npz = activations_dir / f"{doc_id}.npz"
        if not npz.exists():
            continue
        with np.load(npz) as acts:
            keys = set(acts.files)
            elems = {}
            for k in keys:
                m = ELEM_RE.match(k)
                if m:
                    elems.setdefault((m.group(1), int(m.group(3))), []).append(int(m.group(2)))
            for fld in data.get("labels", []):
                if not fld.get("extracted_present", True):
                    continue
                ps = fld["path_str"]
                if all(f"{ps}__layer{L}" in keys for L in layers):
                    vecs = {}
                    for L in layers:
                        v = acts[f"{ps}__layer{L}"].astype(np.float32)
                        vecs[L] = v[-1] if v.ndim > 1 else v
                    rows.append((doc_id, ps, 0, int(fld.get("is_error", 0)), vecs))
                    continue
                if agg != "none" and all((ps, L) in elems for L in layers):
                    vecs = {}
                    for L in layers:
                        idx = sorted(elems[(ps, L)])
                        use = idx[-1:] if agg == "last" else idx
                        lasts = []
                        for i in use:
                            v = acts[f"{ps}.{i}__layer{L}"].astype(np.float32)
                            lasts.append(v[-1] if v.ndim > 1 else v)
                        vecs[L] = np.mean(lasts, axis=0)
                    rows.append((doc_id, ps, 1, int(fld.get("is_error", 0)), vecs))
                    continue
                n_unmatched += 1
                if len(unmatched_ex) < 5:
                    unmatched_ex.append((doc_id[:24], ps, type(fld.get("extracted_value")).__name__))
    logger.info("fields present but without usable activations: %d, e.g. %s",
                n_unmatched, unmatched_ex)
    doc_ids = np.array([r[0] for r in rows])
    paths = np.array([r[1] for r in rows])
    kind = np.array([r[2] for r in rows])
    y = np.array([r[3] for r in rows])
    X = {L: np.stack([r[4][L] for r in rows]) for L in layers}
    return X, y, doc_ids, paths, kind, n_unmatched


# ----------------------------------------------------------------------------
# CV
# ----------------------------------------------------------------------------

def pooled_auroc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def select_layer_inner(X, y, groups, idx, layers, C, k, seed):
    """Pooled inner-OOF AUROC per layer, using only rows in idx."""
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    splits = list(sgkf.split(idx, y[idx], groups[idx]))
    per_layer = {}
    for L in layers:
        oof = np.full(len(idx), np.nan)
        for tr, te in splits:
            if y[idx[tr]].sum() in (0, len(tr)):
                continue
            oof[te] = _fit_score(X[L][idx[tr]], y[idx[tr]], X[L][idx[te]], C)
        ok = ~np.isnan(oof)
        per_layer[L] = pooled_auroc(y[idx][ok], oof[ok])
    best = max(per_layer, key=lambda L: -1 if np.isnan(per_layer[L]) else per_layer[L])
    return best, per_layer


def threshold_at_recall(y, s, target=0.5):
    prec, rec, thr = precision_recall_curve(y, s)
    ok = np.where(rec[:-1] >= target)[0]
    return float(thr[ok[-1]]) if len(ok) else None


def main():
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_dir=cfg.logging.log_dir,
                  log_name="05c_nested_groupkfold", log_to_file=cfg.logging.log_to_file)
    seed = cfg.experiment.seed
    art = cfg.artifacts_path
    labels_dir, acts_dir = art / "labels", art / "activations"
    layers = args.layers or cfg.activations.layers

    data_checks(art, args.n_dump, seed)

    t0 = time.time()
    logger.info("Loading activations for layers %s ...", layers)
    if args.list_agg == "none":
        X, y, groups = load_layer_matrix(labels_dir, acts_dir, layers)
        p_docs, paths, p_y = aligned_paths(labels_dir, acts_dir, layers)
        if not (np.array_equal(p_docs, groups) and np.array_equal(p_y, y)):
            raise RuntimeError("path alignment with 05b.load_layer_matrix failed; "
                               "05b's filters have changed, update aligned_paths()")
        kind = np.zeros(len(y), dtype=int)
        n_unmatched = None
    else:
        X, y, groups, paths, kind, n_unmatched = load_with_lists(
            labels_dir, acts_dir, layers, args.list_agg)
    logger.info("Loaded in %.0fs", time.time() - t0)

    report("=" * 72)
    report("PROBE DATA")
    report("=" * 72)
    report(f"fields={len(y)} records={len(set(groups))} errors={int(y.sum())} "
           f"({100*y.mean():.1f}%) dim={X[layers[0]].shape[1]} layers={layers}")
    report(f"list-agg={args.list_agg}  scalar fields={int((kind == 0).sum())} "
           f"(err {int(y[kind == 0].sum())})  list fields={int((kind == 1).sum())} "
           f"(err {int(y[kind == 1].sum())})  present-but-unusable={n_unmatched}")
    recs_with_err = len(set(groups[y == 1]))
    report(f"records with >=1 probe-eligible error: {recs_with_err}")

    outer = StratifiedGroupKFold(n_splits=args.outer_folds, shuffle=True, random_state=seed)
    outer_splits = list(outer.split(np.zeros(len(y)), y, groups))
    fold_id = np.full(len(y), -1)
    for f, (_, te) in enumerate(outer_splits):
        fold_id[te] = f

    # ---- nested ----
    oof = np.full(len(y), np.nan)
    sel_layer = np.full(len(y), -1)
    fold_rows = []
    for f, (tr, te) in enumerate(outer_splits):
        t = time.time()
        L, inner = select_layer_inner(X, y, groups, tr, layers, args.C,
                                      args.inner_folds, seed + 1 + f)
        s = _fit_score(X[L][tr], y[tr], X[L][te], args.C)
        oof[te] = s
        sel_layer[te] = L
        a = pooled_auroc(y[te], s)
        fold_rows.append({"fold": f, "layer": L, "test_auroc": a,
                          "n_test": int(len(te)), "n_test_err": int(y[te].sum()),
                          "inner_auroc": {int(k): v for k, v in inner.items()}})
        logger.info("outer %d: layer %d, inner %s, test AUROC %.4f (%.0fs)", f, L,
                    {k: round(v, 3) for k, v in inner.items()}, a, time.time() - t)

    # ---- fixed-layer sweep on the same outer splits ----
    fixed = {}
    fixed_oof = {}
    if not args.skip_fixed_sweep:
        for L in layers:
            o = np.full(len(y), np.nan)
            for tr, te in outer_splits:
                o[te] = _fit_score(X[L][tr], y[tr], X[L][te], args.C)
            fixed_oof[L] = o
            fixed[L] = {"pooled_auroc": pooled_auroc(y, o),
                        "auprc": float(average_precision_score(y, o))}
            logger.info("fixed layer %d: pooled AUROC %.4f", L, fixed[L]["pooled_auroc"])

    # ---- final probes on all data ----
    probes_dir = art / "probes"
    if not args.skip_final_probes:
        probes_dir.mkdir(parents=True, exist_ok=True)
        for L in layers:
            clf = LogisticRegression(C=args.C, max_iter=1000, class_weight="balanced")
            clf.fit(X[L], y)
            o = fixed_oof.get(L)
            if o is not None:
                met = ProbeMetrics(
                    layer=L, n_train=int(len(y)), n_test=int(len(y)),
                    n_test_errors=int(y.sum()), auroc=pooled_auroc(y, o),
                    auprc=float(average_precision_score(y, o)),
                    brier=float(brier_score_loss(y, o)),
                    threshold_at_50pct_recall=threshold_at_recall(y, o),
                    accuracy_at_default_threshold=float(((o >= 0.5) == y).mean()),
                    per_fold_auroc=[pooled_auroc(y[te], o[te]) for _, te in outer_splits])
            else:
                met = ProbeMetrics(L, int(len(y)), 0, 0, float("nan"), float("nan"),
                                   float("nan"), None, float("nan"))
            probe = LinearProbe(layer=L, weights=clf.coef_[0].astype(np.float32),
                                bias=float(clf.intercept_[0]), classes=clf.classes_,
                                n_train=int(len(y)), metrics=met,
                                config={"C": args.C, "class_weight": "balanced",
                                        "source": "05c_nested_groupkfold",
                                        "metrics_are": "grouped 5-fold OOF"})
            with (probes_dir / f"probe_layer{L}.pkl").open("wb") as fh:
                pickle.dump(probe, fh)
        logger.info("final probes saved to %s", probes_dir)

    # ---- report ----
    fold_aurocs = [r["test_auroc"] for r in fold_rows]
    head = pooled_auroc(y, oof)
    auprc = float(average_precision_score(y, oof))
    report("")
    report("=" * 72)
    report("NESTED GROUPED K-FOLD RESULTS")
    report("=" * 72)
    for r in fold_rows:
        report(f"  outer {r['fold']}: layer {r['layer']}  test AUROC {r['test_auroc']:.4f}  "
               f"(n={r['n_test']}, err={r['n_test_err']})")
    if fixed:
        report("  fixed-layer pooled AUROC (same outer splits; max is optimistic):")
        for L in layers:
            report(f"    layer {L}: AUROC {fixed[L]['pooled_auroc']:.4f}  "
                   f"AUPRC {fixed[L]['auprc']:.4f}")
    report(f"  per-fold AUROC   = {np.mean(fold_aurocs):.4f} +/- {np.std(fold_aurocs):.4f}")
    report(f"  error base rate  = {y.mean():.4f} (AUPRC of a random scorer)")
    report(f"  layers selected  = {dict(Counter(r['layer'] for r in fold_rows))}")
    for k, name in ((0, "scalar"), (1, "list")):
        m = kind == k
        if m.any():
            report(f"  {name:6s} fields    : n={int(m.sum())} err={int(y[m].sum())} "
                   f"AUROC {pooled_auroc(y[m], oof[m]):.4f}")
    report(f"  pooled-OOF AUPRC = {auprc:.4f}")
    report(f"  pooled-OOF AUROC = {head:.4f}   <- headline")
    report(f"  runtime          = {(time.time() - t0) / 60:.1f} min")

    res_dir = art / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / args.out_name).write_text(json.dumps({
        "pooled_oof_auroc": head, "auprc": auprc,
        "auroc_mean": float(np.mean(fold_aurocs)), "auroc_std": float(np.std(fold_aurocs)),
        "n_fields": int(len(y)), "n_records": int(len(set(groups))),
        "n_errors": int(y.sum()), "outer_folds": fold_rows,
        "fixed_layer": {int(k): v for k, v in fixed.items()},
        "candidate_layers": layers, "C": args.C, "seed": seed,
        "list_agg": args.list_agg, "n_list_fields": int((kind == 1).sum()),
    }, indent=2))
    np.savez_compressed(
        res_dir / "oof_scores.npz",
        doc_id=groups, path_str=paths, y=y, oof=oof,
        selected_layer=sel_layer, outer_fold=fold_id, kind=kind,
        **{f"fixed_oof_layer{L}": o for L, o in fixed_oof.items()})
    report(f"  saved {res_dir / args.out_name} and {res_dir / 'oof_scores.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
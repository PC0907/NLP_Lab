#!/usr/bin/env python3
"""Nested leave-one-document-out CV: unbiased layer selection.

PROBLEM this fixes:
  The standard pipeline runs LODO at every layer, then reports the BEST layer's
  score (e.g. "layer 18: 0.849"). But the layer was CHOSEN by looking at the
  same held-out documents it is then reported on -> test-set peeking. Because
  neighbouring layers' scores have random variation, taking the max over layers
  optimistically inflates the reported number.

THE FIX (nested LODO):
  Outer loop: hold out document D_test (reported on, never used for selection).
  Inner loop: among the remaining docs, run LODO over candidate layers and pick
              the layer with the best INNER score (D_test is not involved).
  Then train that selected layer on ALL non-D_test docs and score D_test.
  Repeat over every D_test. The reported AUROC uses, for each test doc, a layer
  chosen WITHOUT seeing that doc -> unbiased.

We also log which layer was selected per outer fold, so you can see how stable
the choice is (with few docs it will vary -- that variation is exactly the bias
the naive "best layer" number was hiding).

DOMAIN FILTERING (new):
  Mirrors 05_lodo_cv.py. The labels dir may hold several domains; some (e.g.
  finance/10kq) carry benchmark-annotation-convention "errors" rather than
  genuine model errors and must be EXCLUDED from a clean cross-model comparison.
  Filtering by config alone does nothing because the loader globs every label on
  disk. Use --include-domains / --exclude-domains to filter on each label file's
  top-level "domain" key. No flags = all domains (original behaviour preserved).

Reuses existing per-field activations (works on last-token activations; this is
about LAYER selection, not aggregation). No GPU.

Usage:
    # all domains (unchanged behaviour)
    python scripts/05b_nested_lodo.py --config configs/exp_qwen35_4b_pooled.yaml

    # clean cross-model set: drop the 10kq convention-noise
    python scripts/05b_nested_lodo.py --config configs/exp_qwen35_2b_pooled.yaml \\
        --exclude-domains finance/10kq --out-name nested_lodo_nofin.json
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from probe_extraction.config import load_config
from probe_extraction.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Nested LODO (unbiased layer selection)")
    p.add_argument("--config", required=True)
    p.add_argument("--layers", type=int, nargs="*", default=None,
                   help="Candidate layers (default: all in config).")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--include-domains", type=str, nargs="*", default=None,
                   help="If set, ONLY load docs whose label-file 'domain' is in this list.")
    p.add_argument("--exclude-domains", type=str, nargs="*", default=None,
                   help="If set, SKIP docs whose label-file 'domain' is in this list. "
                        "Applied after --include-domains.")
    p.add_argument("--include-docs-file", type=str, default=None,
                   help="Path to a file with one doc_id per line. If set, ONLY these "
                        "docs are used (the cross-model intersection). Applied on top "
                        "of domain filters.")
    p.add_argument("--out-name", type=str, default="nested_lodo.json",
                   help="Filename for the results JSON (so a filtered run doesn't "
                        "overwrite the all-domain run).")
    p.add_argument("--exclude-gold-empty", action="store_true",
                   help="Skip fields whose gold is empty at that path "
                        "(error_type == 'hallucination', matcher.py:65). "
                        "Sensitivity analysis only: removes positives by "
                        "construction.")
    return p.parse_args()


def _domain_allowed(domain, include, exclude) -> bool:
    if include is not None and domain not in include:
        return False
    if exclude is not None and domain in exclude:
        return False
    return True


def load_layer_matrix(labels_dir: Path, activations_dir: Path, layers,
                      include=None, exclude=None, include_docs=None,
                      exclude_gold_empty=False):
    """Return per-layer X dict, y, doc_ids — aligned across layers.

    Only keeps fields that have activations for ALL candidate layers, so the
    layer comparison is on an identical field set. Documents whose 'domain' is
    excluded by include/exclude are skipped. If include_docs is given (a set of
    doc_ids), ONLY those docs are kept — used for the cross-model intersection
    so 2B/4B/9B are compared on an identical document set.
    """
    rows = []  # each: (doc_id, y, {layer: vec})
    skipped_domains = {}
    n_gold_empty_skipped = 0
    for lp in sorted(labels_dir.glob("*.json")):
        if lp.name.startswith("_"):
            continue
        doc_id = lp.stem

        # --- doc intersection filter (NEW) ---------------------------------
        if include_docs is not None and doc_id not in include_docs:
            continue
        # -------------------------------------------------------------------

        data = json.load(lp.open())

        # --- domain filter -------------------------------------------------
        domain = data.get("domain")
        if not _domain_allowed(domain, include, exclude):
            skipped_domains[domain] = skipped_domains.get(domain, 0) + 1
            continue
        # -------------------------------------------------------------------

        npz = activations_dir / f"{doc_id}.npz"
        if not npz.exists():
            continue
        with np.load(npz) as acts:
            for fld in data.get("labels", []):
                if not fld.get("extracted_present", True):
                    continue
                # Gold-empty: the model filled a path the annotator left empty.
                # matcher.py:65 calls these "hallucination", but here most are
                # annotation omissions with the value present in the document.
                if exclude_gold_empty and fld.get("error_type") == "hallucination":
                    n_gold_empty_skipped += 1
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
                    if v.ndim > 1:           # all_tokens stored: reduce to last
                        v = v[-1]
                    vecs[L] = v
                if not ok:
                    continue
                rows.append((doc_id, int(fld.get("is_error", 0)), vecs, ps))

    if skipped_domains:
        logger.info("Domain filter skipped: %s",
                    ", ".join(f"{d}={n}" for d, n in sorted(skipped_domains.items())))
    if exclude_gold_empty:
        logger.info("Gold-empty fields skipped: %d", n_gold_empty_skipped)

    doc_ids = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    X = {L: np.stack([r[2][L] for r in rows]) for L in layers}
    paths = np.array([r[3] for r in rows])
    return X, y, doc_ids, paths


def _fit_score(Xtr, ytr, Xte, C):
    clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def inner_select_layer(X, y, doc_ids, train_docs, layers, C):
    """Among train_docs, run LODO per layer; return the layer with best mean
    inner AUROC. D_test is excluded entirely (not in train_docs)."""
    best_layer, best_auroc = None, -1.0
    for L in layers:
        XL = X[L]
        fold_aurocs = []
        for vd in train_docs:
            inner_te = (doc_ids == vd)
            inner_tr = np.isin(doc_ids, train_docs) & ~inner_te
            ytr, yte = y[inner_tr], y[inner_te]
            if ytr.sum() == 0 or ytr.sum() == len(ytr):
                continue
            if yte.sum() == 0 or yte.sum() == len(yte):
                continue
            s = _fit_score(XL[inner_tr], ytr, XL[inner_te], C)
            fold_aurocs.append(roc_auc_score(yte, s))
        if fold_aurocs:
            m = float(np.mean(fold_aurocs))
            if m > best_auroc:
                best_auroc, best_layer = m, L
    return best_layer, best_auroc


def nested_lodo(X, y, doc_ids, layers, C):
    docs = list(dict.fromkeys(doc_ids.tolist()))
    outer_scores = []
    selected = []
    oof = np.full(len(y), np.nan)

    for d_test in docs:
        test_mask = (doc_ids == d_test)
        train_docs = [d for d in docs if d != d_test]
        if y[test_mask].sum() in (0, int(test_mask.sum())):
            # test doc has no positive/negative variety -> AUROC undefined
            continue

        # inner: pick layer using only train_docs
        L_sel, inner_auroc = inner_select_layer(X, y, doc_ids, train_docs, layers, C)
        if L_sel is None:
            continue
        selected.append(L_sel)

        # retrain selected layer on ALL train docs, score the held-out test doc
        tr = np.isin(doc_ids, train_docs)
        s = _fit_score(X[L_sel][tr], y[tr], X[L_sel][test_mask], C)
        oof[test_mask] = s
        outer_scores.append(roc_auc_score(y[test_mask], s))
        logger.info("  outer fold test=%s -> selected layer %d (inner AUROC %.3f), "
                    "test AUROC %.3f", d_test, L_sel, inner_auroc, outer_scores[-1])

    return outer_scores, selected, oof

def field_types(doc_ids, paths, level):
    """Field type per row. Domain comes from the doc_id prefix so types never
    merge across domains. Numeric list indices are stripped, so authors.3.name
    and authors.7.name are one type.
      level="path": full path without indices, e.g. balance_sheet.cash.value
      level="leaf": last component only,      e.g. value
    """
    out = []
    for d, p in zip(doc_ids, paths):
        dom = "__".join(str(d).split("__")[:2])
        parts = [x for x in str(p).split(".") if not x.lstrip("-").isdigit()]
        key = ".".join(parts) if level == "path" else (parts[-1] if parts else "")
        out.append(f"{dom}|{key}")
    return np.array(out)


def field_type_baseline(types, y, doc_ids, valid, C):
    """LODO baseline that sees ONLY the field type, never the activations.
    Scored on exactly the probe's held-out fields (`valid`), so both pooled
    AUROCs cover identical fields. Documents the probe skipped are skipped
    here too, because their fields are not in `valid`."""
    from sklearn.preprocessing import OneHotEncoder
    oof = np.full(len(y), np.nan)
    for d_test in dict.fromkeys(doc_ids.tolist()):
        te = (doc_ids == d_test) & valid
        if not te.any():
            continue
        tr = doc_ids != d_test
        if y[tr].sum() in (0, int(tr.sum())):
            continue
        enc = OneHotEncoder(handle_unknown="ignore")
        Xtr = enc.fit_transform(types[tr].reshape(-1, 1))
        Xte = enc.transform(types[te].reshape(-1, 1))
        oof[te] = _fit_score(Xtr, y[tr], Xte, C)
    m = valid & ~np.isnan(oof)
    return roc_auc_score(y[m], oof[m]), int(m.sum())


def within_type_auroc(types, y, oof, valid):
    """AUROC inside each field type, then averaged. A field-type baseline
    scores exactly 0.5 here by construction, so this isolates what the probe
    detects beyond field identity. Types with no label variation, such as a
    sub-field that is wrong every time, drop out automatically."""
    res, covered = [], 0
    for t in np.unique(types[valid]):
        m = valid & (types == t)
        if y[m].sum() in (0, int(m.sum())):
            continue
        res.append((roc_auc_score(y[m], oof[m]), int(m.sum())))
        covered += int(m.sum())
    if not res:
        return float("nan"), float("nan"), 0, 0
    a = np.array([r[0] for r in res]); w = np.array([r[1] for r in res])
    return float(a.mean()), float((a * w).sum() / w.sum()), len(res), covered


def main():
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_dir=cfg.logging.log_dir,
                  log_name="05b_nested_lodo", log_to_file=cfg.logging.log_to_file)

    layers = args.layers or cfg.activations.layers
    labels_dir = cfg.artifacts_path / "labels"
    activations_dir = cfg.artifacts_path / "activations"

    if args.include_domains is not None:
        logger.info("INCLUDE domains: %s", args.include_domains)
    if args.exclude_domains is not None:
        logger.info("EXCLUDE domains: %s", args.exclude_domains)

    include_docs = None
    if args.include_docs_file is not None:
        include_docs = {
            line.strip() for line in Path(args.include_docs_file).read_text().splitlines()
            if line.strip()
        }
        logger.info("INCLUDE docs: %d doc_ids from %s (cross-model intersection)",
                    len(include_docs), args.include_docs_file)

    logger.info("Loading activations for layers %s ...", layers)
    X, y, doc_ids, paths = load_layer_matrix(labels_dir, activations_dir, layers,
                                      include=args.include_domains,
                                      exclude=args.exclude_domains,
                                      include_docs=include_docs,
                                      exclude_gold_empty=args.exclude_gold_empty)
    logger.info("Loaded %d fields across %d docs (%d errors, %.1f%%).",
                len(y), len(set(doc_ids)), int(y.sum()), 100 * y.mean())

    logger.info("Running nested LODO (unbiased layer selection):")
    outer_scores, selected, oof = nested_lodo(X, y, doc_ids, layers, args.C)

    if not outer_scores:
        logger.error("No valid outer folds.")
        return 1

    mean_auroc = float(np.mean(outer_scores))
    std_auroc = float(np.std(outer_scores))
    valid = ~np.isnan(oof)
    # Pooled out-of-fold metrics: every field was scored by a probe that never
    # saw its document (so still leakage-free), but the metric is computed ONCE
    # over all held-out fields rather than averaged per-document. This avoids the
    # small-fold saturation (tiny credit docs hitting AUROC 1.000) that inflates
    # and widens the per-fold mean. This is the stable "all fields" number.
    pooled_ok = y[valid].sum() not in (0, valid.sum())
    pooled_auroc = (roc_auc_score(y[valid], oof[valid]) if pooled_ok else float("nan"))
    auprc = (average_precision_score(y[valid], oof[valid]) if pooled_ok else float("nan"))

    # ---- field-type controls ----------------------------------------------
    # Where error rates differ sharply by field type (10kq segment_name is
    # wrong every time), a probe can score well by recognising the field.
    controls = {}
    for level in ("path", "leaf"):
        types = field_types(doc_ids, paths, level)
        b_auroc, b_n = field_type_baseline(types, y, doc_ids, valid, args.C)
        wt_mean, wt_w, wt_k, wt_cov = within_type_auroc(types, y, oof, valid)
        controls[level] = {
            "n_types": int(len(np.unique(types[valid]))),
            "baseline_pooled_auroc": b_auroc, "baseline_n_fields": b_n,
            "within_type_auroc_mean": wt_mean,
            "within_type_auroc_weighted": wt_w,
            "within_type_n_types": wt_k,
            "within_type_fields_covered": wt_cov,
        }

    logger.info("=" * 64)
    logger.info("NESTED LODO (unbiased):")
    logger.info("  per-fold AUROC   = %.4f ± %.4f over %d outer folds (avg of per-doc AUROCs)",
                mean_auroc, std_auroc, len(outer_scores))
    logger.info("  pooled-OOF AUROC = %.4f  (one AUROC over all %d held-out fields; stable)",
                pooled_auroc, int(valid.sum()))
    logger.info("  pooled-OOF AUPRC = %.4f  (precision-recall; governs regeneration budget)", auprc)
    logger.info("  layers selected across folds: %s", dict(Counter(selected)))
    logger.info("  scored set: %d fields, %d errors", int(valid.sum()), int(y[valid].sum()))
    logger.info("-" * 64)
    logger.info("FIELD-TYPE CONTROLS (same held-out fields as the probe):")
    for level, c in controls.items():
        logger.info("  [%s] %d types | field-type-only baseline pooled AUROC %.4f "
                    "over %d fields", level, c["n_types"],
                    c["baseline_pooled_auroc"], c["baseline_n_fields"])
        logger.info("  [%s] probe within-type AUROC: mean %.4f, weighted %.4f, "
                    "%d types covering %d of %d fields", level,
                    c["within_type_auroc_mean"], c["within_type_auroc_weighted"],
                    c["within_type_n_types"], c["within_type_fields_covered"],
                    int(valid.sum()))
    logger.info("-" * 64)
    logger.info("Compare to the NAIVE best-layer LODO (which peeks at the test")
    logger.info("set to choose the layer). The nested number is the honest one;")
    logger.info("expect it at or slightly below the naive best-layer score.")

    out = cfg.artifacts_path / "results" / args.out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "auroc_mean": mean_auroc, "auroc_std": std_auroc,
        "pooled_oof_auroc": pooled_auroc,
        "auprc": auprc, "n_folds": len(outer_scores),
        "layers_selected": dict(Counter(selected)),
        "candidate_layers": layers,
        "include_domains": args.include_domains,
        "exclude_domains": args.exclude_domains,
        "n_fields_loaded": int(len(y)),
        "n_errors_loaded": int(y.sum()),
        "n_fields_scored": int(valid.sum()),
        "n_errors_scored": int(y[valid].sum()),
        "controls": controls,
    }, indent=2))
    np.savez(out.with_suffix(".oof.npz"), oof=oof, y=y, doc_ids=doc_ids, paths=paths)
    logger.info("Saved to %s (+ per-field OOF predictions)", out)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
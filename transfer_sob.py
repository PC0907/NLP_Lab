#!/usr/bin/env python3
"""Cross-task probe transfer: ExtractBench <-> SOB, zero-shot.

Both datasets were extracted by the same model, Qwen3.5-4B (hidden 2560,
32 layers), through the same pipeline and labeled by the same matcher. A
probe is a weight vector in that model's residual stream, so a probe trained
on one dataset scores the other directly, with no retraining.

  EB -> SOB : train on ExtractBench (document extraction), score SOB
              (multi-hop QA). Does the trust signal cross task types?
  SOB -> EB : the reverse.

PROTOCOL
  The layer is selected on the SOURCE only, by cross-validation inside the
  source. The target is never consulted, so the reported transfer AUROC does
  not peek. (The older ExtractBench <-> insurance transfer read its layers off
  a sweep over the target. This does not.) A per-layer sweep over the target
  is also reported, for the shape of the curve only. Its maximum is not a
  headline.

  Source selection mirrors 05b: mean of per-fold AUROCs, folds with a single
  class skipped. Leave-one-document-out when the source has at most
  --max-logo documents (ExtractBench); 5-fold GroupKFold by document
  otherwise (SOB, 997 documents), matching 05c.

  The final probe is trained on ALL source fields at the selected layer and
  scores ALL target fields. AUROC is threshold-free, so no recalibration.

  Fields come from 05b's own load_layer_matrix: emitted fields with
  activations at every candidate layer, all-tokens storage reduced to the
  last token. So the field set matches nested LODO exactly.

WHY THIS ALSO TESTS THE FIELD-TYPE CONFOUND
  The two datasets share no field types. A probe that only recognised which
  field it was looking at would carry nothing across. Transfer clearly above
  0.5 is evidence the probe reads something beyond field identity. Within-type
  AUROC on the target is reported as well.

NORMALISATION
  raw (primary: the ExtractBench <-> insurance transfer found raw beats L2)
  and L2-per-sample, applied identically on both sides.

Usage (from the repo root):
  python transfer_sob.py                              # ExtractBench excl. 10-K/Q
  python transfer_sob.py --eb-exclude-domains         # ExtractBench all four
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "scripts")
nl = import_module("05b_nested_lodo")          # reuse the pipeline's own code
from probe_extraction.config import load_config

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("transfer_sob")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eb-config", default="configs/exp_fresh_alltokens.yaml")
    p.add_argument("--sob-config", default="configs/exp_qwen35_4b_sob_1k.yaml")
    p.add_argument("--layers", type=int, nargs="+", default=[14, 16, 18, 20, 22])
    p.add_argument("--eb-exclude-domains", nargs="*", default=["finance/10kq"],
                   help="Give the flag with no values to include all four domains.")
    p.add_argument("--norms", nargs="+", default=["raw", "l2"], choices=["raw", "l2"])
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--max-logo", type=int, default=40,
                   help="Use leave-one-document-out for layer selection when the "
                        "source has at most this many documents.")
    p.add_argument("--out", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------

def load(cfg_path, layers, exclude):
    cfg = load_config(cfg_path)
    return nl.load_layer_matrix(
        cfg.artifacts_path / "labels", cfg.artifacts_path / "activations",
        layers, exclude=(exclude or None))


def l2(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)


def domain_prefix(doc_id):
    """ExtractBench ids look like academic__research__NAME. Anything without
    that shape (e.g. SOB record ids) gets no domain, so types and per-domain
    breakdowns never fragment by document."""
    parts = str(doc_id).split("__")
    return "__".join(parts[:2]) if len(parts) >= 3 else ""


def ftypes(doc_ids, paths, level):
    out = []
    for d, p in zip(doc_ids, paths):
        parts = [x for x in str(p).split(".") if not x.lstrip("-").isdigit()]
        key = ".".join(parts) if level == "path" else (parts[-1] if parts else "")
        out.append(f"{domain_prefix(d)}|{key}")
    return np.array(out)


def source_splits(y, groups, max_logo):
    idx = np.arange(len(y))
    ug = list(dict.fromkeys(groups.tolist()))
    if len(ug) <= max_logo:
        return ([(idx[groups != g], idx[groups == g]) for g in ug],
                f"leave-one-document-out, {len(ug)} folds")
    return (list(GroupKFold(n_splits=5).split(idx, y, groups)),
            "GroupKFold by document, 5 folds")


def select_layer(X, y, splits, layers, C):
    table = {}
    for L in layers:
        aucs = []
        for tr, te in splits:
            if y[tr].sum() in (0, len(tr)) or y[te].sum() in (0, len(te)):
                continue
            s = nl._fit_score(X[L][tr], y[tr], X[L][te], C)
            aucs.append(roc_auc_score(y[te], s))
        table[L] = (float(np.mean(aucs)), len(aucs)) if aucs else (float("nan"), 0)
    ok = [L for L in layers if not np.isnan(table[L][0])]
    return max(ok, key=lambda L: table[L][0]), table


def run_direction(src, tgt, layers, norm, C, max_logo):
    Xs, ys, ds, _ = src
    Xt, yt, dt, pt = tgt
    if norm == "l2":
        Xs = {L: l2(Xs[L]) for L in layers}
        Xt = {L: l2(Xt[L]) for L in layers}

    splits, scheme = source_splits(ys, ds, max_logo)
    L_sel, sel_table = select_layer(Xs, ys, splits, layers, C)

    sweep, s_sel = {}, None
    for L in layers:
        s = nl._fit_score(Xs[L], ys, Xt[L], C)
        sweep[L] = float(roc_auc_score(yt, s))
        if L == L_sel:
            s_sel = s

    valid = np.ones(len(yt), bool)
    within = {}
    for level in ("path", "leaf"):
        m, w, k, cov = nl.within_type_auroc(ftypes(dt, pt, level), yt, s_sel, valid)
        within[level] = {"mean": m, "weighted": w, "n_types": k, "fields_covered": cov}

    dom = np.array([domain_prefix(d) for d in dt])
    per_dom = {}
    if 0 < len(set(dom) - {""}) <= 20:
        for dn in sorted(set(dom) - {""}):
            m = dom == dn
            if yt[m].sum() in (0, int(m.sum())):
                continue
            per_dom[dn] = {"n_fields": int(m.sum()), "n_errors": int(yt[m].sum()),
                           "auroc": float(roc_auc_score(yt[m], s_sel[m]))}

    return {
        "source_selection": scheme,
        "source_cv_auroc": {str(L): v[0] for L, v in sel_table.items()},
        "source_cv_folds": {str(L): v[1] for L, v in sel_table.items()},
        "selected_layer": int(L_sel),
        "transfer_auroc": sweep[L_sel],
        "transfer_auprc": float(average_precision_score(yt, s_sel)),
        "sweep_auroc_shape_only": {str(L): v for L, v in sweep.items()},
        "within_type": within,
        "per_domain": per_dom,
    }, s_sel


def main():
    a = parse_args()
    excl = a.eb_exclude_domains
    tag = ("excl10kq" if excl == ["finance/10kq"] else "all4" if not excl else "custom")

    logger.info("Loading ExtractBench from %s (exclude: %s) ...", a.eb_config, excl or "none")
    eb = load(a.eb_config, a.layers, excl)
    logger.info("Loading SOB from %s ...", a.sob_config)
    sob = load(a.sob_config, a.layers, None)

    for nm, (X, y, d, _) in (("ExtractBench", eb), ("SOB", sob)):
        logger.info("  %-12s %5d fields, %4d docs, %4d errors (%.1f%%)",
                    nm, len(y), len(set(d.tolist())), int(y.sum()), 100 * y.mean())
    de, dsb = eb[0][a.layers[0]].shape[1], sob[0][a.layers[0]].shape[1]
    if de != dsb:
        logger.error("Hidden dims differ (%d vs %d): these are not the same model.", de, dsb)
        return 1

    out_dir = Path(a.out).parent if a.out else Path(
        "artifacts/fresh_qwen35_4b_pooled_alltokens/results")
    out = Path(a.out) if a.out else out_dir / f"transfer_sob_{tag}.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    results, arrays = {}, {}
    for norm in a.norms:
        for name, src, tgt in (("EB->SOB", eb, sob), ("SOB->EB", sob, eb)):
            logger.info("=" * 70)
            logger.info("%s   [%s]", name, norm)
            r, s = run_direction(src, tgt, a.layers, norm, a.C, a.max_logo)
            results[f"{name}|{norm}"] = r
            arrays[f"{name.replace('->', '_to_')}_{norm}"] = s

            logger.info("  source layer selection: %s", r["source_selection"])
            logger.info("  source CV AUROC by layer: %s",
                        ", ".join(f"{L}:{v:.4f}" for L, v in r["source_cv_auroc"].items()))
            logger.info("  SELECTED layer %d  ->  target AUROC %.4f, AUPRC %.4f",
                        r["selected_layer"], r["transfer_auroc"], r["transfer_auprc"])
            logger.info("  target sweep (shape only, not a headline): %s",
                        ", ".join(f"{L}:{v:.4f}" for L, v in r["sweep_auroc_shape_only"].items()))
            for lv, w in r["within_type"].items():
                logger.info("  within-type [%s]: mean %.4f, weighted %.4f, "
                            "%d types covering %d fields",
                            lv, w["mean"], w["weighted"], w["n_types"], w["fields_covered"])
            for dn, v in r["per_domain"].items():
                logger.info("  target domain %-26s %5d fields %4d errors  AUROC %.4f",
                            dn, v["n_fields"], v["n_errors"], v["auroc"])

    logger.info("=" * 70)
    logger.info("READING")
    logger.info("  Compare each transfer AUROC with the TARGET's own within-dataset")
    logger.info("  nested result. Transfer near that figure means the signal is shared")
    logger.info("  across task types. Transfer near 0.5 means it is not.")
    logger.info("  The two datasets share no field types, so field identity cannot carry")
    logger.info("  across. Transfer well above 0.5 is evidence against the field-type")
    logger.info("  explanation.")

    out.write_text(json.dumps({
        "eb_config": a.eb_config, "sob_config": a.sob_config,
        "eb_exclude_domains": excl, "candidate_layers": a.layers,
        "eb_counts": {"fields": int(len(eb[1])), "errors": int(eb[1].sum()),
                      "docs": len(set(eb[2].tolist()))},
        "sob_counts": {"fields": int(len(sob[1])), "errors": int(sob[1].sum()),
                       "docs": len(set(sob[2].tolist()))},
        "results": results,
    }, indent=2))
    np.savez(out.with_suffix(".scores.npz"), **arrays,
             eb_y=eb[1], eb_doc_ids=eb[2], eb_paths=eb[3],
             sob_y=sob[1], sob_doc_ids=sob[2], sob_paths=sob[3])
    logger.info("Saved to %s (+ per-field target scores)", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
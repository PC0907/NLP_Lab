#!/usr/bin/env python3
"""Cross-dataset transfer: train probe on ExtractBench, apply to insurance.

Answers the STRONG generalization question: does a probe trained ENTIRELY on
ExtractBench (financial/academic/swimming docs) detect errors on INSURANCE
documents it never saw -- with NO retraining?

  - Retrained-on-insurance (prior result): 0.863 pooled-OOF AUROC. That shows
    the METHOD generalizes.
  - THIS experiment: train on ExtractBench, test on insurance, zero-shot. That
    would show the TRUST SIGNAL ITSELF is dataset-independent -- a stronger claim.

Self-contained so preprocessing is applied IDENTICALLY to train and test (the
thing that makes a transfer number valid). Two normalization variants reported:
  - L2-per-sample : each field vector unit-normalized independently (no dataset
                    stats crossing over -> the safe/primary transfer number).
  - raw           : no normalization (sanity check; sensitive to scale shift).

Both datasets scored at the SAME layer (must match). CPU only.

Usage (as a job):
  python transfer_probe.py \\
     --train-config configs/exp_qwen35_4b_pooled_alltokens.yaml \\
     --test-config  configs/exp_qwen35_4b_insurance.yaml \\
     --layer 16 --train-exclude-domains finance/10kq
"""
from __future__ import annotations
import argparse, json, logging
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import normalize as l2normalize
from probe_extraction.config import load_config
from probe_extraction.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-config", required=True)
    p.add_argument("--test-config", required=True)
    p.add_argument("--layer", type=int, required=True,
                   help="Layer index used for BOTH train and test (must match).")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--train-include-domains", nargs="*", default=None)
    p.add_argument("--train-exclude-domains", nargs="*", default=None)
    p.add_argument("--test-include-domains", nargs="*", default=None)
    p.add_argument("--test-exclude-domains", nargs="*", default=None)
    p.add_argument("--out", default="transfer_extractbench_to_insurance.json")
    return p.parse_args()


def _domain_ok(domain, include, exclude):
    if include is not None and domain not in include:
        return False
    if exclude is not None and domain in exclude:
        return False
    return True


def load_layer(cfg, layer, include, exclude):
    """Load raw (X, y) at one layer for all fields, applying domain filter and
    the extracted_present filter (matching the pipeline)."""
    labels_dir = cfg.artifacts_path / "labels"
    acts_dir = cfg.artifacts_path / "activations"
    Xs, ys = [], []
    for lp in sorted(labels_dir.glob("*.json")):
        if lp.name.startswith("_"):
            continue
        data = json.load(lp.open())
        if not _domain_ok(data.get("domain"), include, exclude):
            continue
        npz = acts_dir / f"{lp.stem}.npz"
        if not npz.exists():
            continue
        with np.load(npz) as acts:
            for fld in data.get("labels", []):
                if not fld.get("extracted_present", True):
                    continue
                key = f"{fld['path_str']}__layer{layer}"
                if key not in acts:
                    continue
                v = acts[key].astype(np.float32)
                if v.ndim > 1:
                    v = v[-1]
                Xs.append(v)
                ys.append(int(fld.get("is_error", 0)))
    return np.stack(Xs), np.array(ys)


def evaluate(Xtr, ytr, Xte, yte, C, tag):
    clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    auroc = roc_auc_score(yte, p) if yte.sum() not in (0, len(yte)) else float("nan")
    auprc = average_precision_score(yte, p) if yte.sum() not in (0, len(yte)) else float("nan")
    logger.info("  [%s] transfer AUROC=%.4f  AUPRC=%.4f", tag, auroc, auprc)
    return auroc, auprc


def main():
    args = parse_args()
    cfg_tr = load_config(args.train_config)
    cfg_te = load_config(args.test_config)
    setup_logging(level="INFO", log_dir="logs", log_name="transfer_probe", log_to_file=True)

    L = args.layer
    logger.info("Loading TRAIN (ExtractBench) at layer %d ...", L)
    Xtr, ytr = load_layer(cfg_tr, L, args.train_include_domains, args.train_exclude_domains)
    logger.info("  train: %d fields, %d errors (%.1f%%)", len(ytr), int(ytr.sum()), 100*ytr.mean())

    logger.info("Loading TEST (insurance) at layer %d ...", L)
    Xte, yte = load_layer(cfg_te, L, args.test_include_domains, args.test_exclude_domains)
    logger.info("  test: %d fields, %d errors (%.1f%%)", len(yte), int(yte.sum()), 100*yte.mean())

    if Xtr.shape[1] != Xte.shape[1]:
        logger.error("Hidden-dim mismatch: train %d vs test %d. Same model required.",
                     Xtr.shape[1], Xte.shape[1])
        return 1

    logger.info("=" * 60)
    logger.info("CROSS-DATASET TRANSFER: ExtractBench -> insurance (layer %d)", L)
    logger.info("-" * 60)

    # variant 1: L2 per-sample (primary -- no dataset stats cross over)
    a1, p1 = evaluate(l2normalize(Xtr), ytr, l2normalize(Xte), yte, args.C, "L2-per-sample")
    # variant 2: raw (sanity)
    a2, p2 = evaluate(Xtr, ytr, Xte, yte, args.C, "raw")

    logger.info("-" * 60)
    logger.info("Compare to retrained-on-insurance baseline (~0.863 pooled-OOF).")
    logger.info("Transfer >> 0.5 = the trust signal is dataset-independent (strong).")
    logger.info("Transfer near retrained = remarkable. Below but >0.5 = real partial transfer.")
    logger.info("=" * 60)

    out = Path(cfg_te.artifacts_path) / "results" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "layer": L,
        "train_fields": int(len(ytr)), "train_error_rate": float(ytr.mean()),
        "test_fields": int(len(yte)), "test_error_rate": float(yte.mean()),
        "transfer_l2_auroc": a1, "transfer_l2_auprc": p1,
        "transfer_raw_auroc": a2, "transfer_raw_auprc": p2,
    }, indent=2))
    logger.info("Saved to %s", out)
    return 0


if __name__ == "__main__":
    import sys; sys.exit(main())
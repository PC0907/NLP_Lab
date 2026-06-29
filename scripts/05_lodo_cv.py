"""Leave-one-document-out cross-validation for probe quality.

Loads the activations and labels produced by Stages 1-2 of an experiment,
then trains the probe under leave-one-document-out (LODO) CV:
  - For each document, hold it out, train on the others' fields, test on it.
  - Aggregate per-fold AUROC.

This is methodologically stronger than random-fold CV (the default in
03_train_probe.py) because it directly tests document-level generalization.
If random-fold AUROC is much higher than LODO AUROC, the probe is partially
learning document-level features that don't generalize to new documents.

DOMAIN FILTERING (new):
  The labels directory may contain documents from several domains. Some domains
  (e.g. finance/10kq) have error labels dominated by a benchmark-annotation
  convention rather than genuine model errors, and must be EXCLUDED from a clean
  cross-model comparison. The original version of this script globbed EVERY
  label file in the dir and ignored domain entirely, so excluding a domain via a
  separate config did nothing (the labels were still on disk and were loaded).

  Use --include-domains and/or --exclude-domains to filter by the top-level
  "domain" key inside each label file. If neither flag is given, ALL domains are
  loaded (original behaviour preserved -> the 4B pipeline is unaffected).

Usage:
    # all domains (unchanged behaviour)
    python scripts/05_lodo_cv.py --config configs/exp_qwen35_4b_pymupdf.yaml

    # clean cross-model set: drop the 10kq convention-noise
    python scripts/05_lodo_cv.py --config configs/exp_qwen35_2b_pooled.yaml \\
        --exclude-domains finance/10kq

    # equivalently, name exactly what you want to keep
    python scripts/05_lodo_cv.py --config configs/exp_qwen35_2b_pooled.yaml \\
        --include-domains academic/research finance/credit_agreement sport/swimming
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from probe_extraction.config import load_config
from probe_extraction.utils.logging import setup_logging


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LODO cross-validation for probes")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--include-domains", type=str, nargs="*", default=None,
        help="If set, ONLY load docs whose label-file 'domain' is in this list.",
    )
    parser.add_argument(
        "--exclude-domains", type=str, nargs="*", default=None,
        help="If set, SKIP docs whose label-file 'domain' is in this list. "
             "Applied after --include-domains.",
    )
    parser.add_argument(
        "--out-name", type=str, default="lodo_cv.json",
        help="Filename for the results JSON (so a filtered run doesn't overwrite "
             "the all-domain run).",
    )
    return parser.parse_args()


def _domain_allowed(domain, include, exclude) -> bool:
    """Decide whether a document's domain passes the include/exclude filter."""
    if include is not None and domain not in include:
        return False
    if exclude is not None and domain in exclude:
        return False
    return True


def load_doc_data(activations_dir: Path, labels_dir: Path, layer: int,
                  include=None, exclude=None):
    """For each labelable document, load its activations at the given layer
    and its per-field labels. Returns a list of (doc_id, X, y) tuples.

    Filters out:
      - Documents whose 'domain' is excluded by include/exclude (NEW)
      - Fields marked is_empty (synthetic activation positions)
      - Documents with no labels or no activations
    """
    docs = []
    skipped_domains = {}  # domain -> count, for a transparency log
    for labels_path in sorted(labels_dir.glob("*.json")):
        if labels_path.name.startswith("_"):
            continue  # skip _summary.json
        doc_id = labels_path.stem

        with labels_path.open() as f:
            labels_data = json.load(f)

        # --- domain filter (NEW) -------------------------------------------
        domain = labels_data.get("domain")
        if not _domain_allowed(domain, include, exclude):
            skipped_domains[domain] = skipped_domains.get(domain, 0) + 1
            continue
        # -------------------------------------------------------------------

        # Each labels file is a list of {path, label, value, ...} under "labels".
        # The labeler already excludes is_empty fields. Each entry has path_str
        # and is_error.
        fields = labels_data.get("labels", [])
        if not fields:
            continue

        # Load activations
        act_path = activations_dir / f"{doc_id}.npz"
        if not act_path.exists():
            logger.warning("No activations for %s", doc_id)
            continue

        with np.load(act_path) as act_data:
            X_list = []
            y_list = []
            for field in fields:
                path_str = field["path_str"]
                label = int(field["is_error"])
                key = f"{path_str}__layer{layer}"
                if key not in act_data:
                    continue
                X_list.append(act_data[key])
                y_list.append(label)
        if not X_list:
            continue

        X = np.stack(X_list, axis=0).astype(np.float32)
        y = np.array(y_list, dtype=np.int64)
        docs.append((doc_id, X, y))

    if skipped_domains:
        logger.info("Domain filter skipped: %s",
                    ", ".join(f"{d}={n}" for d, n in sorted(skipped_domains.items())))
    return docs


def lodo_for_layer(docs, layer: int, C: float = 1.0) -> dict:
    """Run leave-one-document-out CV for one layer.

    Returns: dict with per-fold AUROCs, mean, std.
    """
    n_docs = len(docs)
    if n_docs < 2:
        return {"n_docs": n_docs, "error": "need at least 2 docs for LODO"}

    fold_aurocs = []
    fold_details = []

    for i in range(n_docs):
        test_doc_id, X_test, y_test = docs[i]
        X_train = np.concatenate([d[1] for j, d in enumerate(docs) if j != i], axis=0)
        y_train = np.concatenate([d[2] for j, d in enumerate(docs) if j != i], axis=0)

        # Sanity checks
        if y_train.sum() == 0 or y_train.sum() == len(y_train):
            fold_details.append({"test_doc": test_doc_id, "skipped": "train has one class"})
            continue
        if y_test.sum() == 0 or y_test.sum() == len(y_test):
            fold_details.append({"test_doc": test_doc_id, "skipped": "test has one class",
                                 "n_test": len(y_test), "n_test_errors": int(y_test.sum())})
            continue

        clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced")
        clf.fit(X_train, y_train)
        scores = clf.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, scores)
        fold_aurocs.append(auroc)
        fold_details.append({
            "test_doc": test_doc_id,
            "auroc": float(auroc),
            "n_test": int(len(y_test)),
            "n_test_errors": int(y_test.sum()),
        })

    if fold_aurocs:
        return {
            "layer": layer,
            "n_docs": n_docs,
            "n_valid_folds": len(fold_aurocs),
            "lodo_auroc_mean": float(np.mean(fold_aurocs)),
            "lodo_auroc_std": float(np.std(fold_aurocs)),
            "per_fold": fold_details,
        }
    return {
        "layer": layer,
        "n_docs": n_docs,
        "n_valid_folds": 0,
        "lodo_auroc_mean": None,
        "lodo_auroc_std": None,
        "per_fold": fold_details,
        "warning": "no folds had both classes present in test",
    }


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    setup_logging(level=cfg.logging.level, log_dir=cfg.logging.log_dir,
                  log_name="05_lodo_cv", log_to_file=cfg.logging.log_to_file)

    activations_dir = cfg.artifacts_path / "activations"
    labels_dir = cfg.artifacts_path / "labels"
    results_dir = cfg.artifacts_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("LODO CV for experiment: %s", cfg.experiment.name)
    logger.info("Activations dir: %s", activations_dir)
    logger.info("Labels dir: %s", labels_dir)
    if args.include_domains is not None:
        logger.info("INCLUDE domains: %s", args.include_domains)
    if args.exclude_domains is not None:
        logger.info("EXCLUDE domains: %s", args.exclude_domains)

    all_results = {}
    for layer in cfg.activations.layers:
        docs = load_doc_data(activations_dir, labels_dir, layer,
                             include=args.include_domains,
                             exclude=args.exclude_domains)
        logger.info("Layer %d: %d documents with usable data", layer, len(docs))

        if not docs:
            logger.warning("Layer %d: no usable documents", layer)
            continue

        result = lodo_for_layer(docs, layer, C=cfg.probe.C)
        all_results[str(layer)] = result

        if result.get("lodo_auroc_mean") is not None:
            logger.info(
                "Layer %d: LODO AUROC = %.4f ± %.4f (n_valid=%d/%d)",
                layer, result["lodo_auroc_mean"], result["lodo_auroc_std"],
                result["n_valid_folds"], result["n_docs"],
            )
        else:
            logger.warning("Layer %d: no valid LODO folds (%s)",
                           layer, result.get("warning", ""))

    out_path = results_dir / args.out_name
    with out_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved LODO results to %s", out_path)

    logger.info("=" * 70)
    logger.info("LODO CV SUMMARY:")
    for layer_str, res in all_results.items():
        if res.get("lodo_auroc_mean") is not None:
            logger.info(
                "  Layer %s: %.4f ± %.4f (n_valid_folds=%d)",
                layer_str, res["lodo_auroc_mean"], res["lodo_auroc_std"],
                res["n_valid_folds"],
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
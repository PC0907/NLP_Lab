"""Stage 7: Leave-one-document-out cross-validation for CLAP probe.

Mirrors Stage 5 (05_lodo_cv.py) exactly — same LODO protocol, same output
structure — but trains a CLAP probe instead of logistic regression per fold.

Why LODO matters
-----------------
Random-fold CV is optimistic because test fields from the same document as
training fields share document-level features (formatting, domain vocabulary).
LODO is the methodologically honest regime: train on 24 documents, test on 1.
If CLAP's LODO AUROC matches or exceeds the best per-layer LODO AUROC, it
means CLAP generalises across documents without requiring layer selection.

Comparison with Stage 5
-------------------------
Stage 5 output:  artifacts/{exp}/results/lodo_cv.json
    → per-layer LODO AUROCs (14 values, one per layer)

Stage 7 output:  artifacts/{exp}/results/lodo_clap.json
    → single CLAP LODO AUROC (uses all 14 layers jointly)

Both use the SAME 25-fold leave-one-document-out split.
The comparison is therefore direct and fair.

Speed note
-----------
We train CLAP 25 times (once per LODO fold). Each fit uses 30 epochs
instead of the 50 used in Stage 6. With 2000 fields on CPU, each fold
trains in ~5 seconds. Total: ~2 minutes.

Usage
------
    python scripts/07_lodo_clap.py --config configs/exp_deepseek_r1_7b_pooled.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from probe_extraction.config import load_config
from probe_extraction.probes.clap import fit_clap_for_lodo, predict_clap_proba
from probe_extraction.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LODO CV for CLAP probe.")
    p.add_argument("--config", required=True)
    return p.parse_args()


def load_docs_multilayer(
    activations_dir: Path,
    labels_dir: Path,
    layers: list[int],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Load per-document multi-layer activation matrices.

    Returns a list of (doc_id, X, y) where:
        X : (n_fields_in_doc, n_layers, hidden_dim)  float32
        y : (n_fields_in_doc,)                        int64

    Same filtering as Stage 5: skips synthetic fields and missing layers.
    Grouped by document so the LODO loop can hold each document out.
    """
    docs: list[tuple[str, np.ndarray, np.ndarray]] = []

    for label_path in sorted(labels_dir.glob("*.json")):
        if label_path.name.startswith("_"):
            continue

        with label_path.open(encoding="utf-8") as f:
            label_doc = json.load(f)

        doc_id = label_doc["doc_id"]
        act_path = activations_dir / f"{doc_id}.npz"
        if not act_path.exists():
            logger.warning("No activations for %s; skipping.", doc_id)
            continue

        X_list: list[np.ndarray] = []
        y_list: list[int] = []

        with np.load(act_path) as npz:
            available = set(npz.keys())

            for lab in label_doc.get("labels", []):
                if not lab.get("extracted_present", False):
                    continue

                path_str = lab["path_str"]
                keys = [f"{path_str}__layer{layer}" for layer in layers]
                if not all(k in available for k in keys):
                    continue

                vecs = np.stack(
                    [npz[k].astype(np.float32) for k in keys],
                    axis=0,
                )  # (n_layers, hidden_dim)
                X_list.append(vecs)
                y_list.append(int(lab["is_error"]))

        if not X_list:
            continue

        X = np.stack(X_list, axis=0)            # (n_fields, n_layers, hidden_dim)
        y = np.array(y_list, dtype=np.int64)
        docs.append((doc_id, X, y))

    logger.info(
        "Loaded %d documents with usable activation data.", len(docs)
    )
    return docs


def run_lodo_clap(
    docs: list[tuple[str, np.ndarray, np.ndarray]],
    arch_config: dict,
    *,
    lr: float,
    weight_decay: float,
    random_state: int,
) -> dict:
    """Run CLAP leave-one-document-out CV.

    For each document:
      1. Train CLAP on the other (n-1) documents' fields.
      2. Predict on the held-out document's fields.
      3. Compute AUROC.

    Returns a results dict compatible with the lodo_cv.json structure
    produced by Stage 5, so they can be compared side by side.
    """
    n_docs = len(docs)
    if n_docs < 2:
        return {"n_docs": n_docs, "error": "Need at least 2 documents for LODO."}

    n_layers   = docs[0][1].shape[1]
    hidden_dim = docs[0][1].shape[2]
    arch_config = dict(arch_config)   # copy; keys already set by caller

    fold_aurocs: list[float] = []
    fold_details: list[dict] = []

    for i, (test_doc_id, X_test, y_test) in enumerate(docs):
        # Concatenate all OTHER documents for training
        X_train = np.concatenate(
            [d[1] for j, d in enumerate(docs) if j != i], axis=0
        )  # (total_train_fields, n_layers, hidden_dim)
        y_train = np.concatenate(
            [d[2] for j, d in enumerate(docs) if j != i], axis=0
        )

        # Skip folds where train or test set has only one class
        if y_train.sum() in (0, len(y_train)):
            fold_details.append({
                "test_doc": test_doc_id,
                "skipped": "train set has only one class",
            })
            logger.debug("Fold %d (%s): skipped — train has one class.",
                         i, test_doc_id)
            continue

        if y_test.sum() in (0, len(y_test)):
            fold_details.append({
                "test_doc": test_doc_id,
                "skipped": "test set has only one class",
                "n_test": int(len(y_test)),
                "n_test_errors": int(y_test.sum()),
            })
            logger.debug("Fold %d (%s): skipped — test has one class.",
                         i, test_doc_id)
            continue

        # Train CLAP on training documents
        model = fit_clap_for_lodo(
            X_train=X_train,
            y_train=y_train.astype(np.int32),
            arch_config=arch_config,
            lr=lr,
            epochs=30,          # fewer than Stage 6; speed vs quality tradeoff
            weight_decay=weight_decay,
            random_state=random_state + i,
        )

        # Evaluate on held-out document
        proba = predict_clap_proba(model, X_test)
        auroc = float(roc_auc_score(y_test, proba))
        fold_aurocs.append(auroc)

        fold_details.append({
            "test_doc": test_doc_id,
            "auroc": auroc,
            "n_test": int(len(y_test)),
            "n_test_errors": int(y_test.sum()),
        })
        logger.info(
            "Fold %2d / %2d  doc=%-30s  AUROC=%.4f  "
            "(n_test=%d, n_errors=%d)",
            i + 1, n_docs, test_doc_id, auroc,
            len(y_test), int(y_test.sum()),
        )

    if fold_aurocs:
        return {
            "probe_type": "clap",
            "n_layers": n_layers,
            "n_docs": n_docs,
            "n_valid_folds": len(fold_aurocs),
            "lodo_auroc_mean": float(np.mean(fold_aurocs)),
            "lodo_auroc_std":  float(np.std(fold_aurocs)),
            "per_fold": fold_details,
        }

    return {
        "probe_type": "clap",
        "n_docs": n_docs,
        "n_valid_folds": 0,
        "lodo_auroc_mean": None,
        "lodo_auroc_std":  None,
        "per_fold": fold_details,
        "warning": "No valid LODO folds (some documents may have only one class).",
    }


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    setup_logging(
        level=cfg.logging.level,
        log_dir=cfg.logging.log_dir,
        log_name="07_lodo_clap",
        log_to_file=cfg.logging.log_to_file,
    )

    artifacts   = cfg.artifacts_path
    act_dir     = artifacts / "activations"
    labels_dir  = artifacts / "labels"
    results_dir = artifacts / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not labels_dir.exists():
        logger.error("labels/ not found. Run Stage 2 first.")
        return 1
    if not act_dir.exists():
        logger.error("activations/ not found. Run Stage 1 first.")
        return 1

    layers = cfg.activations.layers
    logger.info(
        "CLAP LODO | experiment=%s | n_layers=%d | d_model=%d | n_enc=%d",
        cfg.experiment.name, len(layers), cfg.clap.d_model, cfg.clap.n_enc,
    )

    # ── Build arch_config (mirrors train_clap internals) ──────────────────
    # We don't know hidden_dim until we load data, so it's set inside
    # run_lodo_clap after loading. We pass the other hyperparameters here.
    docs = load_docs_multilayer(act_dir, labels_dir, layers)
    if len(docs) < 2:
        logger.error("Need at least 2 documents; found %d.", len(docs))
        return 1

    # Read hidden_dim from first document's data
    _, X0, _ = docs[0]
    n_layers_actual, hidden_dim = X0.shape[1], X0.shape[2]

    arch_config = {
        "hidden_dim": hidden_dim,
        "n_layers":   n_layers_actual,
        "d_model":    cfg.clap.d_model,
        "n_enc":      cfg.clap.n_enc,
        "dropout":    cfg.clap.dropout,
    }
    logger.info(
        "Architecture: hidden_dim=%d, n_layers=%d, d_model=%d, n_enc=%d",
        hidden_dim, n_layers_actual, cfg.clap.d_model, cfg.clap.n_enc,
    )

    # ── Run LODO ──────────────────────────────────────────────────────────
    result = run_lodo_clap(
        docs,
        arch_config=arch_config,
        lr=cfg.clap.lr,
        weight_decay=cfg.clap.weight_decay,
        random_state=cfg.experiment.seed,
    )

    # ── Add metadata ──────────────────────────────────────────────────────
    result["model_name"]  = cfg.model.name
    result["experiment"]  = cfg.experiment.name
    result["layers"]      = layers
    result["d_model"]     = cfg.clap.d_model
    result["n_enc"]       = cfg.clap.n_enc
    result["hidden_dim"]  = hidden_dim

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = results_dir / "lodo_clap.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("LODO CLAP results saved → %s", out_path)

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("CLAP LODO CV SUMMARY:")
    if result.get("lodo_auroc_mean") is not None:
        logger.info(
            "  CLAP LODO AUROC: %.4f ± %.4f  (n_valid_folds=%d / %d)",
            result["lodo_auroc_mean"],
            result["lodo_auroc_std"],
            result["n_valid_folds"],
            result["n_docs"],
        )
        logger.info(
            "  Compare with Stage 5 per-layer LODO:"
            " cat artifacts/%s/results/lodo_cv.json",
            cfg.experiment.name,
        )
    else:
        logger.warning("No valid LODO folds. Check the per_fold details.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
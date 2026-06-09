"""Stage 6: Train CLAP cross-layer attention probe.

Loads the activations and labels produced by Stages 1–2 and trains a CLAP
probe that attends jointly over all captured layers — in contrast to Stage 3,
which trains one independent logistic regression per layer.

Key difference from Stage 3
------------------------------
Stage 3 (train_probe.py) loads ONE layer at a time:
    X shape: (n_fields, hidden_dim)
    Trains: 14 independent logistic regressions

Stage 6 (this script) loads ALL layers simultaneously:
    X shape: (n_fields, n_layers, hidden_dim)
    Trains: 1 transformer that attends over all 14 layers

Both use the same .npz activation files and labels/ JSON files.
No re-extraction needed.

Artifacts written
------------------
    artifacts/{exp}/probes/clap_probe.pt     — trained model weights + arch config
    artifacts/{exp}/results/clap_summary.json — CV AUROC, split AUROC, metadata

Usage
------
    python scripts/06_train_clap.py --config configs/exp_deepseek_r1_7b_pooled.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from probe_extraction.config import load_config
from probe_extraction.probes.clap import train_clap
from probe_extraction.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CLAP cross-layer probe.")
    p.add_argument("--config", required=True, help="Path to YAML config file.")
    return p.parse_args()


def load_multilayer_dataset(
    labels_dir: Path,
    activations_dir: Path,
    layers: list[int],
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Load all fields as multi-layer activation matrices.

    For each labelable field, stacks the activation vectors from all
    captured layers into shape (n_layers, hidden_dim). Returns:

        X    : (n_fields, n_layers, hidden_dim) float32
        y    : (n_fields,)                      int32   {0=correct, 1=error}
        meta : list of {'doc_id': str, 'path_str': str}

    Skips:
      - Fields without extracted_present=True (synthetic activations)
      - Fields missing any layer key in the .npz file
      - Documents with no matching activations file
    """
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    meta: list[dict] = []
    n_skip_synthetic = 0
    n_skip_missing_layer = 0

    for label_path in sorted(labels_dir.glob("*.json")):
        if label_path.name.startswith("_"):
            continue

        with label_path.open(encoding="utf-8") as f:
            label_doc = json.load(f)

        doc_id = label_doc["doc_id"]
        act_path = activations_dir / f"{doc_id}.npz"
        if not act_path.exists():
            logger.debug("No activations file for %s, skipping.", doc_id)
            continue

        with np.load(act_path) as npz:
            available = set(npz.keys())

            for lab in label_doc.get("labels", []):
                # Skip synthetic (field was absent in extracted JSON)
                if not lab.get("extracted_present", False):
                    n_skip_synthetic += 1
                    continue

                path_str = lab["path_str"]

                # Check every requested layer is present for this field
                keys = [f"{path_str}__layer{layer}" for layer in layers]
                if not all(k in available for k in keys):
                    n_skip_missing_layer += 1
                    continue

                # Stack: list of (hidden_dim,) → (n_layers, hidden_dim)
                vecs = np.stack(
                    [npz[k].astype(np.float32) for k in keys],
                    axis=0,
                )  # (n_layers, hidden_dim)

                X_list.append(vecs)
                y_list.append(int(lab["is_error"]))
                meta.append({"doc_id": doc_id, "path_str": path_str})

    if not X_list:
        logger.error(
            "No usable fields found. "
            "Check that Stage 1 ran with the same config and layers."
        )
        return (
            np.empty((0, len(layers), 1), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            [],
        )

    X = np.stack(X_list, axis=0)        # (n_fields, n_layers, hidden_dim)
    y = np.array(y_list, dtype=np.int32)

    logger.info(
        "Dataset: %d fields | %d errors (%.1f%%) | skipped: %d synthetic, "
        "%d missing-layer",
        len(y), int(y.sum()), 100.0 * y.mean(),
        n_skip_synthetic, n_skip_missing_layer,
    )
    return X, y, meta


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    setup_logging(
        level=cfg.logging.level,
        log_dir=cfg.logging.log_dir,
        log_name="06_train_clap",
        log_to_file=cfg.logging.log_to_file,
    )

    artifacts   = cfg.artifacts_path
    labels_dir  = artifacts / "labels"
    act_dir     = artifacts / "activations"
    probes_dir  = artifacts / "probes"
    results_dir = artifacts / "results"
    probes_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not labels_dir.exists():
        logger.error("labels/ dir not found at %s. Run Stage 2 first.", labels_dir)
        return 1
    if not act_dir.exists():
        logger.error("activations/ dir not found at %s. Run Stage 1 first.", act_dir)
        return 1

    layers = cfg.activations.layers
    logger.info("Experiment: %s | layers: %s", cfg.experiment.name, layers)

    # ── Load data ──────────────────────────────────────────────────────────
    X, y, meta = load_multilayer_dataset(labels_dir, act_dir, layers)
    if len(y) == 0:
        return 1

    n_errors = int(y.sum())
    logger.info(
        "Training CLAP: %d fields, %d errors, "
        "d_model=%d, n_enc=%d, epochs=%d",
        len(y), n_errors,
        cfg.clap.d_model, cfg.clap.n_enc, cfg.clap.epochs,
    )

    # ── Train ──────────────────────────────────────────────────────────────
    result = train_clap(
        X=X,
        y=y,
        layers=layers,
        d_model=cfg.clap.d_model,
        n_enc=cfg.clap.n_enc,
        dropout=cfg.clap.dropout,
        lr=cfg.clap.lr,
        epochs=cfg.clap.epochs,
        batch_size=cfg.clap.batch_size,
        weight_decay=cfg.clap.weight_decay,
        warmup_epochs=cfg.clap.warmup_epochs,
        patience=cfg.clap.patience,
        cv_folds=cfg.probe.cv_folds,
        test_size=cfg.probe.test_size,
        random_state=cfg.experiment.seed,
    )

    # ── Save model ─────────────────────────────────────────────────────────
    probe_path = probes_dir / "clap_probe.pt"
    torch.save(result, probe_path)
    logger.info("CLAP probe saved → %s", probe_path)

    # ── Save summary ───────────────────────────────────────────────────────
    summary = {
        "probe_type": "clap",
        "model_name": cfg.model.name,
        "experiment": cfg.experiment.name,
        "n_layers": len(layers),
        "layers": layers,
        "d_model": cfg.clap.d_model,
        "n_enc": cfg.clap.n_enc,
        "n_fields": int(len(y)),
        "n_errors": n_errors,
        "error_rate": float(y.mean()),
        "auroc_cv_mean": result.metrics.cv_auroc_mean,
        "auroc_cv_std":  result.metrics.cv_auroc_std,
        "auroc_cv_folds": result.metrics.per_fold_auroc,
        "auroc_split": result.metrics.auroc,
        "auprc_split": result.metrics.auprc,
        "n_train_split": result.metrics.n_train,
        "n_test_split":  result.metrics.n_test,
    }
    summary_path = results_dir / "clap_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── Headline log ───────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("CLAP RESULTS:")
    if result.metrics.cv_auroc_mean is not None:
        logger.info(
            "  CV AUROC  : %.4f ± %.4f (%d folds)",
            result.metrics.cv_auroc_mean,
            result.metrics.cv_auroc_std or 0.0,
            len(result.metrics.per_fold_auroc),
        )
    logger.info("  Split AUROC : %.4f", result.metrics.auroc)
    logger.info("  Split AUPRC : %.4f", result.metrics.auprc)
    logger.info("Summary saved → %s", summary_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
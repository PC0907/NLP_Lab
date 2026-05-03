"""Stage 3: Train linear probes on (activation, label) pairs.

For each captured layer, builds a probe to predict per-field is_error from
the field's hidden-state activation. Reports cross-validated AUROC and
related metrics. Saves trained probes to disk.

Filtering decisions made here:
  - Skip fields where the model emitted null/empty (extracted_present=False).
    These fields' activations are SYNTHETIC (extractor placed them at the
    cursor, not at a real value position) and so don't carry probe-relevant
    signal. Including them would train the probe on noise.
  - Skip fields with type_mismatch errors (we have 0 anyway).

Artifacts written:
  artifacts/probes/probe_layer{N}.pkl   — trained probe per layer
  artifacts/probes/_summary.json        — per-layer metrics

Usage:
    python scripts/03_train_probe.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from probe_extraction.config import load_config
from probe_extraction.probes import train_probe
from probe_extraction.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train linear probes from activations + labels.")
    p.add_argument("--config", required=True)
    return p.parse_args()


def load_dataset(
    extractions_dir: Path,
    activations_dir: Path,
    labels_dir: Path,
    layers: list[int],
) -> dict:
    """Load (X, y, meta) for each layer.

    Returns a dict mapping layer index → {"X": (n, d), "y": (n,), "meta": [...]}.

    Filtering:
      - Skip docs that failed extraction (no activations file).
      - Skip fields where extracted_present is False (synthetic activations).
      - Skip fields where activations file doesn't have the expected key.
    """
    per_layer: dict[int, dict] = {
        ℓ: {"X": [], "y": [], "meta": []} for ℓ in layers
    }

    n_docs = 0
    n_fields_total = 0
    n_fields_skipped_synthetic = 0
    n_fields_skipped_missing_activation = 0

    for label_path in sorted(labels_dir.glob("*.json")):
        if label_path.name.startswith("_"):
            continue

        with label_path.open("r", encoding="utf-8") as f:
            label_doc = json.load(f)

        doc_id = label_doc["doc_id"]
        activations_path = activations_dir / f"{doc_id}.npz"
        if not activations_path.exists():
            logger.warning("No activations file for %s; skipping doc.", doc_id)
            continue

        n_docs += 1
        with np.load(activations_path) as activations_npz:
            available_keys = set(activations_npz.keys())

            for lab in label_doc["labels"]:
                n_fields_total += 1

                if not lab["extracted_present"]:
                    # Synthetic activation; not informative for probing.
                    n_fields_skipped_synthetic += 1
                    continue

                path_str = lab["path_str"]
                is_error = int(lab["is_error"])

                # Pull this field's activation at each requested layer.
                missing_any = False
                for ℓ in layers:
                    key = f"{path_str}__layer{ℓ}"
                    if key not in available_keys:
                        missing_any = True
                        break

                if missing_any:
                    n_fields_skipped_missing_activation += 1
                    continue

                for ℓ in layers:
                    key = f"{path_str}__layer{ℓ}"
                    vec = activations_npz[key].astype(np.float32)
                    per_layer[ℓ]["X"].append(vec)
                    per_layer[ℓ]["y"].append(is_error)
                    per_layer[ℓ]["meta"].append({
                        "doc_id": doc_id,
                        "path_str": path_str,
                        "error_type": lab["error_type"],
                    })

    # Stack into arrays.
    for ℓ in layers:
        if per_layer[ℓ]["X"]:
            per_layer[ℓ]["X"] = np.stack(per_layer[ℓ]["X"], axis=0)
            per_layer[ℓ]["y"] = np.array(per_layer[ℓ]["y"], dtype=np.int32)
        else:
            per_layer[ℓ]["X"] = np.empty((0, 0), dtype=np.float32)
            per_layer[ℓ]["y"] = np.empty((0,), dtype=np.int32)

    logger.info(
        "Dataset loaded: %d docs, %d fields total, "
        "%d skipped (synthetic), %d skipped (missing activation), "
        "%d kept per layer.",
        n_docs, n_fields_total,
        n_fields_skipped_synthetic, n_fields_skipped_missing_activation,
        len(per_layer[layers[0]]["y"]) if layers else 0,
    )

    return per_layer


def write_summary(
    per_layer_results: dict,
    summary_path: Path,
    n_kept: int,
    n_errors: int,
) -> None:
    summary = {
        "n_kept_fields": int(n_kept),
        "n_errors_in_kept": int(n_errors),
        "error_rate_in_kept": float(n_errors / n_kept) if n_kept else 0.0,
        "per_layer": {
            str(ℓ): {
                "auroc": probe.metrics.auroc,
                "auprc": probe.metrics.auprc,
                "brier": probe.metrics.brier,
                "accuracy_at_default_threshold":
                    probe.metrics.accuracy_at_default_threshold,
                "threshold_at_50pct_recall":
                    probe.metrics.threshold_at_50pct_recall,
                "n_train": probe.metrics.n_train,
                "n_test": probe.metrics.n_test,
                "n_test_errors": probe.metrics.n_test_errors,
                "per_fold_auroc": probe.metrics.per_fold_auroc,
                "cv_auroc_mean": (
                    float(np.mean(probe.metrics.per_fold_auroc))
                    if probe.metrics.per_fold_auroc else None
                ),
                "cv_auroc_std": (
                    float(np.std(probe.metrics.per_fold_auroc))
                    if probe.metrics.per_fold_auroc else None
                ),
            }
            for ℓ, probe in per_layer_results.items()
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    setup_logging(
        level=cfg.logging.level,
        log_dir=cfg.logging.log_dir,
        log_name="03_train_probe",
        log_to_file=cfg.logging.log_to_file,
    )

    artifacts = cfg.artifacts_path
    extractions_dir = artifacts / "extractions"
    activations_dir = artifacts / "activations"
    labels_dir = artifacts / "labels"
    probes_dir = artifacts / "probes"
    probes_dir.mkdir(parents=True, exist_ok=True)

    layers = cfg.activations.layers

    # ------ Load dataset ------
    per_layer = load_dataset(
        extractions_dir=extractions_dir,
        activations_dir=activations_dir,
        labels_dir=labels_dir,
        layers=layers,
    )

    # Sanity check: same n_samples across layers.
    sizes = {ℓ: len(per_layer[ℓ]["y"]) for ℓ in layers}
    if len(set(sizes.values())) > 1:
        logger.error("Layer sample counts differ: %s", sizes)
        return 1

    n_kept = sizes[layers[0]] if layers else 0
    if n_kept == 0:
        logger.error("No usable fields after filtering. Aborting.")
        return 1

    n_errors = int(per_layer[layers[0]]["y"].sum())
    logger.info(
        "Training data: %d fields, %d errors (%.1f%%)",
        n_kept, n_errors, 100 * n_errors / n_kept,
    )

    if n_errors < 5:
        logger.warning(
            "Only %d errors in training set; AUROC will be very noisy.",
            n_errors,
        )

    # ------ Train one probe per layer ------
    per_layer_results = {}
    for ℓ in layers:
        X = per_layer[ℓ]["X"]
        y = per_layer[ℓ]["y"]
        logger.info("Training probe for layer %d (X=%s, y=%s)...", ℓ, X.shape, y.shape)
        probe = train_probe(
            X=X, y=y, layer=ℓ,
            C=cfg.probe.C,
            max_iter=cfg.probe.max_iter,
            class_weight=cfg.probe.class_weight,
            cv_folds=cfg.probe.cv_folds,
            test_size=cfg.probe.test_size,
            random_state=cfg.experiment.seed,
        )
        per_layer_results[ℓ] = probe

        # Save probe to disk.
        probe_path = probes_dir / f"probe_layer{ℓ}.pkl"
        with probe_path.open("wb") as f:
            pickle.dump(probe, f)
        logger.info(
            "Layer %d: AUROC=%.3f, AUPRC=%.3f, n_test=%d (%d err)",
            ℓ, probe.metrics.auroc, probe.metrics.auprc,
            probe.metrics.n_test, probe.metrics.n_test_errors,
        )

    # ------ Summary ------
    summary_path = probes_dir / "_summary.json"
    write_summary(per_layer_results, summary_path, n_kept, n_errors)

    logger.info("=" * 70)
    logger.info("Probes saved to: %s", probes_dir)
    logger.info("Summary: %s", summary_path)

    # Best layer printout.
    best_layer = max(per_layer_results.items(),
                     key=lambda kv: kv[1].metrics.auroc
                     if not np.isnan(kv[1].metrics.auroc) else -1)
    logger.info(
        "Best layer: %d (AUROC=%.3f)",
        best_layer[0], best_layer[1].metrics.auroc,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
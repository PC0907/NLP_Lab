"""Stage 4: Evaluate probe vs. token-logprob baselines.

Builds on Stage 3 outputs (trained probes) and Stage 1 outputs (extractions
with per-token logprobs and per-field token spans). Computes AUROC for:
  - Each trained probe (re-evaluated on the same held-out test split for
    fair comparison)
  - Mean-logprob baseline
  - Min-logprob baseline

Reports per-field comparisons so we can see where probe and baselines
agree/disagree.

Artifacts written:
  artifacts/results/comparison.json   — head-to-head AUROC numbers

Usage:
    python scripts/04_evaluate.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

from probe_extraction.baselines import (
    compute_token_logprob_scores,
    evaluate_baseline,
)
from probe_extraction.config import load_config
from probe_extraction.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare probe vs. baselines.")
    p.add_argument("--config", required=True)
    return p.parse_args()


def load_extraction_lookup(extractions_dir: Path) -> dict:
    """Map doc_id → {token_logprobs, fields_meta}.

    fields_meta is a dict mapping path_str → token_span tuple, so we can
    look up logprob slices per field quickly.
    """
    lookup: dict[str, dict] = {}
    for path in sorted(extractions_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        with path.open("r", encoding="utf-8") as f:
            ext = json.load(f)
        if ext.get("parse_error") or not ext.get("token_logprobs"):
            continue
        fields_by_path = {
            f["path_str"]: tuple(f["token_span"]) for f in ext.get("fields", [])
        }
        lookup[ext["doc_id"]] = {
            "token_logprobs": ext["token_logprobs"],
            "fields_by_path": fields_by_path,
        }
    return lookup


def build_evaluation_dataset(
    *,
    labels_dir: Path,
    activations_dir: Path,
    extraction_lookup: dict,
    layers: list[int],
):
    """Build aligned arrays of (per-layer activations, baseline scores, labels).

    Returns:
        per_layer_X: dict[layer] -> (n, hidden_dim) activation array
        mean_logprobs: (n,) array
        min_logprobs: (n,) array
        y: (n,) binary error labels
        meta: list of per-row dicts with doc_id and path_str
    """
    per_layer_X: dict[int, list] = {ℓ: [] for ℓ in layers}
    mean_logprobs: list[float] = []
    min_logprobs: list[float] = []
    y: list[int] = []
    meta: list[dict] = []

    for label_path in sorted(labels_dir.glob("*.json")):
        if label_path.name.startswith("_"):
            continue
        with label_path.open("r", encoding="utf-8") as f:
            label_doc = json.load(f)

        doc_id = label_doc["doc_id"]
        if doc_id not in extraction_lookup:
            continue

        ext_info = extraction_lookup[doc_id]
        token_logprobs = ext_info["token_logprobs"]
        fields_by_path = ext_info["fields_by_path"]

        activations_path = activations_dir / f"{doc_id}.npz"
        if not activations_path.exists():
            continue

        with np.load(activations_path) as activations_npz:
            available_keys = set(activations_npz.keys())

            for lab in label_doc["labels"]:
                if not lab["extracted_present"]:
                    continue  # synthetic activations, skip
                path_str = lab["path_str"]
                if path_str not in fields_by_path:
                    continue  # extractor didn't save this field

                # Pull all per-layer activations.
                missing = False
                vecs = {}
                for ℓ in layers:
                    key = f"{path_str}__layer{ℓ}"
                    if key not in available_keys:
                        missing = True
                        break
                    vecs[ℓ] = activations_npz[key].astype(np.float32)
                if missing:
                    continue

                # Compute baseline scores.
                lp = compute_token_logprob_scores(
                    token_logprobs=token_logprobs,
                    token_span=fields_by_path[path_str],
                )

                # Append to all parallel arrays.
                for ℓ in layers:
                    per_layer_X[ℓ].append(vecs[ℓ])
                mean_logprobs.append(lp["mean_logprob"])
                min_logprobs.append(lp["min_logprob"])
                y.append(int(lab["is_error"]))
                meta.append({"doc_id": doc_id, "path_str": path_str})

    # Stack
    per_layer_X_arr = {
        ℓ: np.stack(per_layer_X[ℓ], axis=0) if per_layer_X[ℓ]
        else np.empty((0, 0), dtype=np.float32)
        for ℓ in layers
    }
    return (
        per_layer_X_arr,
        np.array(mean_logprobs, dtype=np.float32),
        np.array(min_logprobs, dtype=np.float32),
        np.array(y, dtype=np.int32),
        meta,
    )


def evaluate_probe_on_full_set(probe, X: np.ndarray, y: np.ndarray, layer: int):
    """Score the trained probe on the entire dataset, return AUROC/AUPRC.

    NOTE: This is NOT held-out evaluation — the probe was trained on this
    same data (refit on full set after CV). We're using it here only for
    fair side-by-side comparison with baselines that didn't train on
    anything. The HONEST AUROC is the CV AUROC reported by Stage 3; this
    evaluation is for ranking-pattern comparison, not generalization claims.
    """
    from probe_extraction.baselines import evaluate_baseline as eval_b
    proba = probe.score(X)
    return eval_b(
        scores=proba, y=y,
        name=f"probe_layer{layer}",
        score_higher_is_error=True,  # probe outputs P(error)
    )


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    setup_logging(
        level=cfg.logging.level,
        log_dir=cfg.logging.log_dir,
        log_name="04_evaluate",
        log_to_file=cfg.logging.log_to_file,
    )

    artifacts = cfg.artifacts_path
    extractions_dir = artifacts / "extractions"
    activations_dir = artifacts / "activations"
    labels_dir = artifacts / "labels"
    probes_dir = artifacts / "probes"
    results_dir = artifacts / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    layers = cfg.activations.layers

    # ------ Build dataset ------
    extraction_lookup = load_extraction_lookup(extractions_dir)
    per_layer_X, mean_lp, min_lp, y, meta = build_evaluation_dataset(
        labels_dir=labels_dir,
        activations_dir=activations_dir,
        extraction_lookup=extraction_lookup,
        layers=layers,
    )

    n = len(y)
    n_errors = int(y.sum())
    logger.info(
        "Evaluation set: %d fields, %d errors (%.1f%%)",
        n, n_errors, 100 * n_errors / max(n, 1),
    )

    # ------ Baselines ------
    mean_lp_metrics = evaluate_baseline(
        scores=mean_lp, y=y,
        name="mean_logprob", score_higher_is_error=False,
    )
    min_lp_metrics = evaluate_baseline(
        scores=min_lp, y=y,
        name="min_logprob", score_higher_is_error=False,
    )

    logger.info(
        "Baseline mean_logprob: AUROC=%.3f, AUPRC=%.3f",
        mean_lp_metrics.auroc, mean_lp_metrics.auprc,
    )
    logger.info(
        "Baseline min_logprob:  AUROC=%.3f, AUPRC=%.3f",
        min_lp_metrics.auroc, min_lp_metrics.auprc,
    )

    # ------ Probes ------
    probe_results = {}
    for ℓ in layers:
        probe_path = probes_dir / f"probe_layer{ℓ}.pkl"
        if not probe_path.exists():
            logger.warning("No probe for layer %d at %s", ℓ, probe_path)
            continue
        with probe_path.open("rb") as f:
            probe = pickle.load(f)

        m = evaluate_probe_on_full_set(probe, per_layer_X[ℓ], y, ℓ)
        probe_results[ℓ] = {
            "auroc_full_set": m.auroc,
            "auprc_full_set": m.auprc,
            # The honest CV-AUROC from training stage:
            "auroc_cv_mean": (
                float(np.mean(probe.metrics.per_fold_auroc))
                if probe.metrics.per_fold_auroc else None
            ),
            "auroc_cv_std": (
                float(np.std(probe.metrics.per_fold_auroc))
                if probe.metrics.per_fold_auroc else None
            ),
        }
        logger.info(
            "Probe layer %d: AUROC (CV)=%.3f±%.3f, AUROC (full-set, optimistic)=%.3f",
            ℓ,
            probe_results[ℓ]["auroc_cv_mean"] or float("nan"),
            probe_results[ℓ]["auroc_cv_std"] or 0.0,
            m.auroc,
        )

    # ------ Comparison summary ------
    comparison = {
        "n_samples": n,
        "n_errors": n_errors,
        "baselines": {
            "mean_logprob": {
                "auroc": mean_lp_metrics.auroc,
                "auprc": mean_lp_metrics.auprc,
            },
            "min_logprob": {
                "auroc": min_lp_metrics.auroc,
                "auprc": min_lp_metrics.auprc,
            },
        },
        "probes": probe_results,
    }
    comparison_path = results_dir / "comparison.json"
    with comparison_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    # ------ Headline output ------
    logger.info("=" * 70)
    logger.info("HEAD-TO-HEAD AUROC (higher = better):")
    logger.info("  Baseline mean_logprob:  %.3f", mean_lp_metrics.auroc)
    logger.info("  Baseline min_logprob:   %.3f", min_lp_metrics.auroc)
    for ℓ, pr in probe_results.items():
        cv = pr["auroc_cv_mean"]
        cv_str = f"{cv:.3f}" if cv is not None else "n/a"
        logger.info(
            "  Probe layer %d (CV):     %s ± %.3f",
            ℓ, cv_str, pr["auroc_cv_std"] or 0.0,
        )
    logger.info("Comparison saved to: %s", comparison_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
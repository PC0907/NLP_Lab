#!/usr/bin/env python3
"""Fit a probe on the fresh all-tokens artifacts, for use as a steering direction.

WHY NOT scripts/03_train_probe.py
---------------------------------
Two reasons, both blocking:
  1. It takes only --config, so there is no way to exclude a domain. It would
     train on 10kq, ~54% of whose fields have gold absent from the document
     and so are unanswerable by any extractor.
  2. It appends `activations_npz[key]` directly with no handling for 2D
     arrays. On all-tokens artifacts every stored array is [n_tokens, hidden],
     so it would either fail on ragged stacking or silently train on the wrong
     thing.

This reuses probe_extraction.probes.train_probe -- the same fitting code, the
same LinearProbe output -- so the result is consistent with every other probe
in the project and steering_smoke.py reads it unchanged via obj.weights.

TOKEN POSITION
--------------
Default is the span MIDPOINT, not the last token. The position experiment
found the midpoint better at all five layers tested, e.g. layer 18: 0.9612 vs
0.9232 on multi-token fields. The pipeline's existing results all use the last
token and should stay that way for consistency, but this is a NEW experiment,
so there is no consistency argument -- and the stronger direction is the better
thing to intervene with. `--position last` reproduces the old behaviour if you
want both.

Usage:
  python fit_steering_direction.py --config configs/exp_fresh_alltokens.yaml \
      --exclude-domains finance/10kq --position mid --out-suffix mid
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

from probe_extraction.config import load_config
from probe_extraction.probes import train_probe

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

POSITIONS = {
    "first": lambda s: s[0],
    "mid":   lambda s: s[int(round(0.5 * (len(s) - 1)))],
    "last":  lambda s: s[-1],
    "mean":  lambda s: s.mean(axis=0),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Default: the config's own layer list.")
    p.add_argument("--exclude-domains", nargs="*", default=["finance/10kq"])
    p.add_argument("--position", choices=list(POSITIONS), default="mid")
    p.add_argument("--out-suffix", default=None,
                   help="Written as probe_layer{L}_{suffix}.pkl so the "
                        "pipeline's own probes are never overwritten. "
                        "Defaults to the position name.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    layers = args.layers or cfg.activations.layers
    suffix = args.out_suffix or args.position
    pos_fn = POSITIONS[args.position]

    artifacts = cfg.artifacts_path
    labels_dir = artifacts / "labels"
    acts_dir = artifacts / "activations"
    probes_dir = artifacts / "probes"
    probes_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Artifacts : %s", artifacts)
    logger.info("Layers    : %s", layers)
    logger.info("Position  : %s", args.position)
    logger.info("Excluding : %s", args.exclude_domains)

    per_layer = {L: {"X": [], "y": []} for L in layers}
    n_docs = n_fields = n_synth = n_missing = 0
    skipped_domains = {}

    for lp in sorted(labels_dir.glob("*.json")):
        if lp.name.startswith("_"):
            continue
        doc = json.load(lp.open())
        dom = doc.get("domain", "?")
        if dom in args.exclude_domains:
            skipped_domains[dom] = skipped_domains.get(dom, 0) + 1
            continue
        doc_id = doc["doc_id"]
        npz_path = acts_dir / f"{doc_id}.npz"
        if not npz_path.exists():
            logger.warning("No activations for %s; skipping document.", doc_id)
            continue
        n_docs += 1

        with np.load(npz_path) as A:
            keys = set(A.files)
            for lab in doc["labels"]:
                n_fields += 1
                if not lab.get("extracted_present", True):
                    # Synthetic activation -- the extractor placed it at the
                    # cursor, not at a real value position.
                    n_synth += 1
                    continue
                ps = lab["path_str"]
                if any(f"{ps}__layer{L}" not in keys for L in layers):
                    n_missing += 1
                    continue
                y = int(lab["is_error"])
                for L in layers:
                    v = A[f"{ps}__layer{L}"].astype(np.float32)
                    if v.ndim == 1:
                        v = v[None, :]
                    per_layer[L]["X"].append(pos_fn(v))
                    per_layer[L]["y"].append(y)

    if skipped_domains:
        logger.info("Domain filter skipped: %s",
                    ", ".join(f"{k}={v}" for k, v in sorted(skipped_domains.items())))

    sizes = {L: len(per_layer[L]["y"]) for L in layers}
    if len(set(sizes.values())) != 1:
        logger.error("Layer sample counts differ: %s", sizes)
        return 1
    n = sizes[layers[0]]
    if n == 0:
        logger.error("No usable fields after filtering.")
        return 1
    n_err = int(np.sum(per_layer[layers[0]]["y"]))
    logger.info("Loaded: %d documents, %d fields seen, %d synthetic skipped, "
                "%d missing-activation skipped", n_docs, n_fields, n_synth, n_missing)
    logger.info("Training set: %d fields, %d errors (%.1f%%)",
                n, n_err, 100 * n_err / n)
    if n_err < 5:
        logger.warning("Only %d errors -- metrics will be very noisy.", n_err)

    summary = {}
    for L in layers:
        X = np.stack(per_layer[L]["X"])
        y = np.asarray(per_layer[L]["y"], dtype=np.int32)
        probe = train_probe(
            X=X, y=y, layer=L,
            C=cfg.probe.C, max_iter=cfg.probe.max_iter,
            class_weight=cfg.probe.class_weight,
            cv_folds=cfg.probe.cv_folds, test_size=cfg.probe.test_size,
            random_state=cfg.experiment.seed,
        )
        out = probes_dir / f"probe_layer{L}_{suffix}.pkl"
        with out.open("wb") as f:
            pickle.dump(probe, f)
        w = np.asarray(probe.weights)
        summary[str(L)] = {
            "auroc": probe.metrics.auroc,
            "auprc": probe.metrics.auprc,
            "weight_norm": float(np.linalg.norm(w)),
            "path": str(out),
        }
        logger.info("  layer %-3d AUROC %.4f  AUPRC %.4f  |w| %.3f  -> %s",
                    L, probe.metrics.auroc, probe.metrics.auprc,
                    float(np.linalg.norm(w)), out.name)

    sp = probes_dir / f"_summary_{suffix}.json"
    sp.write_text(json.dumps({
        "position": args.position,
        "exclude_domains": args.exclude_domains,
        "n_fields": n, "n_errors": n_err,
        "per_layer": summary,
    }, indent=2))

    logger.info("")
    logger.info("Saved to %s", sp)
    logger.info("")
    logger.info("NOTE: the AUROC figures above come from train_probe's own")
    logger.info("internal CV/test split, which is RANDOM-FOLD and therefore has")
    logger.info("document-level leakage. They are not comparable to the nested")
    logger.info("LODO headline and must not be quoted. They are printed only as")
    logger.info("a sanity check that the direction is not degenerate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""Nested-LODO probe comparison across the (model x parser) grid.

For each parser:
  1. Find the documents BOTH models completed (the intersection). Comparing
     each model on its own document set would confound model quality with
     which documents each happened to finish -- Gemma loses long credit
     agreements to memory, Qwen does not.
  2. Drop documents that cannot serve as LODO folds (single-class) and any
     excluded domains (10-K by default, as everywhere else in the project).
  3. Run nested LODO on each model restricted to that intersection, so both
     numbers come from identical documents under identical protocol.

Reuses 05b_nested_lodo.py's loader and its --include-docs-file mechanism.
CPU only.

Usage:
  python grid_probe.py --parsers pymupdf docling camelot \
      --models qwen35_4b gemma3_4b --exclude-domains finance/10kq
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parsers", nargs="+", default=["pymupdf", "docling", "camelot"])
    p.add_argument("--models", nargs="+", default=["qwen35_4b", "gemma3_4b"])
    p.add_argument("--exclude-domains", nargs="*", default=["finance/10kq"])
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Candidate layers for nested selection. Default: each "
                        "config's own band, which differs between models.")
    p.add_argument("--out", default="artifacts/grid_probe_comparison.json")
    return p.parse_args()


def usable_docs(artifact_dir: Path, exclude_domains: list[str]) -> dict[str, dict]:
    """doc_id -> {domain, n_fields, n_errors} for documents that have labels,
    activations, both classes present, and are not in an excluded domain."""
    labels_dir = artifact_dir / "labels"
    acts_dir = artifact_dir / "activations"
    out = {}
    for lp in sorted(labels_dir.glob("*.json")):
        if lp.name.startswith("_"):
            continue
        d = json.load(lp.open())
        dom = d.get("domain", "?")
        if dom in exclude_domains:
            continue
        doc_id = d.get("doc_id", lp.stem)
        if not (acts_dir / f"{doc_id}.npz").exists():
            continue
        L = [f for f in d.get("labels", []) if f.get("extracted_present", True)]
        n = len(L)
        e = sum(int(f.get("is_error", 0)) for f in L)
        if n == 0 or e == 0 or e == n:
            continue  # cannot be a LODO fold
        out[doc_id] = {"domain": dom, "n_fields": n, "n_errors": e}
    return out


def run_nested(config: Path, include_file: Path, out_name: str, layers):
    cmd = [sys.executable, "scripts/05b_nested_lodo.py",
           "--config", str(config),
           "--include-docs-file", str(include_file),
           "--out-name", out_name]
    if layers:
        cmd += ["--layers"] + [str(l) for l in layers]
    logger.info("  $ %s", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        logger.error("  nested LODO failed:\n%s", r.stderr[-2000:])
        return None
    # 05b writes <artifacts>/results/<out_name>
    exp = json.load(open(config))["experiment"]["name"] if config.suffix == ".json" else None
    return r.stdout


def main():
    args = parse_args()
    results = {}

    for parser in args.parsers:
        logger.info("=" * 72)
        logger.info("PARSER: %s", parser)

        per_model = {}
        for m in args.models:
            d = Path(f"artifacts/grid_{m}_{parser}")
            if not d.exists():
                logger.warning("  %s: no artifacts at %s, skipping parser", m, d)
                break
            per_model[m] = usable_docs(d, args.exclude_domains)
            logger.info("  %-12s %2d usable docs", m, len(per_model[m]))
        if len(per_model) < len(args.models):
            continue

        inter = sorted(set.intersection(*(set(v) for v in per_model.values())))
        logger.info("  INTERSECTION: %d docs", len(inter))
        by_dom = {}
        for doc in inter:
            by_dom[per_model[args.models[0]][doc]["domain"]] = \
                by_dom.get(per_model[args.models[0]][doc]["domain"], 0) + 1
        logger.info("  by domain: %s", by_dom)
        if len(inter) < 3:
            logger.warning("  too few documents for LODO, skipping")
            continue

        inc = Path(f"intersection_{parser}.txt")
        inc.write_text("\n".join(inter) + "\n")

        results[parser] = {"intersection": inter, "by_domain": by_dom, "models": {}}
        for m in args.models:
            cfg = Path(f"configs/exp_grid_{m}_{parser}.yaml")
            out_name = f"nested_lodo_grid_{parser}.json"
            logger.info("  running nested LODO: %s", m)
            run_nested(cfg, inc, out_name, args.layers)
            res_path = Path(f"artifacts/grid_{m}_{parser}/results/{out_name}")
            if res_path.exists():
                r = json.load(res_path.open())
                results[parser]["models"][m] = r
                logger.info("  %-12s pooled-OOF AUROC %.4f | per-fold %.4f +/- %.4f | layers %s",
                            m, r.get("pooled_oof_auroc", float("nan")),
                            r.get("auroc_mean", float("nan")),
                            r.get("auroc_std", float("nan")),
                            r.get("layers_selected"))
            else:
                logger.error("  %s: no result written", m)

    # ---- summary table ----
    logger.info("=" * 72)
    logger.info("SUMMARY  (nested LODO, pooled-OOF AUROC, identical documents per parser)")
    hdr = f"{'parser':10} {'n_docs':>6}" + "".join(f"{m:>14}" for m in args.models)
    logger.info(hdr)
    for parser, r in results.items():
        row = f"{parser:10} {len(r['intersection']):>6}"
        for m in args.models:
            v = r["models"].get(m, {}).get("pooled_oof_auroc")
            row += f"{v:>14.4f}" if v is not None else f"{'---':>14}"
        logger.info(row)

    Path(args.out).write_text(json.dumps(results, indent=2))
    logger.info("Saved to %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""Re-score a regeneration cache under STRICT and LENIENT matchers.

The strict AUTO matcher (numbers/dates/case) counts semantically-equivalent
values as wrong:
  - currency "USD" vs "Dollars"  (synonym)
  - "New York" vs "State of New York"  (containment)
  - long legal clauses  (paraphrase — unscorable by exact match)

This inflates apparent "breakage" in selective regeneration. We report BOTH:
  - strict  : the original AUTO matcher (conservative, what we had).
  - lenient : AUTO + synonym groups + substring containment, with long-clause
              fields excluded from scoring entirely.

Reporting both is deliberate: the matcher leniency is a post-hoc observation,
so we show the strict number too rather than silently swapping. Per the
pre-registration logic, a matcher change is justified only if an audit shows
>25% of a field's errors are matcher artefacts — these numbers feed that audit.

Usage:
  python scripts/rescore_regen.py \
      --domain qwen35_4b_credit \
      --cache artifacts/qwen35_4b_credit/results/regen_cache_t0.json \
      --probe-path artifacts/qwen35_4b_pooled/probes/probe_layer18.pkl
"""
from __future__ import annotations

import argparse
import json
import glob
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from probe_extraction.labeling.value_compare import compare_values, ComparisonStrategy

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ---- lenient matcher pieces ----

_SYNONYMS = [
    {"usd", "dollars", "us dollars", "u.s. dollars", "$", "us$"},
    {"eur", "euros", "euro", "€"},
    {"gbp", "pounds", "british pounds", "£", "pounds sterling"},
    {"true", "yes"},
    {"false", "no"},
]

# Leaf field names whose values are long free-text clauses that exact-match
# scoring cannot fairly judge. Excluded from the LENIENT score.
_UNSCORABLE = {
    "borrowing_request", "use_of_proceeds", "authorized_officer_definition",
    # add domain-specific long-text leaves here as the audit identifies them
}


def _norm(s) -> str:
    return str(s).lower().strip().strip(".,;:'\" ").replace("  ", " ")


def _synonym_equal(g, n) -> bool:
    gn, nn = _norm(g), _norm(n)
    for grp in _SYNONYMS:
        if gn in grp and nn in grp:
            return True
    return False


def strict_ok(gold, new) -> bool:
    try:
        return bool(compare_values(gold, new, strategy=ComparisonStrategy.AUTO,
                                   number_tolerance=0.0))
    except Exception:
        return False


def lenient_ok(gold, new) -> bool:
    if strict_ok(gold, new):
        return True
    if gold is None or new is None:
        return False
    if _synonym_equal(gold, new):
        return True
    g, n = _norm(gold), _norm(new)
    # containment, guarded by min length to avoid trivial coincidences
    if len(g) >= 4 and len(n) >= 4 and (g in n or n in g):
        return True
    return False


def is_scorable_lenient(path_str: str) -> bool:
    return path_str.split(".")[-1] not in _UNSCORABLE


# ---- load labels + probe scores ----

def load_labels(domain: str) -> dict:
    labels = {}
    for f in glob.glob(f"artifacts/{domain}/labels/*.json"):
        if "_summary" in f:
            continue
        d = json.load(open(f))
        for l in d["labels"]:
            labels[(d["doc_id"], l["path_str"])] = l
    return labels


def probe_scores(domain: str, probe, layer: int) -> dict:
    """Map (doc_id, path_str) -> probe P(error)."""
    out = {}
    actdir = Path(f"artifacts/{domain}/activations")
    for f in glob.glob(f"artifacts/{domain}/labels/*.json"):
        if "_summary" in f:
            continue
        d = json.load(open(f))
        doc_id = d["doc_id"]
        npz = actdir / f"{doc_id}.npz"
        if not npz.exists():
            continue
        with np.load(npz) as acts:
            for l in d["labels"]:
                key = f"{l['path_str']}__layer{layer}"
                if key in acts:
                    s = float(probe.score(acts[key][None, :].astype(np.float32))[0])
                    out[(doc_id, l["path_str"])] = s
    return out


def sweep(enriched, score_key, reverse=True):
    ordered = sorted(enriched, key=lambda c: c[score_key], reverse=reverse)
    cost, net, run = [0], [0], 0
    for k, c in enumerate(ordered, 1):
        run += c["delta"]
        cost.append(k); net.append(run)
    return cost, net


def evaluate(cache, labels, pscores, mode: str):
    """mode in {'strict','lenient'}. Returns enriched candidates + totals."""
    ok = strict_ok if mode == "strict" else lenient_ok
    enriched, fixed, broke, stillwrong, keptok = [], 0, 0, 0, 0
    for key, entry in cache.items():
        doc_id, path = key.split("::", 1)
        lab = labels.get((doc_id, path))
        if lab is None:
            continue
        if mode == "lenient" and not is_scorable_lenient(path):
            continue  # drop unscorable long clauses in lenient mode
        gold = lab.get("gold_value")
        was_err = int(lab.get("is_error", 0)) == 1
        now_ok = ok(gold, entry["new_value"])
        if was_err and now_ok:
            delta = +1; fixed += 1
        elif (not was_err) and (not now_ok):
            delta = -1; broke += 1
        elif was_err:
            delta = 0; stillwrong += 1
        else:
            delta = 0; keptok += 1
        enriched.append({
            "delta": delta,
            "probe": pscores.get((doc_id, path), 0.0),
            "is_error": 1 if was_err else 0,
        })
    return enriched, dict(fixed=fixed, broke=broke, still_wrong=stillwrong, kept_ok=keptok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, help="e.g. qwen35_4b_credit")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--probe-path", required=True)
    ap.add_argument("--layer", type=int, default=18)
    args = ap.parse_args()

    cache = json.load(open(args.cache))
    labels = load_labels(args.domain)
    probe = pickle.load(open(args.probe_path, "rb"))
    pscores = probe_scores(args.domain, probe, args.layer)

    for mode in ("strict", "lenient"):
        enriched, totals = evaluate(cache, labels, pscores, mode)
        n = len(enriched)
        logger.info("=" * 64)
        logger.info("MODE: %s  (%s)", mode.upper(), args.cache.split("/")[-1])
        logger.info("  candidates scored: %d", n)
        logger.info("  fixed=%d  broke=%d  still_wrong=%d  kept_ok=%d",
                    totals["fixed"], totals["broke"], totals["still_wrong"], totals["kept_ok"])
        logger.info("  net if regenerate ALL: %d", totals["fixed"] - totals["broke"])
        # probe vs random vs oracle ordering curve
        c_probe, n_probe = sweep(enriched, "probe", reverse=True)
        c_oracle, n_oracle = sweep(enriched, "is_error", reverse=True)
        import random; rng = random.Random(42)
        sh = enriched[:]; rng.shuffle(sh)
        cr, nr, run = [0], [0], 0
        for k, c in enumerate(sh, 1):
            run += c["delta"]; cr.append(k); nr.append(run)
        budgets = [int(n * f) for f in (0.1, 0.25, 0.5, 1.0)]
        logger.info("  budget    probe   random   oracle")
        for b in budgets:
            i = min(b, n)
            logger.info("   %4d    %5d   %5d   %5d", b, n_probe[i], nr[i], n_oracle[i])


if __name__ == "__main__":
    main()
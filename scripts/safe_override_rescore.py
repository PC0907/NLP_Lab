#!/usr/bin/env python3
"""Safe-override re-scoring for selective regeneration (PromptPort-style policy).

The plain rescore measures "net if you regenerate EVERY candidate", which is a
strawman: it overwrites ~1900 correct fields just to fix a few errors, so it goes
strongly net-negative. The deployed policy is NOT "regenerate everything" -- it is
a conservative gate (cf. PromptPort's Safe-Override): only override a field when a
trust signal indicates the original is likely wrong.

This script applies that gate using the PROBE as the trust signal:
  - For a threshold tau, "override" a field only if its ORIGINAL probe P(error)
    >= tau (the probe flags it as risky). Otherwise keep the original untouched.
  - Among overridden fields, the new (regenerated) value replaces the original.
  - Report, per tau: override rate, override PRECISION (fixed / overridden),
    fixed, broke, net. Override precision is the key metric -- of the fields you
    chose to replace, what fraction did you improve rather than break.

It also picks tau on HELD-OUT documents (leave-one-doc-out over the cached docs),
so the reported operating point is not tuned on the same fields it is scored on.

Reuses rescore_regen.py for the matcher (strict/lenient), value normalization,
and probe scoring -- so outcomes are identical to that script, just gated.

Usage:
  python scripts/safe_override_rescore.py \
      --domain qwen35_4b_pooled_alltokens \
      --cache artifacts/qwen35_4b_pooled_alltokens/results/regen_cache_pooled.json \
      --probe-path artifacts/qwen35_4b_pooled/probes/probe_layer18.pkl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
_rs = import_module("rescore_regen")
strict_ok = _rs.strict_ok
lenient_ok = _rs.lenient_ok
is_scorable_lenient = _rs.is_scorable_lenient
load_labels = _rs.load_labels
probe_scores = _rs.probe_scores

import pickle
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def build_rows(cache, labels, pscores, mode):
    """One row per cached candidate: original-correct?, regen-correct?, the
    delta if overridden, the ORIGINAL field's probe P(error), and doc_id."""
    ok = strict_ok if mode == "strict" else lenient_ok
    rows = []
    for key, entry in cache.items():
        doc_id, path = key.split("::", 1)
        lab = labels.get((doc_id, path))
        if lab is None:
            continue
        if mode == "lenient" and not is_scorable_lenient(path, lab.get("gold_value")):
            continue
        gold = lab.get("gold_value")
        was_err = int(lab.get("is_error", 0)) == 1
        now_ok = ok(gold, entry["new_value"])
        # delta IF we override this field with the regenerated value
        if was_err and now_ok:
            delta = +1          # fixed
        elif (not was_err) and (not now_ok):
            delta = -1          # broke a correct field
        else:
            delta = 0           # error->still wrong, or correct->still correct
        rows.append({
            "doc_id": doc_id,
            "path": path,
            "was_err": was_err,
            "delta": delta,
            "p_err": float(pscores.get((doc_id, path), 0.0)),  # ORIGINAL probe score
        })
    return rows


def gate_metrics(rows, tau):
    """Apply the override gate at threshold tau (override iff p_err >= tau).
    Returns override count, precision, fixed, broke, net."""
    overridden = [r for r in rows if r["p_err"] >= tau]
    n_over = len(overridden)
    fixed = sum(1 for r in overridden if r["delta"] == +1)
    broke = sum(1 for r in overridden if r["delta"] == -1)
    net = fixed - broke
    # override precision: of fields we replaced, fraction that improved.
    # (count only fields where the override actually CHANGED correctness:
    #  fixed + broke are the changes; precision = fixed / (fixed + broke))
    changed = fixed + broke
    precision = (fixed / changed) if changed else float("nan")
    return {
        "tau": tau, "n_override": n_over,
        "override_rate": n_over / len(rows) if rows else 0.0,
        "fixed": fixed, "broke": broke, "net": net,
        "override_precision": precision,
    }


def best_tau_heldout(rows, taus):
    """Leave-one-document-out: for each held-out doc, pick tau that maximizes net
    on the OTHER docs, apply it to the held-out doc, accumulate. Returns the
    held-out net + the tau distribution (so the operating point isn't tuned on
    the same data it's scored on)."""
    docs = sorted({r["doc_id"] for r in rows})
    if len(docs) < 2:
        return None
    heldout_net = 0
    heldout_fixed = 0
    heldout_broke = 0
    tau_chosen = {}
    for d in docs:
        train = [r for r in rows if r["doc_id"] != d]
        test = [r for r in rows if r["doc_id"] == d]
        # pick tau maximizing net on train
        best = max(taus, key=lambda t: gate_metrics(train, t)["net"])
        tau_chosen[best] = tau_chosen.get(best, 0) + 1
        m = gate_metrics(test, best)
        heldout_net += m["net"]
        heldout_fixed += m["fixed"]
        heldout_broke += m["broke"]
    return {
        "heldout_net": heldout_net,
        "heldout_fixed": heldout_fixed,
        "heldout_broke": heldout_broke,
        "tau_chosen": tau_chosen,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--probe-path", required=True)
    ap.add_argument("--layer", type=int, default=18)
    args = ap.parse_args()

    cache = json.load(open(args.cache))
    labels = load_labels(args.domain)
    probe = pickle.load(open(args.probe_path, "rb"))
    pscores = probe_scores(args.domain, probe, args.layer)

    taus = [round(x, 2) for x in np.linspace(0.0, 0.95, 20)]

    for mode in ("strict", "lenient"):
        rows = build_rows(cache, labels, pscores, mode)
        n = len(rows)
        logger.info("=" * 72)
        logger.info("MODE: %s   (%d candidates, %s)", mode.upper(), n, args.cache.split("/")[-1])
        logger.info("  Safe-override gate: override field iff ORIGINAL probe P(error) >= tau")
        logger.info("  %-6s %9s %9s %7s %7s %7s   %s", "tau", "n_over", "ovr_rate",
                    "fixed", "broke", "net", "ovr_precision")
        # full-set sweep (for the curve / picking an operating point)
        best_net_row = None
        for t in taus:
            m = gate_metrics(rows, t)
            if best_net_row is None or m["net"] > best_net_row["net"]:
                best_net_row = m
            prec = m["override_precision"]
            prec_s = f"{prec:.2f}" if prec == prec else "  n/a"
            logger.info("  %-6.2f %9d %9.3f %7d %7d %7d   %s",
                        t, m["n_override"], m["override_rate"],
                        m["fixed"], m["broke"], m["net"], prec_s)
        logger.info("  -> best full-set net = %d at tau=%.2f (override_rate=%.3f, precision=%.2f)",
                    best_net_row["net"], best_net_row["tau"],
                    best_net_row["override_rate"],
                    best_net_row["override_precision"]
                    if best_net_row["override_precision"] == best_net_row["override_precision"] else 0.0)
        # held-out tau selection (honest operating point)
        ho = best_tau_heldout(rows, taus)
        if ho:
            logger.info("  -> HELD-OUT (tau picked per-fold on other docs): "
                        "net=%d (fixed=%d, broke=%d); tau dist=%s",
                        ho["heldout_net"], ho["heldout_fixed"], ho["heldout_broke"],
                        ho["tau_chosen"])


if __name__ == "__main__":
    main()
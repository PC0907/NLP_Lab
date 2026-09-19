#!/usr/bin/env python3
"""Where in a field's token span does the error signal live?

WHY THIS MATTERS FOR STEERING
-----------------------------
The probe is trained on the activation at the LAST token of each field's value:
for `authors.0.name` = "Y. Le Cun", the position after the name is finished. To
*prevent* an error by steering, you must intervene at or before the FIRST token
of the value -- before it is committed. Those are different positions, and it is
an open question whether the error direction exists at the earlier one.

If the signal holds at the first token, steering-as-prevention is plausible. If
it appears only at the last token, the model does not yet "know" it is about to
err, and steering to prevent would be fighting the mechanism rather than using
it. Either answer is informative; the first is a green light.

This uses the existing all-tokens activations. No GPU, no re-extraction.

CRITICAL READING NOTE
---------------------
~46% of fields are SINGLE-TOKEN (median span 2 on the pooled set). For those,
first == last by construction, which dilutes any first-vs-last comparison
toward "no difference". Every result below is therefore reported twice: over
all fields, and over MULTI-TOKEN fields only. The second is the one that
answers the question.

UNVERIFIED ASSUMPTION
---------------------
This assumes the stored token span covers the field's VALUE tokens. If the
extractor's span also includes key/punctuation tokens, index 0 is not the first
value token and the "first" column means something slightly different. Worth
confirming against extraction/extractor.py before drawing conclusions.

Usage:
  python token_position_probe.py --domain fresh_qwen35_4b_pooled_alltokens \
      --layers 14 16 18 20 22 --exclude-domains finance/10kq
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="fresh_qwen35_4b_pooled_alltokens")
    p.add_argument("--layers", type=int, nargs="+", default=[14, 16, 18, 20, 22])
    p.add_argument("--exclude-domains", nargs="*", default=["finance/10kq"])
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--out", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loading. Keeps the FULL 2D span per field so positions can be indexed.
# ---------------------------------------------------------------------------

def load_spans(domain, layers, exclude):
    """Returns rows: (doc_id, y, {layer: 2D array [n_tokens, hidden]})."""
    labels_dir = Path(f"artifacts/{domain}/labels")
    acts_dir = Path(f"artifacts/{domain}/activations")
    rows = []
    skipped = Counter()

    for lp in sorted(labels_dir.glob("*.json")):
        if lp.name.startswith("_"):
            continue
        data = json.load(lp.open())
        dom = data.get("domain", "?")
        if dom in exclude:
            skipped[dom] += 1
            continue
        doc_id = data.get("doc_id", lp.stem)
        npz = acts_dir / f"{doc_id}.npz"
        if not npz.exists():
            continue
        with np.load(npz) as acts:
            for fld in data.get("labels", []):
                if not fld.get("extracted_present", True):
                    continue
                ps = fld["path_str"]
                spans, ok = {}, True
                for L in layers:
                    key = f"{ps}__layer{L}"
                    if key not in acts:
                        ok = False
                        break
                    v = acts[key].astype(np.float32)
                    if v.ndim == 1:          # stored as a single vector
                        v = v[None, :]
                    spans[L] = v
                if not ok:
                    continue
                rows.append((doc_id, int(fld.get("is_error", 0)), spans))
    if skipped:
        logger.info("Domain filter skipped: %s",
                    ", ".join(f"{k}={v}" for k, v in sorted(skipped.items())))
    return rows


# ---------------------------------------------------------------------------
# Position extractors. Each maps a [n_tokens, hidden] span to one vector.
# ---------------------------------------------------------------------------

def at_first(span):  return span[0]
def at_last(span):   return span[-1]
def at_mean(span):   return span.mean(axis=0)
def at_second(span): return span[1] if len(span) > 1 else span[0]
def at_penult(span): return span[-2] if len(span) > 1 else span[0]


def at_rel(frac):
    """Relative position within the span: 0.0 = first, 1.0 = last."""
    def f(span):
        if len(span) == 1:
            return span[0]
        idx = int(round(frac * (len(span) - 1)))
        return span[idx]
    return f


POSITIONS = {
    "first":      at_first,
    "second":     at_second,
    "rel_25%":    at_rel(0.25),
    "rel_50%":    at_rel(0.50),
    "rel_75%":    at_rel(0.75),
    "penult":     at_penult,
    "last":       at_last,
    "mean":       at_mean,
}


# ---------------------------------------------------------------------------
# Plain LODO at a fixed layer. Pooled out-of-fold AUROC.
#
# NOT nested -- this is a diagnostic about token position, and nesting the
# layer here would confound the two questions. Layers are reported separately
# so no post-hoc maximum is quoted as a headline.
# ---------------------------------------------------------------------------

def lodo_pooled_auroc(X, y, docs, C):
    oof = np.full(len(y), np.nan)
    for d in sorted(set(docs)):
        te = docs == d
        tr = ~te
        if y[tr].sum() == 0 or y[tr].sum() == tr.sum():
            continue
        clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced",
                                 random_state=42)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    m = ~np.isnan(oof)
    if m.sum() == 0 or y[m].sum() in (0, m.sum()):
        return float("nan")
    return roc_auc_score(y[m], oof[m])


def main():
    args = parse_args()
    logger.info("Loading all-tokens spans, layers %s ...", args.layers)
    rows = load_spans(args.domain, args.layers, args.exclude_domains)
    if not rows:
        logger.error("No fields loaded. Check --domain and that this run used "
                     "position: all_tokens.")
        return 1

    docs_all = np.array([r[0] for r in rows])
    y_all = np.array([r[1] for r in rows])
    lens = np.array([len(r[2][args.layers[0]]) for r in rows])

    logger.info("  %d fields, %d documents, %d errors (%.1f%%)",
                len(rows), len(set(docs_all)), y_all.sum(), 100 * y_all.mean())

    # ---- span-length context: the dilution problem ----------------------
    single = (lens == 1).sum()
    logger.info("")
    logger.info("SPAN LENGTHS")
    logger.info("  single-token fields : %d (%.1f%%)  <- first == last by construction",
                single, 100 * single / len(lens))
    logger.info("  median span         : %d tokens", int(np.median(lens)))
    logger.info("  mean span           : %.1f tokens", lens.mean())
    logger.info("  max span            : %d tokens", lens.max())
    hist = Counter(np.clip(lens, 1, 10))
    logger.info("  distribution (10+ bucketed): %s",
                ", ".join(f"{k}:{hist[k]}" for k in sorted(hist)))

    multi = lens > 1
    logger.info("  multi-token subset  : %d fields, %d errors (%.1f%%)",
                multi.sum(), y_all[multi].sum(),
                100 * y_all[multi].mean() if multi.sum() else 0.0)

    results = {"all": {}, "multi": {}}

    for subset_name, mask in (("all", np.ones(len(rows), bool)), ("multi", multi)):
        if mask.sum() < 20 or y_all[mask].sum() < 5:
            logger.warning("subset %s too small, skipping", subset_name)
            continue
        sub = [r for r, m in zip(rows, mask) if m]
        docs = docs_all[mask]
        y = y_all[mask]

        logger.info("")
        logger.info("=" * 78)
        logger.info("POOLED-OOF AUROC BY TOKEN POSITION  --  subset: %s (%d fields)",
                    subset_name.upper(), len(sub))
        logger.info("=" * 78)
        header = f"{'layer':>6}" + "".join(f"{p:>10}" for p in POSITIONS)
        logger.info(header)

        for L in args.layers:
            cells = []
            for pname, pfn in POSITIONS.items():
                X = np.stack([pfn(r[2][L]) for r in sub])
                a = lodo_pooled_auroc(X, y, docs, args.C)
                results[subset_name].setdefault(pname, {})[str(L)] = \
                    None if np.isnan(a) else float(a)
                cells.append(f"{a:10.4f}" if not np.isnan(a) else f"{'---':>10}")
            logger.info(f"{L:>6}" + "".join(cells))

    # ---- reading --------------------------------------------------------
    logger.info("")
    logger.info("=" * 78)
    logger.info("READING")
    logger.info("=" * 78)
    mm = results.get("multi", {})
    if mm:
        firsts = [v for v in mm.get("first", {}).values() if v]
        lasts = [v for v in mm.get("last", {}).values() if v]
        if firsts and lasts:
            bf, bl = max(firsts), max(lasts)
            logger.info("  multi-token fields, best over layers:")
            logger.info("    first token : %.4f", bf)
            logger.info("    last token  : %.4f", bl)
            logger.info("    difference  : %+.4f", bf - bl)
            logger.info("")
            if bf >= bl - 0.03:
                logger.info("  The signal is present at the FIRST token. Steering as")
                logger.info("  prevention is plausible: the model's state already carries")
                logger.info("  the error direction before the value is committed.")
            else:
                logger.info("  The signal is substantially WEAKER at the first token. The")
                logger.info("  model does not yet carry the error direction before writing")
                logger.info("  the value, so steering to prevent would be fighting the")
                logger.info("  mechanism. Steering to INDUCE errors at the last token is")
                logger.info("  still a valid causality test; steering to fix is not.")
    logger.info("")
    logger.info("  Compare the whole positional profile, not just the endpoints. A")
    logger.info("  curve rising monotonically toward 'last' means the signal accrues")
    logger.info("  as the value is written. A flat curve means it is present from the")
    logger.info("  start, which is the better case for intervention.")
    logger.info("")
    logger.info("  NOT nested: layers are reported separately and no maximum here")
    logger.info("  should be quoted as a headline. This is a positional diagnostic.")

    out = Path(args.out) if args.out else Path(
        f"artifacts/{args.domain}/results/token_position_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "domain": args.domain,
        "layers": args.layers,
        "n_fields": len(rows),
        "n_single_token": int(single),
        "median_span": int(np.median(lens)),
        "results": results,
    }, indent=2))
    logger.info("Saved to %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
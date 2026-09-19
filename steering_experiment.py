#!/usr/bin/env python3
"""Is the probe direction causally involved in producing extraction errors?

THE ARGUMENT
------------
Detection establishes correlation: a direction in activation space predicts
which fields are wrong. Causality requires intervention: add the direction to
the residual stream during generation and see whether the error rate moves.

Changed output alone proves nothing -- perturbing activations changes behaviour
trivially. The claim rests on three controls, all measured here:

  1. RANDOM DIRECTION at matched norm and the SAME coefficients. If a random
     vector moves the error rate as much as the probe direction, the result is
     about perturbation, not about this direction.
  2. DOSE RESPONSE across coefficients. A causally implicated direction should
     produce an effect that scales, not one that appears at a single
     hand-picked value.
  3. SIGN. Adding the error direction should increase errors and subtracting
     should decrease them. If both directions degrade output, that is damage,
     not mechanism.

A fourth measure is specific to structured output: PARSE FAILURE RATE. A
malformed record is worse than a wrong field, so the usable coefficient range
is bounded by validity, and the error rate must be read alongside it.

WHAT IT CANNOT ESTABLISH
------------------------
That the probe found *the* error mechanism. A direction can be sufficient to
perturb behaviour without being what the model normally uses. And failure to
REDUCE errors is not a failure of the causal claim -- reduction additionally
requires the model to hold the right information and be misapplying it, which
is a stronger precondition than mere involvement.

COST
----
One full generation pass per (direction, coefficient). With 20 documents and
13 conditions that is ~260 document-generations. Budget accordingly; --domains
and --max-docs restrict it.

Usage:
  python steering_experiment.py --config configs/exp_fresh_alltokens.yaml \
      --probe artifacts/fresh_qwen35_4b_pooled_alltokens/probes/probe_layer18_mid.pkl \
      --layer 18 --exclude-domains finance/10kq \
      --coeffs 0.0 0.1 0.2 0.3 0.4 0.5 -0.1 -0.2 -0.3 -0.4 -0.5
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from probe_extraction.extraction.parser import parse_json_output
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "scripts")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--probe", required=True)
    p.add_argument("--layer", type=int, required=True,
                   help="Probe layer in CONFIG numbering (1-indexed). The hook "
                        "attaches to decoder block layer-1; see the mapping "
                        "note in hf_model.py's docstring.")
    p.add_argument("--coeffs", type=float, nargs="+",
                   default=[0.0, 0.3, 0.4, 0.5, -0.3, -0.4, -0.5])
    p.add_argument("--exclude-domains", nargs="*", default=["finance/10kq"])
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--n-random", type=int, default=1,
                   help="Random control directions. Each is run at every "
                        "nonzero coefficient, so this multiplies cost.")
    p.add_argument("--out-name", default="steering_experiment.json")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Hook. Identical mechanics to steering_smoke.py, which verified that
# coefficient 0 reproduces the unsteered output byte-for-byte.
# ---------------------------------------------------------------------------

class SteeringHook:
    def __init__(self, unit_direction, coeff, ref_norm, dtype, device):
        self.unit = torch.tensor(unit_direction, dtype=dtype, device=device)
        self.coeff = float(coeff)
        self.ref_norm = float(ref_norm)
        self.n_steered = 0

    def __call__(self, module, args, output):
        if self.coeff == 0.0:
            return output
        if isinstance(output, tuple):
            hs, rest = output[0], output[1:]
        else:
            hs, rest = output, None
        # Decode steps only (seq_len == 1). Steering the prefill would alter
        # the model's reading of the document, a different intervention.
        if hs.dim() != 3 or hs.shape[1] != 1:
            return output
        hs = hs + self.unit * (self.coeff * self.ref_norm)
        self.n_steered += 1
        return (hs, *rest) if rest is not None else hs


def load_direction(path, hidden_dim):
    obj = pickle.load(open(path, "rb"))
    for attr in ("weights", "w", "direction"):
        if hasattr(obj, attr):
            v = np.asarray(getattr(obj, attr), dtype=np.float32).ravel()
            if v.size == hidden_dim:
                return v
    if hasattr(obj, "coef_"):
        v = np.asarray(obj.coef_, dtype=np.float32).ravel()
        if v.size == hidden_dim:
            return v
    raise RuntimeError(f"No {hidden_dim}-dim weight vector in {path}")


def unit(v):
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero-norm direction")
    return (v / n).astype(np.float32)


def main():
    args = parse_args()

    from probe_extraction.config import load_config
    from probe_extraction.models.hf_model import HuggingFaceLLM
    from probe_extraction.extraction.prompts import build_extraction_prompt
    from probe_extraction.labeling.matcher import label_extraction
    from importlib import import_module
    ex = import_module("01_extract")

    cfg = load_config(args.config)

    # ---- documents ----
    docs = [d for d in ex.load_benchmark(cfg)
            if d.domain not in args.exclude_domains]
    if args.max_docs:
        docs = docs[:args.max_docs]
    logger.info("Documents: %d (excluding %s)", len(docs), args.exclude_domains)

    # ---- model ----
    llm = HuggingFaceLLM(
        model_name=cfg.model.name, dtype=cfg.model.dtype,
        quantization=cfg.model.quantization, device_map=cfg.model.device_map,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    blocks = llm.model.model.layers
    block_idx = args.layer - 1
    target = blocks[block_idx]
    logger.info("Probe layer %d -> hook on model.model.layers[%d] of %d",
                args.layer, block_idx, len(blocks))

    dtype = next(llm.model.parameters()).dtype
    device = next(llm.model.parameters()).device

    # ---- directions ----
    probe_dir = unit(load_direction(args.probe, llm.hidden_dim))
    rng = np.random.default_rng(cfg.experiment.seed)
    randoms = [unit(rng.normal(size=llm.hidden_dim).astype(np.float32))
               for _ in range(args.n_random)]
    logger.info("Directions: 1 probe + %d random (all unit-normalised, so the "
                "random control is matched in norm by construction)",
                len(randoms))

    # ---- reference norm per document, from the stored activations ----
    acts_dir = cfg.artifacts_path / "activations"

    def ref_norm_for(doc):
        npz = acts_dir / f"{doc.doc_id}.npz"
        if not npz.exists():
            return None
        with np.load(npz) as A:
            ks = [k for k in A.files if k.endswith(f"__layer{args.layer}")]
            if not ks:
                return None
            norms = []
            for k in ks[:200]:
                v = A[k].astype(np.float32)
                if v.ndim > 1:
                    v = v[len(v) // 2]
                norms.append(np.linalg.norm(v))
        return float(np.mean(norms)) if norms else None

    # ---- conditions ----
    conditions = [("probe", probe_dir, c) for c in args.coeffs]
    for i, r in enumerate(randoms):
        conditions += [(f"random{i}", r, c) for c in args.coeffs if c != 0.0]
    logger.info("Conditions: %d, over %d documents = %d generations",
                len(conditions), len(docs), len(conditions) * len(docs))

    # ---- run ----
    per_condition = {}
    t_start = time.time()

    for kind, vec, coeff in conditions:
        key = f"{kind}@{coeff:+.2f}"
        n_fields = n_errors = n_parse_fail = n_docs_ok = 0
        per_doc = {}

        n_dropped = 0
        for doc in docs:
            rn = ref_norm_for(doc)
            if rn is None:
                n_dropped += 1
                logger.warning("  no layer-%d activations for %s -- DROPPED "
                               "from this condition", args.layer, doc.doc_id)
                continue
            text_in = doc.text
            if cfg.extraction.max_input_chars and \
                    len(text_in) > cfg.extraction.max_input_chars:
                text_in = text_in[:cfg.extraction.max_input_chars]
            system_msg, user_msg = build_extraction_prompt(
                schema=doc.schema, document_text=text_in,
                include_schema=cfg.extraction.include_schema,
            )
            prompt = llm.format_chat(system_msg, user_msg)

            hook = SteeringHook(vec, coeff, rn, dtype, device)
            handle = target.register_forward_hook(hook)
            try:
                out = llm.generate(
                    prompt=prompt, max_new_tokens=cfg.model.max_new_tokens,
                    temperature=cfg.model.temperature, top_p=cfg.model.top_p,
                    return_logprobs=False,
                )
            finally:
                handle.remove()

            # ---- score against gold with the pipeline's own matcher ----
            # Call copied from extractor.py:236. Returns a 3-tuple, not a dict.
            parsed, parse_error, _json_text = parse_json_output(out.text)
            if parsed is None:
                n_parse_fail += 1
                per_doc[doc.doc_id] = {"parse_failed": True,
                                       "parse_error": parse_error}
                continue

            # Call copied from 02_label.py:153-161 so scoring is identical to
            # the pipeline's. Any divergence here and the error rates are not
            # comparable to the baseline.
            result = label_extraction(
                doc_id=doc.doc_id,
                domain=doc.domain,
                schema=doc.schema,
                gold=doc.gold,
                extracted=parsed,
                fuzzy_threshold=cfg.labeling.fuzzy_threshold,
                number_tolerance=cfg.labeling.number_tolerance,
            )
            labels = result.labels

            present = [l for l in labels if getattr(l, "extracted_present", True)]
            nf = len(present)
            ne = sum(int(getattr(l, "is_error", 0)) for l in present)
            n_fields += nf
            n_errors += ne
            n_docs_ok += 1
            per_doc[doc.doc_id] = {"n_fields": nf, "n_errors": ne}

        rate = n_errors / n_fields if n_fields else float("nan")
        per_condition[key] = {
            "kind": kind, "coeff": coeff,
            "n_docs_scored": n_docs_ok, "n_parse_failed": n_parse_fail,
            "n_fields": n_fields, "n_errors": n_errors,
            "error_rate": rate,
            "parse_fail_rate": n_parse_fail / max(n_docs_ok + n_parse_fail, 1),
            "per_doc": per_doc,
        }
        logger.info("  %-14s fields %5d  errors %5d  rate %6.2f%%  "
                    "parse-fail %d/%d  (%.0fs elapsed)",
                    key, n_fields, n_errors, 100 * rate,
                    n_parse_fail, n_parse_fail + n_docs_ok, time.time() - t_start)

    # ---- report ----
    base = per_condition.get(f"probe@+0.00")
    logger.info("")
    logger.info("=" * 78)
    logger.info("DOSE RESPONSE")
    logger.info("=" * 78)
    if base is None:
        logger.warning("No coefficient-0 condition; there is no baseline to "
                       "compare against. Add 0.0 to --coeffs.")
    else:
        logger.info("baseline (coeff 0): %.2f%% error over %d fields",
                    100 * base["error_rate"], base["n_fields"])
    logger.info("")
    logger.info("%-14s %8s %10s %12s %12s", "condition", "fields",
                "err rate", "vs baseline", "parse-fail")
    for key in sorted(per_condition, key=lambda k: (per_condition[k]["kind"],
                                                    per_condition[k]["coeff"])):
        r = per_condition[key]
        delta = (r["error_rate"] - base["error_rate"]) if base else float("nan")
        logger.info("%-14s %8d %9.2f%% %+11.2fpp %11.1f%%",
                    key, r["n_fields"], 100 * r["error_rate"],
                    100 * delta, 100 * r["parse_fail_rate"])

    logger.info("")
    logger.info("=" * 78)
    logger.info("READING")
    logger.info("=" * 78)
    logger.info("  A causal result looks like: error rate rising with positive")
    logger.info("  coefficient, falling (or at least not rising) with negative,")
    logger.info("  monotone in magnitude, and the random control flat across the")
    logger.info("  same coefficients.")
    logger.info("")
    logger.info("  If the random control moves as much as the probe direction,")
    logger.info("  the finding is about perturbation and not about this")
    logger.info("  direction. If both signs raise the error rate, that is damage.")
    logger.info("  If nothing moves inside the range where output stays valid,")
    logger.info("  the honest report is a null -- which does not undermine the")
    logger.info("  detection result.")
    logger.info("")
    logger.info("  Read the error rate ALONGSIDE parse-fail: a rate that falls")
    logger.info("  only where a third of records stop parsing is not a gain.")

    out_p = cfg.artifacts_path / "results" / args.out_name
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps({
        "probe": args.probe, "layer": args.layer,
        "block_index": block_idx,
        "coeffs": args.coeffs, "n_random": args.n_random,
        "exclude_domains": args.exclude_domains,
        "n_documents": len(docs),
        "conditions": per_condition,
    }, indent=2))
    logger.info("")
    logger.info("Saved to %s", out_p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
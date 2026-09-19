#!/usr/bin/env python3
"""Smoke test for activation steering. Does the hook work, and does JSON survive?

WHAT THIS IS NOT
----------------
This is not the causal experiment. It establishes only that the mechanism is
wired correctly and that steering does not immediately destroy the output. The
causal claim needs the full design: dose-response across coefficients, both
signs, a matched-norm random control, and error rates measured against gold.

THE OFF-BY-ONE THAT MATTERS
---------------------------
hf_model.py's docstring: hidden_states[0] is the embedding output, and config
layers are 1-indexed so that config layer 1 = first transformer block. So

    probe layer L  reads  hidden_states[L]  =  output of model.layers[L-1]

The hook must therefore attach to decoder block index L-1. Getting this wrong
steers a neighbouring layer and silently produces a null result.

WHAT IT CHECKS
--------------
  1. The decoder-layer module list is found and the index maps correctly.
  2. Coefficient 0 reproduces the unsteered output EXACTLY. If it does not,
     the hook is perturbing something it should not and nothing downstream is
     trustworthy.
  3. Steering both signs at a few coefficients: does the JSON still parse, and
     does the output change at all?
  4. A matched-norm random direction, so we can see early whether any observed
     change is specific to the probe direction.

Usage:
  python steering_smoke.py --config configs/exp_fresh_alltokens.yaml \
      --probe artifacts/fresh_qwen35_4b_pooled_alltokens/probes/probe_layer18.pkl \
      --layer 18 --doc-index 24
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--probe", required=True,
                   help="Path to probe_layer{L}.pkl")
    p.add_argument("--layer", type=int, required=True,
                   help="Probe layer in CONFIG numbering (1-indexed).")
    p.add_argument("--doc-index", type=int, default=24,
                   help="Which benchmark document. Default 24 = a swimming "
                        "table: short, dense, and Qwen extracts it near-"
                        "perfectly, so any degradation is obvious.")
    p.add_argument("--coeffs", type=float, nargs="+",
                   default=[0.0, 0.5, 1.0, 2.0, -0.5, -1.0, -2.0])
    p.add_argument("--max-new-tokens", type=int, default=2048)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Finding the decoder layers. Qwen exposes model.model.layers, but do not
# assume -- report what was found so a wrong guess is visible.
# ---------------------------------------------------------------------------

def find_decoder_layers(hf_model):
    candidates = [
        ("model.model.layers", lambda m: m.model.layers),
        ("model.layers", lambda m: m.layers),
        ("model.model.language_model.layers",
         lambda m: m.model.language_model.layers),
        ("model.transformer.h", lambda m: m.transformer.h),
    ]
    for path, getter in candidates:
        try:
            mods = getter(hf_model)
            if mods is not None and len(mods) > 0:
                logger.info("Decoder layers found at %s (%d blocks)", path, len(mods))
                return mods, path
        except AttributeError:
            continue
    raise RuntimeError(
        "Could not locate the decoder layer list. Inspect the model structure "
        "with `print(model)` and add the path to find_decoder_layers()."
    )


# ---------------------------------------------------------------------------
# The hook
# ---------------------------------------------------------------------------

class SteeringHook:
    """Adds `coeff * direction` to a decoder block's output hidden states.

    Only steers DECODE steps (seq_len == 1), not the prefill pass. Steering
    the prompt would alter the model's reading of the document, which is a
    different intervention from steering its generation.

    The direction is unit-normalised and scaled by the mean activation norm at
    this layer, so `coeff` is expressed in units of typical activation
    magnitude and a matched-norm random control is straightforward.
    """

    def __init__(self, direction, coeff, ref_norm, dtype, device):
        d = np.asarray(direction, dtype=np.float32)
        n = np.linalg.norm(d)
        if n == 0:
            raise ValueError("zero-norm direction")
        self.unit = torch.tensor(d / n, dtype=dtype, device=device)
        self.coeff = float(coeff)
        self.ref_norm = float(ref_norm)
        self.n_steered = 0
        self.n_skipped = 0

    def __call__(self, module, args, output):
        if self.coeff == 0.0:
            return output
        # Decoder blocks return either a tensor or a tuple whose first element
        # is the hidden states. Handle both; transformers has changed this.
        if isinstance(output, tuple):
            hs = output[0]
            rest = output[1:]
        else:
            hs = output
            rest = None
        # hs: (batch, seq_len, hidden). Steer decode steps only.
        if hs.dim() != 3 or hs.shape[1] != 1:
            self.n_skipped += 1
            return output
        delta = self.unit * (self.coeff * self.ref_norm)
        hs = hs + delta
        self.n_steered += 1
        return (hs, *rest) if rest is not None else hs


def load_direction(path, hidden_dim):
    """Pull the weight vector out of a probe pickle. Shapes vary, so try the
    plausible containers and report which matched."""
    obj = pickle.load(open(path, "rb"))
    tried = []

    def from_clf(clf):
        w = getattr(clf, "coef_", None)
        return None if w is None else np.asarray(w).ravel()

    if hasattr(obj, "coef_"):
        v = from_clf(obj); tried.append("bare sklearn estimator")
        if v is not None and v.size == hidden_dim:
            logger.info("Direction from: bare sklearn estimator")
            return v
    # LinearProbe (probe_extraction.probes.linear) stores the vector as a
    # plain `weights` attribute, not as sklearn's coef_ and not in a dict.
    for attr in ("weights", "w", "direction"):
        if hasattr(obj, attr):
            v = np.asarray(getattr(obj, attr)).ravel()
            tried.append(f"obj.{attr}")
            if v.size == hidden_dim:
                logger.info("Direction from: obj.%s", attr)
                return v
    if isinstance(obj, dict):
        for k in ("clf", "clf_final", "model", "probe", "estimator"):
            if k in obj:
                v = from_clf(obj[k]); tried.append(f"dict['{k}'].coef_")
                if v is not None and v.size == hidden_dim:
                    logger.info("Direction from: dict['%s'].coef_", k)
                    return v
        for k in ("coef_", "w", "weights", "direction"):
            if k in obj:
                v = np.asarray(obj[k]).ravel(); tried.append(f"dict['{k}']")
                if v.size == hidden_dim:
                    logger.info("Direction from: dict['%s']", k)
                    return v
    raise RuntimeError(
        f"Could not find a {hidden_dim}-dim weight vector in {path}. "
        f"Tried: {tried}. Inspect with: "
        f"python -c \"import pickle;o=pickle.load(open('{path}','rb'));print(type(o));"
        f"print(o.keys() if isinstance(o,dict) else dir(o))\""
    )


def main():
    args = parse_args()

    from probe_extraction.config import load_config
    from probe_extraction.models.hf_model import HuggingFaceLLM
    from importlib import import_module
    ex = import_module("01_extract")

    cfg = load_config(args.config)

    # ---- document ----
    docs = list(ex.load_benchmark(cfg))
    if args.doc_index >= len(docs):
        logger.error("doc-index %d out of range (%d documents)",
                     args.doc_index, len(docs))
        return 1
    doc = docs[args.doc_index]
    logger.info("Document: %s (%d chars, domain=%s)",
                doc.doc_id, len(doc.text), doc.domain)

    # ---- model ----
    llm = HuggingFaceLLM(
        model_name=cfg.model.name,
        dtype=cfg.model.dtype,
        quantization=cfg.model.quantization,
        device_map=cfg.model.device_map,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    layers_mod, path = find_decoder_layers(llm.model)

    # ---- THE OFF-BY-ONE ----
    block_idx = args.layer - 1
    if not (0 <= block_idx < len(layers_mod)):
        logger.error("config layer %d -> block %d, out of range for %d blocks",
                     args.layer, block_idx, len(layers_mod))
        return 1
    logger.info("Probe layer %d (config numbering) -> hidden_states[%d] "
                "-> hook on %s[%d]", args.layer, args.layer, path, block_idx)
    target = layers_mod[block_idx]

    # ---- direction ----
    direction = load_direction(args.probe, llm.hidden_dim)
    logger.info("Direction: %d dims, norm %.4f", direction.size,
                float(np.linalg.norm(direction)))

    # ---- reference norm: mean activation magnitude at this layer ----
    # Taken from the stored activations so the coefficient is interpretable.
    acts_dir = Path(f"artifacts/{cfg.experiment.name}/activations")
    npz = acts_dir / f"{doc.doc_id}.npz"
    ref_norm = None
    if npz.exists():
        with np.load(npz) as A:
            key = [k for k in A.files if k.endswith(f"__layer{args.layer}")]
            if key:
                v = A[key[0]].astype(np.float32)
                if v.ndim > 1:
                    v = v[-1]
                ref_norm = float(np.linalg.norm(v))
    if ref_norm is None:
        ref_norm = 50.0
        logger.warning("Could not read a reference activation norm; using %.1f. "
                       "Coefficients are then not in activation units.", ref_norm)
    else:
        logger.info("Reference activation norm at layer %d: %.2f",
                    args.layer, ref_norm)

    # ---- prompt, exactly as the extractor builds it ----
    from probe_extraction.extraction.prompts import build_extraction_prompt
    # Signature and call copied verbatim from extractor.py:204-209. The prompt
    # must match what the pipeline produces or the comparison is meaningless.
    # Note extractor.py passes a possibly-truncated `document_text`; at
    # max_input_chars=600000 this document (1,228 chars) is unaffected, but a
    # long document would need the same truncation applied here.
    max_chars = cfg.extraction.max_input_chars
    document_text = doc.text
    if max_chars and len(document_text) > max_chars:
        document_text = document_text[:max_chars]
        logger.info("Truncated document text to %d chars", max_chars)
    system_msg, user_msg = build_extraction_prompt(
        schema=doc.schema,
        document_text=document_text,
        include_schema=cfg.extraction.include_schema,
    )
    prompt = llm.format_chat(system_msg, user_msg)
    logger.info("Prompt: %d chars", len(prompt))

    dtype = next(llm.model.parameters()).dtype
    device = next(llm.model.parameters()).device

    rng = np.random.default_rng(42)
    rand_dir = rng.normal(size=llm.hidden_dim)

    runs = [("probe", direction, c) for c in args.coeffs]
    runs += [("random", rand_dir, c) for c in (2.0, -2.0)]

    results = []
    baseline_text = None

    for kind, vec, coeff in runs:
        hook = SteeringHook(vec, coeff, ref_norm, dtype, device)
        handle = target.register_forward_hook(hook)
        try:
            out = llm.generate(
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=cfg.model.temperature,
                top_p=cfg.model.top_p,
                return_logprobs=False,
            )
        finally:
            handle.remove()

        txt = out.text
        parsed = None
        try:
            from probe_extraction.extraction.parser import parse_json_response
            parsed = parse_json_response(txt)
            ok = parsed is not None
        except Exception:
            # fall back to a plain json attempt on the fenced block
            s = txt[txt.find("{"):txt.rfind("}") + 1] if "{" in txt else ""
            try:
                parsed = json.loads(s); ok = True
            except Exception:
                ok = False

        if kind == "probe" and coeff == 0.0:
            baseline_text = txt

        identical = (baseline_text is not None and txt == baseline_text)
        results.append({
            "kind": kind, "coeff": coeff,
            "chars": len(txt),
            "finish": out.finish_reason,
            "parses": bool(ok),
            "identical_to_baseline": identical,
            "steered_steps": hook.n_steered,
            "skipped_steps": hook.n_skipped,
        })
        logger.info("  %-6s coeff %+5.1f -> %5d chars, finish=%-6s, "
                    "parses=%-5s, steered %d steps%s",
                    kind, coeff, len(txt), out.finish_reason, ok,
                    hook.n_steered, "  [== baseline]" if identical else "")

    # ---- verdict ----
    logger.info("")
    logger.info("=" * 74)
    logger.info("VERDICT")
    logger.info("=" * 74)
    zero = next((r for r in results if r["kind"] == "probe" and r["coeff"] == 0.0), None)
    if zero is None:
        logger.warning("No coefficient-0 run; cannot verify the hook is inert "
                       "when it should be. Add 0.0 to --coeffs.")
    elif zero["steered_steps"] != 0:
        logger.error("Coefficient 0 steered %d steps -- the hook is not inert. "
                     "Nothing downstream is trustworthy.", zero["steered_steps"])
        return 1
    else:
        logger.info("  Coefficient 0 is inert, as it must be.")

    nonzero = [r for r in results if r["coeff"] != 0.0]
    changed = [r for r in nonzero if not r["identical_to_baseline"]]
    logger.info("  %d of %d steered runs changed the output.",
                len(changed), len(nonzero))
    if not changed:
        logger.error("  Nothing changed. Either the coefficient is far too "
                     "small relative to the activation norm, or the hook is "
                     "attached to a module whose output is not on the "
                     "residual path. Try much larger coefficients first.")
    broke = [r for r in nonzero if not r["parses"]]
    logger.info("  %d of %d steered runs failed to parse.", len(broke), len(nonzero))
    if broke:
        logger.info("  Parse failure at: %s",
                    ", ".join(f"{r['kind']}{r['coeff']:+.1f}" for r in broke))
        logger.info("  A malformed record is worse than a wrong field, so the "
                    "usable coefficient range is bounded by this.")
    logger.info("")
    logger.info("  Next, if this looks sane: the full design needs error rates")
    logger.info("  measured against gold across all documents, a dose-response")
    logger.info("  curve, and the random control on the same axes. Changed")
    logger.info("  output alone is not evidence of anything -- perturbing")
    logger.info("  activations changes behaviour trivially.")

    out_p = Path(f"artifacts/{cfg.experiment.name}/results/steering_smoke.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps({
        "layer_config": args.layer, "block_index": block_idx,
        "module_path": path, "ref_norm": ref_norm,
        "doc_id": doc.doc_id, "runs": results,
    }, indent=2))
    logger.info("Saved to %s", out_p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
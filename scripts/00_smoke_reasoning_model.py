"""Stage 0 gate: is this model usable for the reasoning-trace experiments?

Run this BEFORE committing GPU hours to a full extraction. It loads a candidate
model, extracts a handful of SOB documents, and answers the questions that
decide whether the pipeline will work at all:

  1. Does the model emit a `</think>` block? Hybrid-thinking models (Qwen3.x)
     suppress it unless the chat template is asked for it, and the pipeline's
     reasoning capture keys off that literal marker. Without it Stage 7 has
     nothing to attribute -- and it fails silently, after the extraction has
     already cost its hours.
  2. How long are the traces? This sets REASONING_TOKEN_CAP and the storage
     budget for per-token capture.
  3. Does the JSON parse, and do fields land with activations attached?
  4. How many layers and how wide are they? This determines what to put in
     `activations.layers`, which cannot be copied across model families.

Exits non-zero if the model is unusable, so it can gate a job chain.

Usage:
    python scripts/00_smoke_reasoning_model.py --config CFG [--n 3]
    python scripts/00_smoke_reasoning_model.py --config CFG --model-override <hf-id>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from probe_extraction.config import get_hf_token, load_config  # noqa: E402
from probe_extraction.data.sob import SOB  # noqa: E402
from probe_extraction.extraction import Extractor  # noqa: E402
from probe_extraction.models import HuggingFaceLLM  # noqa: E402
from probe_extraction.utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

THINK_END = "</think>"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke-test a reasoning model.")
    p.add_argument("--config", required=True)
    p.add_argument("--n", type=int, default=3, help="Documents to try (default 3).")
    p.add_argument("--model-override", type=str, default=None,
                   help="Try a different HF model id than the config names, so "
                        "candidates can be compared without editing configs.")
    p.add_argument("--thinking-override", choices=["on", "off"], default=None,
                   help="Force the chat template's thinking flag, to check "
                        "whether a model responds to it at all.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_dir=cfg.logging.log_dir,
                  log_name="00_smoke_reasoning_model",
                  log_to_file=cfg.logging.log_to_file)

    model_name = args.model_override or cfg.model.name
    # Fail here rather than letting a placeholder reach the Hub, where it
    # surfaces as a confusing 401 "Repository Not Found" thirty lines into a
    # traceback. This fires when --model-override was dropped -- e.g. passed as
    # an environment variable to a job that runs with --export=NONE.
    if "PLACEHOLDER" in model_name.upper():
        logger.error("Model name is still the placeholder %r.", model_name)
        logger.error("Pass a real id:  --model-override Qwen/Qwen3.5-9B")
        logger.error("Via SLURM:       sbatch run_qwen_smoke_a100.sh Qwen/Qwen3.5-9B")
        logger.error("(An exported MODEL= is stripped by --export=NONE.)")
        return 2

    thinking = cfg.model.enable_thinking
    if args.thinking_override:
        thinking = args.thinking_override == "on"

    logger.info("=" * 70)
    logger.info("SMOKE TEST: %s", model_name)
    logger.info("  enable_thinking = %s", thinking)
    logger.info("=" * 70)

    llm = HuggingFaceLLM(
        model_name=model_name,
        dtype=cfg.model.dtype,
        quantization=cfg.model.quantization,
        device_map=cfg.model.device_map,
        trust_remote_code=cfg.model.trust_remote_code,
        hf_token=get_hf_token(),
        enable_thinking=thinking,
    )

    # Architecture first -- the layer list in a config is model-specific and
    # must not be copied across families.
    hf_cfg = llm.model.config
    n_layers = getattr(hf_cfg, "num_hidden_layers", None)
    hidden = getattr(hf_cfg, "hidden_size", None)
    logger.info("Architecture: %s | %s layers | hidden %s",
                getattr(hf_cfg, "model_type", "?"), n_layers, hidden)
    if n_layers:
        # Evenly spread probe layers, as the existing configs do.
        suggested = sorted({1, *[round(n_layers * f) for f in
                                 (.15, .25, .35, .45, .55, .6, .65, .7, .75,
                                  .8, .85, .9, 1.0)]})
        suggested = [l for l in suggested if 1 <= l <= n_layers]
        logger.info("Suggested activations.layers: %s", suggested)

    benchmark = SOB(benchmark_path=cfg.benchmark_path, split=cfg.data.split,
                    domains=cfg.data.domains or None, max_documents=args.n)

    extractor = Extractor(
        llm=llm,
        layers=cfg.activations.layers,
        position=cfg.activations.position,
        max_new_tokens=cfg.model.max_new_tokens,
        temperature=cfg.model.temperature,
        top_p=cfg.model.top_p,
        include_schema=cfg.extraction.include_schema,
        max_input_chars=cfg.extraction.max_input_chars,
        # Capture per-token states for one layer, just to prove the path works.
        reasoning_token_layers=cfg.activations.layers[len(cfg.activations.layers) // 2:][:1],
        reasoning_token_cap=2048,
    )

    n_think = n_parsed = n_fields = n_with_acts = 0
    trace_chars: list[int] = []
    results = []
    for doc in benchmark:
        r = extractor.extract(doc)
        results.append(r)
        text = r.raw_generated_text or ""
        has_think = THINK_END in text
        n_think += has_think
        if has_think:
            trace_chars.append(text.index(THINK_END) + len(THINK_END))
        n_parsed += r.parse_error is None
        n_fields += len(r.fields)
        n_with_acts += sum(1 for f in r.fields if f.activations)
        logger.info("-" * 70)
        logger.info("%s: %d generated tokens, finish=%s",
                    r.doc_id, r.generated_token_count, r.finish_reason)
        logger.info("  </think> present: %s | JSON parsed: %s | fields: %d",
                    has_think, r.parse_error is None, len(r.fields))
        if r.parse_error:
            logger.info("  parse_error: %s", str(r.parse_error)[:160])
        rt = getattr(r, "reasoning_token_strings", []) or []
        logger.info("  reasoning tokens captured: %d", len(rt))
        logger.info("  first 200 chars: %s", text[:200].replace("\n", " "))

    logger.info("=" * 70)
    logger.info("VERDICT")
    logger.info("  documents tried            : %d", len(results))
    logger.info("  emitted </think>           : %d", n_think)
    logger.info("  JSON parsed                : %d", n_parsed)
    logger.info("  fields extracted           : %d", n_fields)
    logger.info("  fields with activations    : %d", n_with_acts)
    if trace_chars:
        logger.info("  trace length (chars)       : min %d / max %d",
                    min(trace_chars), max(trace_chars))

    ok = True
    if n_think == 0:
        ok = False
        logger.error("  FAIL: no </think> block. The reasoning experiments cannot "
                     "run on this model as configured.")
        if not thinking:
            logger.error("        Try again with model.enable_thinking: true "
                         "(or --thinking-override on).")
        else:
            logger.error("        Thinking was already requested, so this model "
                         "may not support the chat-template flag, or may not be "
                         "a reasoning model at all. Pick another candidate.")
    if n_parsed == 0:
        ok = False
        logger.error("  FAIL: no document produced parseable JSON. Check the "
                     "prompt and max_new_tokens before extracting at scale.")
    if n_fields and n_with_acts == 0:
        ok = False
        logger.error("  FAIL: fields parsed but no activations captured. The "
                     "layer indices in activations.layers are probably invalid "
                     "for this architecture (%s layers).", n_layers)

    logger.info("  RESULT: %s", "USABLE" if ok else "NOT USABLE")
    logger.info("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

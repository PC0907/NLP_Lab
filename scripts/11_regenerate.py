"""Stage 11: REAL regeneration pass (GPU).

Stage 9 measured what selective regeneration would be worth *under an
assumption* -- that a re-asked wrong field is repaired with probability 0.7 and
a re-asked correct field is broken with probability 0.05. Those numbers were
invented. This stage replaces them with measurement: it actually re-asks the
model, and Stage 12 checks what actually happened against gold.

What it does. For every document that already has an extraction, it re-runs the
model `--samples` times at a non-zero temperature and saves the parsed JSON of
each attempt. Nothing is selected here and nothing is compared: this stage only
produces the raw material. Which fields get swapped in, at which budget, and
whether that helped, is Stage 12's job -- so one regeneration pass serves every
budget and every scoring signal, instead of re-running the model per budget.

Two design points worth stating, because the result depends on them:

  * TEMPERATURE MUST BE > 0. The original extraction is greedy, so a greedy
    re-run reproduces it token for token and could never repair anything. The
    second attempt only means something if the sampling differs.

  * THE PROMPT IS IDENTICAL to extraction -- both go through
    extraction.build_prompt_for_document. If the prompts differed, a measured
    "repair rate" would partly reflect the prompt change rather than a genuine
    second attempt.

No hidden states are captured here. The probe has already scored the original
fields; regeneration only needs the values. Skipping capture makes the pass
cheaper and keeps the artifacts small.

Usage:
    python scripts/11_regenerate.py --config CFG [--samples 3] [--resume]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from probe_extraction.config import get_hf_token, load_config  # noqa: E402
from probe_extraction.data.sob import SOB  # noqa: E402
from probe_extraction.extraction.extractor import (  # noqa: E402
    build_prompt_for_document,
)
from probe_extraction.extraction.parser import parse_json_output  # noqa: E402
from probe_extraction.models import HuggingFaceLLM  # noqa: E402
from probe_extraction.utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real regeneration pass (GPU).")
    p.add_argument("--config", required=True)
    p.add_argument("--samples", type=int, default=3,
                   help="Resamples per document (default 3). Three gives both a "
                        "single-sample strategy and self-consistency voting.")
    p.add_argument("--temperature", type=float, default=None,
                   help="Sampling temperature. Defaults to the config's "
                        "selective_regen.resample_temperature. MUST be > 0.")
    p.add_argument("--resume", action="store_true",
                   help="Skip documents already regenerated with enough samples.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap on documents (debug aid).")
    p.add_argument("--shard", type=str, default=None, metavar="I/N",
                   help="Process shard I of N, so several GPU jobs can split "
                        "the work and write disjoint files.")
    return p.parse_args()


def existing_sample_count(path: Path) -> int:
    """How many samples a regeneration file already holds (0 if unusable)."""
    if not path.exists():
        return 0
    try:
        return len(json.loads(path.read_text()).get("samples", []))
    except (ValueError, OSError):
        return 0  # corrupt or partial -> regenerate it


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_dir=cfg.logging.log_dir,
                  log_name="11_regenerate", log_to_file=cfg.logging.log_to_file)

    temperature = (args.temperature if args.temperature is not None
                   else cfg.selective_regen.resample_temperature)
    if temperature <= 0:
        logger.error(
            "Temperature is %.3f. Greedy decoding is deterministic, so a "
            "re-run would reproduce the original extraction exactly and repair "
            "nothing. Use a temperature above 0.", temperature)
        return 2
    if args.samples < 1:
        logger.error("--samples must be at least 1, got %d", args.samples)
        return 2

    artifacts = cfg.artifacts_path
    extractions_dir = artifacts / "extractions"
    regen_dir = artifacts / "regen"
    regen_dir.mkdir(parents=True, exist_ok=True)
    if not extractions_dir.is_dir():
        logger.error("No extractions at %s -- run Stage 1 first.", extractions_dir)
        return 1

    # Only documents that were actually extracted can be regenerated: Stage 12
    # compares against the original values, so a document with no original is
    # meaningless here.
    have_extraction = {p.stem for p in extractions_dir.glob("*.json")
                       if not p.name.startswith("_")}
    logger.info("%d documents have an extraction.", len(have_extraction))

    benchmark = SOB(
        benchmark_path=cfg.benchmark_path,
        split=cfg.data.split,
        domains=cfg.data.domains or None,
        max_documents=args.limit if args.limit is not None else cfg.data.max_documents,
    )

    from probe_extraction.utils.resume import parse_shard
    shard_i, shard_n = parse_shard(args.shard)

    todo = []
    n_no_extraction = n_done = 0
    for idx, doc in enumerate(benchmark):
        if idx % shard_n != shard_i:
            continue
        if doc.doc_id not in have_extraction:
            n_no_extraction += 1
            continue
        if args.resume and existing_sample_count(
                regen_dir / f"{doc.doc_id}.json") >= args.samples:
            n_done += 1
            continue
        todo.append(doc)

    if shard_n > 1:
        logger.info("Shard %d/%d.", shard_i + 1, shard_n)
    logger.info("To regenerate: %d documents x %d samples at temperature %.2f "
                "(%d already done, %d have no extraction).",
                len(todo), args.samples, temperature, n_done, n_no_extraction)
    if not todo:
        logger.info("Nothing to do.")
        return 0

    llm = HuggingFaceLLM(
        model_name=cfg.model.name,
        dtype=cfg.model.dtype,
        quantization=cfg.model.quantization,
        device_map=cfg.model.device_map,
        trust_remote_code=cfg.model.trust_remote_code,
        hf_token=get_hf_token(),
        enable_thinking=cfg.model.enable_thinking,
    )

    from tqdm import tqdm
    n_parse_fail = n_truncated = 0
    run_start = time.perf_counter()

    for doc in tqdm(todo, desc="Regenerating", total=len(todo)):
        # Identical to the prompt used at extraction time -- see the helper.
        prompt = build_prompt_for_document(
            llm, doc,
            include_schema=cfg.extraction.include_schema,
            max_input_chars=cfg.extraction.max_input_chars,
        )

        samples = []
        for k in range(args.samples):
            t0 = time.perf_counter()
            try:
                out = llm.generate(
                    prompt=prompt,
                    max_new_tokens=cfg.model.max_new_tokens,
                    temperature=temperature,
                    top_p=cfg.model.top_p,
                    return_logprobs=False,
                )
                parsed, parse_error, _ = parse_json_output(out.text)
                finish_reason = out.finish_reason
                n_tokens = len(out.generated_token_ids)
            except Exception as e:  # one bad sample must not lose the run
                logger.exception("Sample %d failed for %s: %s", k, doc.doc_id, e)
                parsed, parse_error, finish_reason, n_tokens = (
                    None, f"generation error: {type(e).__name__}: {e}", "error", 0)

            if parse_error is not None:
                n_parse_fail += 1
            if finish_reason == "length":
                n_truncated += 1

            samples.append({
                "index": k,
                "parsed_json": parsed,
                "parse_error": parse_error,
                "finish_reason": finish_reason,
                "generated_token_count": n_tokens,
                "elapsed_seconds": round(time.perf_counter() - t0, 3),
            })

        payload = {
            "doc_id": doc.doc_id,
            "domain": doc.domain,
            "temperature": temperature,
            "n_samples": args.samples,
            "model": cfg.model.name,
            "samples": samples,
        }
        # Write per document, so a wall-clock kill keeps everything finished
        # so far and --resume picks up from there.
        (regen_dir / f"{doc.doc_id}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False))

    elapsed = time.perf_counter() - run_start
    total = len(todo) * args.samples
    logger.info("=" * 70)
    logger.info("Regenerated %d documents (%d samples) in %.1f min.",
                len(todo), total, elapsed / 60)
    logger.info("  parse failures : %d / %d (%.1f%%)",
                n_parse_fail, total, 100 * n_parse_fail / max(total, 1))
    logger.info("  hit max_new_tokens: %d / %d (%.1f%%) -- these are excluded "
                "from Stage 12, since a cut-off generation's JSON is a parser "
                "reconstruction rather than something the model committed to.",
                n_truncated, total, 100 * n_truncated / max(total, 1))
    logger.info("Regenerations: %s", regen_dir)
    logger.info("Next: python scripts/12_regen_evaluate.py --config %s", args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())

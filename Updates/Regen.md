# Verified Selective-Regeneration Result (ExtractBench 4B)

**Status:** REPRODUCED and PERSISTED (2026-07). The headline safe-override number
previously existed only in a lost log; it is now re-derived from artifacts on disk
and recorded here with the exact command.

## Result

Probe-gated safe-override selective regeneration, held-out (leave-one-doc-out tau
selection), LENIENT scoring mode:

| Metric | Value |
|---|---|
| Held-out net improvement | **+48** (fixed 60, broke 12) |
| Full-set best net | +50 at tau=0.75 |
| Override precision (at operating point) | **0.85** |
| tau chosen (held-out) | mostly 0.75 |
| Naive "regenerate everything" (tau=0) | net **-237** (net-NEGATIVE) |

The probe turns a strongly net-negative "regenerate everything" policy (-237) into
net +48 by concentrating regeneration on genuinely-wrong fields at 85% precision.

## The probe is causally responsible

Same cache, same fields, same regenerated values -- only the gating probe differs:

| Gating probe | Held-out net | Precision |
|---|---|---|
| qwen35_4b_pooled (correct) | **+48** | 0.85 |
| qwen35_4b_pymupdf (wrong parse) | +6 | ~0.5 |

The entire regeneration gain comes from the probe deciding WHERE to regenerate.
This is direct evidence that detection quality drives regeneration value.

## Exact reproduction command

    python scripts/safe_override_rescore.py \
        --domain qwen35_4b_pooled_alltokens \
        --cache artifacts/qwen35_4b_pooled_alltokens/results/regen_cache_pooled_v2.json \
        --probe-path artifacts/qwen35_4b_pooled/probes/probe_layer18.pkl \
        --layer 18

Key detail that caused an earlier failed reproduction: the probe MUST be the
`qwen35_4b_pooled` probe (not `qwen35_4b_pymupdf`). Wrong probe -> +6, not +48.

## Honest caveats

- The +48 is in LENIENT scoring mode (skips hard-to-score long-text fields like
  legal clauses). STRICT mode nets ~0. Report the mode explicitly.
- Per fixability.json: of 7743 errors, 4703 are "fixable" (gold in text), but many
  are annotation-form mismatches (extraction is a SUPERSET of gold, e.g.
  "AT&T Bell Laboratories, Holmdel" vs gold "AT&T Bell Laboratories"). So the gold
  standard counts some reasonable extractions as errors; the regeneration ceiling
  is shaped partly by annotation strictness, not only model quality.
- "PromptPort" (arXiv 2601.06151), cited in the script as the source of the
  safe-override policy, remains UNVERIFIED -- confirm it exists before citing.

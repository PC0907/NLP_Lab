# Weekly Update — Model Scaling (2B complete, 9B staged) + Selective-Regeneration Literature

## 1. The 2B nested-LODO result (clean domains) — complete

The 2B (Qwen3.5-2B, 24 layers, hidden 2048) was already extracted (22 docs) and labelled
from a prior run; this week it was evaluated under nested LODO on the **clean comparable
domain set** (academic + credit + swimming; 10kq excluded — see §2 for why this is
mandatory and §4 for the fix that makes it real).

**Result (nested LODO, unbiased layer selection, 10kq excluded):**

| Metric | Value |
|---|---|
| Documents | 17 (academic 2, credit 10, swimming 5) |
| Fields | 625 (175 errors, 28.0%) |
| Per-fold AUROC | **0.907 ± 0.100** (17 outer folds) |
| Pooled-OOF AUPRC | 0.61 |
| Layer selected | **14 in 16/17 folds** (12 once) |

**Reading.**
- **Layer selection is near-unanimous (layer 14, 16/17 folds).** This mirrors the 4B, which
  selected layer 16 unanimously. On *relative* depth the two line up: 2B layer 14 ≈ 58%,
  4B layer 16 = 50% — both mid-network. The trust signal lives at a consistent relative
  depth across model size, and the probe auto-selects it without peeking. This is the
  strongest form of the "signal generalises across scale" claim.
- **The headline number is the per-fold mean (0.907), reported with its std.** It is buoyed
  by saturated credit folds (13 fields/doc → AUROC frequently hits 1.000 by small-sample
  separation, not leakage); the realistic low-end folds (0.67, 0.76, 0.79) confirm the
  spread of a genuine, imperfect probe rather than a leaking one.
- **AUROC ~0.9 but AUPRC ~0.61** is the rare-class reality: ranking is strong, but precision
  when acting on the top-scored fields is moderate (errors are 28% of fields). For the
  regeneration use-case (acting on the signal), AUPRC is the governing number and is
  reported alongside AUROC, not hidden behind it.

**Honest caveats.**
- This is **not yet a matched comparison to the 4B's 0.879** — different document
  populations (2B truncated 4 academic docs the 4B retained; see §3). A tight scaling claim
  needs both models scored on the intersection of documents both parsed.
- Per-fold variance is large (±0.10), the same small-data reality flagged throughout the
  project. The mean is solid; the variance is stated openly.

---

## 4. Pipeline fix: domain filtering in the LODO scripts

The LODO scripts (`05_lodo_cv.py`, `05b_nested_lodo.py`) globbed **every** label file in the
artifacts directory and ignored the config's domain list. Excluding a domain via a separate
config therefore did **nothing** — the 10kq labels were still loaded. This is exactly the
silent-contamination failure mode: a "clean" number that quietly includes the noise.

**Fix:** added `--include-domains` / `--exclude-domains` flags that filter on each label
file's top-level `domain` key, plus an `--out-name` flag so a filtered run does not overwrite
the all-domain result, and a log line proving how many docs were dropped
(`Domain filter skipped: finance/10kq=5`). No flags = original behaviour, so the 4B pipeline
is unaffected. `05b` also now reports a **pooled-OOF AUROC** (one AUROC over all held-out
fields) alongside the per-fold mean — a more stable "all fields" number that avoids the
small-fold saturation.

---

## 5. Model scaling: 9B staged, blocked on disk quota

The 9B (Qwen3.5-9B, 32 layers, hidden 4096, GatedDeltaNet/attention hybrid, MoE-free at the
layer level) run is fully prepared:

- **Config corrected.** The drafted config captured only layers [14,16,18,20,22] on the stale
  assumption that signal lives at 16–18. This week's data contradicts that (2B peaked at 14,
  4B re-run selected 12), so the band was widened to **[8,12,14,16,18,20,24,28]** —
  deliberately including the hybrid architecture's full-attention layers (4,8,...,28) and
  bracketing ~25–87% depth so the peak cannot be clipped. Clipping would force a full A100
  re-extraction.
- **Sbatch written** to mirror the working 2B script, with A100-specific changes
  (A100short, `--export=NONE`, `setup_env_a100.sh`, `expandable_segments`, 128G mem), and
  the nested-LODO-with-10kq-excluded step appended so the 9B number is born comparable.

**Blocker:** home directory is at the hard quota limit (110G/110G, in 6-day grace). The 9B
weights (~19GB at bf16 — normal for a 9B model) cannot be downloaded until space is freed or
a scratch/project filesystem is configured for the HF cache. The HF cache (12G, a failed 9B
partial) and artifacts (3.8G) account for only ~16G of the 110G — the bulk is elsewhere in
home and needs to be located. **Resolving disk quota is the prerequisite for the 9B run, and
takes priority over launching it** (grace-period write-lock risk).

---





---

## 7. Status of headline numbers

| Quantity | Number | Status |
|---|---|---|
| Probe nested LODO, 4B (28-doc, all domains) | 0.879 ± 0.11 | Established headline (prior) |
| Probe nested LODO, 2B (17-doc, clean) | **0.907 ± 0.10** | New, clean, layer-stable |
| 2B pooled-OOF AUPRC | 0.61 | Rare-class governing metric |
| 4B matched-domain re-run | layer 12, ~0.80 | Under investigation (§3) |
| 4B vs 2B clean error rate | ~13% vs 28% | Error rate drops with scale |
| 9B nested LODO | — | Staged, blocked on disk quota |
| Task 1 three-signal coefficients | — | Built; results not yet read |

---

## Next steps

1. **Free disk quota** (locate the ~94G in home outside cache/artifacts; configure scratch
   for HF cache) — prerequisite to everything GPU, and to avoid grace-period write-lock.
2. **Reconcile the 4B matched-domain re-run** against the stored nested result (§3); decide
   whether the 0.879 came from this dir or another, and whether 10kq-exclusion legitimately
   shifts the 4B layer selection. Then write the 2B-vs-4B comparison.
3. **Read the Task 1 three-signal regression coefficients** (already run; CPU; the
   professor's explicit ask — expected probe-dominant, logprob small, hand-crafted ~0).
4. **Launch 9B** once disk is resolved; compare on the document intersection across 2B/4B/9B
   (truncation hits different docs per model).
5. **Read REDD (2511.02711) in full** — nearest white-box method; confirm how our
   selective-regeneration framing differs. Verify PromptPort exists.

## Principle this week

The scaling signal is holding (2B reproduces the 4B's mid-network, layer-stable probe), but
the honest comparison is gated on matched documents and a reconciled 4B number — not yet
written. Two infrastructure realities (the silent domain-filter bug; the disk quota) were
surfaced and handled/flagged before they could corrupt a result or lock the account. As
before, the substance is separating real signal from artefact before reporting.

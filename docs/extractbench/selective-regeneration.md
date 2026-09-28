# Selective Regeneration — Design, Implementation, and Findings

_Probe-based trust signals project · Team #9 · ExtractBench · as of 10 June 2026_

---

## 0. The idea in one paragraph

The probe flags fields that are likely extraction errors. Selective regeneration
asks: if we re-extract **only** the flagged fields (spending compute selectively
rather than re-extracting everything), can we fix errors without breaking the
fields that were already correct? The end goal is a **cost-quality curve** —
net errors corrected vs number of regenerations — where probe-guided
prioritisation should beat logprob-guided, random, and approach an oracle.

---

## 1. Files created

| File | Role |
|---|---|
| `scripts/07_regen_single.py` | Atomic single-field test harness. Picks one probe-flagged field, builds a targeted re-extraction prompt, regenerates it, compares to gold. Used to validate the mechanism before scaling. Also home of the `$ref`-aware `schema_for_path` walker reused elsewhere. |
| `scripts/08_fixability_filter.py` | Value-in-text fixability filter. For each error, checks whether the gold value is present in the extracted document text. Errors whose answer is absent are unfixable by regeneration (and likely not real model errors). Defines the fixable candidate set and quantifies extraction-loss. |
| `scripts/09_regen_sweep.py` | Full sweep harness. Phase 1 (`--regenerate`, GPU): correction-with-context regeneration of candidate fields, cached to disk. Phase 2 (default, CPU): threshold sweep over probe scores → cost-quality curve (probe/logprob/random/oracle). |
| `scripts/rescore_regen.py` | Re-scores a cached regeneration run under STRICT and LENIENT matchers side by side. Built after discovering that strict exact-match scoring counts semantically-correct values as errors. |
| `run_regen_credit.sh`, combined temp-0 sbatch | SLURM wrappers for the GPU regeneration phase. |

Caches produced (per domain, under `artifacts/<domain>/results/`):
- `regen_cache.json` — temperature 0.7 run.
- `regen_cache_t0.json` — temperature 0.0 run.
- `fixability.json` — value-in-text results.

---

## 2. First test: atomic single-field regeneration (07)

**Goal:** before building anything, confirm the smallest unit works — pick one
flagged field, re-extract it, see if it can be fixed.

**What we built:** `07_regen_single.py` picks the highest-probe-score true error
among SHORT-gold fields (to avoid long legal clauses for the first test), loads
the document text + schema from the benchmark (same loader as extraction, so the
model sees exactly what it saw originally), resolves the field's schema
description via a `$ref`-aware walker, and builds a targeted prompt.

**Problems hit and fixed, in order:**
1. **Candidate/document mismatch.** The pooled labels span all domains but the
   benchmark loads one domain — so a candidate could be found in the labels but
   its document not loaded. Fix: run per-domain (labels + benchmark aligned).
2. **Probe location.** The probe is pooled; per-domain configs have no probe.
   Fix: added `--probe-path` to point at the pooled probe while using a
   per-domain config.
3. **Empty schema node.** The schema walker returned `{}` because ExtractBench
   schemas use JSON-Schema `$ref` pointers (`#/$defs/...`). Fix: a `$ref`
   resolver, so field descriptions populate correctly. (This is domain-general
   and reused by the sweep.)

**The key finding from the first real generation:** naive single-field
re-extraction of a deeply-nested field **returned `null`**. Stripped of all
context, the model could not locate `cash_flow_statement.shares_issued.0.unit`
in a 280k-character document and gave up. (That field was also a poor test —
likely a gold-quality issue: gold said `USD`, model said `count`, and "shares
issued" is plausibly a count.)

**Design consequence:** blind single-field re-extraction is the weakest possible
regeneration. The model needs *locating context*. This motivated the shift to
**correction-with-context**.

We also switched the demonstration domain from finance to **academic**, then used
**credit** for the main run, because financial fields require domain expertise to
adjudicate while academic/credit fields (names, dates, amounts) are humanly
checkable.

---

## 3. What we did next: correction-with-context + fixability gate

### 3.1 Fixability filter (08)
For every error, search the extracted document text for the gold value (same
matcher logic as labelling: numeric-aware, normalised substring, word-overlap
for long values). Result:

| Domain | Errors | Fixable (gold in text) |
|---|---|---|
| swimming | 13 | 62% |
| credit | 65 | 55% |
| academic | 25 | 72% |
| 10kq | 7,634 | 61% (dominated by a dual-period omission artefact; excluded) |

**Finding:** roughly **40% of errors have gold values not present in the
extracted text** — extraction-loss or gold errors, unfixable by regeneration and
not fair to count against the probe. This also doubles as a parsing-quality
metric.

**Important nuance:** "gold value in text" is necessary but
NOT sufficient for "genuine fixable model error". On swimming, most flagged
errors turned out to be gold gaps (the `records` field), a gold typo, a gold
length error, or matcher artefacts — not model errors. So the genuinely
regeneration-addressable set is smaller than the fixable count suggests.

### 3.2 Correction-with-context regeneration (09, phase 1)
the prompt shows the model:
- the **parent object** of the target field (its siblings), with the target
  field blanked to `<FILL_THIS>` — giving locating context;
- the field's key, schema description, and type;
- the full document text;
- and asks for ONLY the target field's value.

The original (possibly wrong) value is **hidden** to avoid anchoring the model
to its own mistake.

**Validation:** on the first GPU run (credit, 93 candidates), the cache had
**zero nulls** — real values came back (`'ABN AMRO Bank N.V.'`,
`'Adobe Systems Incorporated'`, `'August 9, 2000'`, `91532846.72`, ...). The
correction-with-context design solved the null problem. Mechanism validated.

---

## 4. Metrics

Each regenerated field is classified by comparing its state before vs after:

| | Regen now CORRECT | Regen now WRONG |
|---|---|---|
| **Was an ERROR** | **fixed** (+1) | **still_wrong** (0) |
| **Was CORRECT** | **kept_ok** (0) | **broke** (−1) |

- **fixed** — error repaired (the win).
- **broke** — a correct field damaged (the cost).
- **still_wrong** / **kept_ok** — no change (0).
- **Net = fixed − broke.**

**Cost-quality curve:** at each budget (number of fields regenerated), pick
fields in a strategy's priority order and sum their deltas. Strategies:
- **probe** — highest probe P(error) first.
- **logprob** — lowest min-logprob first (most uncertain).
- **random** — shuffled (control).
- **oracle** — actual errors first (upper bound, uses gold knowledge).

The question: does **probe** recover more net corrections per regeneration than
logprob/random, approaching oracle?

---

## 5. Scoring problem discovered (strict vs lenient)

The first sweeps were **net-negative** (credit −11, academic −19). Per-field
inspection revealed this was largely a **matcher artefact**, not real damage:

- `currency: "USD"` regenerated as `"Dollars"` — semantically identical, exact
  match rejects it → counted as "broke".
- `governing_law: "New York"` vs `"State of New York"` — containment, rejected.
- Long legal clauses (`borrowing_request`, `use_of_proceeds`,
  `authorized_officer_definition`) — paraphrases that exact match cannot score
  at all.

**Temperature was not the cause.** Re-running at temperature 0.0 changed 22 of 93
values but produced the **identical** strict score (fixed=1, broke=12) — the
model faithfully extracts the document's wording (e.g. "Dollars"), which is
correct but differs from gold's canonical form. So the lever is the **matcher**,
not sampling temperature.

We built `rescore_regen.py` to score under both:
- **strict** — original AUTO matcher.
- **lenient** — AUTO + currency/boolean synonym groups + substring containment,
  with long-clause fields excluded from scoring.

**Methodological honesty:** the matcher leniency was decided *after* seeing the
strict result, which is exactly the cherry-picking risk our supervisor warns
about. The justification is by **audit, not by outcome**: of the 12 strict
"breakages" on credit, 11 (~92%) were matcher artefacts, far above a 25%
"matcher-problem" threshold — so a synonym-aware matcher is warranted
independent of the probe. We report BOTH scorings and never silently swap.

---

## 6. Results

### Credit (temperature 0.7; lenient identical-outcome at temp 0)

| | STRICT | LENIENT |
|---|---|---|
| candidates scored | 93 | 68 |
| fixed | 1 | 9 |
| broke | 12 | 1 |
| **net (regenerate all)** | **−11** | **+8** |

Lenient cost-quality curve (net corrections at budget):

| budget | probe | random | oracle |
|---|---|---|---|
| 6 | 5 | −1 | 6 |
| 17 | 8 | 0 | 9 |
| 34 | 8 | 2 | 9 |
| 68 | 8 | 8 | 8 |

**Reading:** under fair (lenient) scoring, regeneration is **net-positive (+8)**:
9 errors fixed, 1 broken. The **probe orders candidates near-optimally** — at a
small budget (top 17 flagged fields) the probe recovers all 8 net corrections
while random recovers 0; the probe tracks the oracle (5 vs 6, 8 vs 9). This is
the deliverable's central result: **selective, probe-guided regeneration is
worthwhile; undirected (random) regeneration is not.**

### Academic (temperature 0.0)

| | STRICT | LENIENT |
|---|---|---|
| candidates scored | 70 | 70 |
| fixed | 1 | 9 |
| broke | 20 | 13 |
| **net (regenerate all)** | **−19** | **−4** |

Lenient cost-quality curve:

| budget | probe | random | oracle |
|---|---|---|---|
| 7 | 6 | −1 | 7 |
| 17 | 8 | −3 | 8 |
| 35 | 5 | −4 | 5 |
| 70 | −4 | −4 | −4 |

**Reading:** lenient scoring improves academic (−19 → −4) and recovers the same
9 genuine fixes, but **breakage stays high (13)** — unlike credit, the lenient
matcher did not explain away academic's breakage. Two possibilities, NOT yet
resolved:
1. Academic has its own class of matcher artefact (author-name formatting,
   title punctuation, etc.) that the current synonym/containment rules miss.
2. Regeneration genuinely damages more correct fields in academic.

**This requires a per-field audit of the 13 academic breakages (open item).**

**However — even at net −4, the probe ordering is excellent:** at budget 7,
probe +6 vs random −1; at budget 17, probe +8 vs random −3; the probe tracks the
oracle. So **regenerating only the probe's top picks is net-positive (+8); the
−4 only arises from regenerating everything.** This is precisely the argument for
*selective* (not wholesale) regeneration.

---

## 7. Problems encountered (chronological)

1. **GPU-on-CPU fallback (recurring).** Twice the model loaded on CPU
   (`device=cpu`) because a second job held the GPU on the same node, or the
   SLURM GPU request did not attach. Fix: one GPU job at a time via sbatch;
   confirm `device=cuda:0` in the first minute (logs go to **stderr**, not
   stdout); use `--gres=gpu:1`.
2. **Null regeneration** on blind single-field prompts → solved by
   correction-with-context (siblings as locating context).
3. **Strict-matcher false breakage** → solved by lenient (audit-justified)
   scoring; reported alongside strict.
4. **Temperature is not the lever** — temp 0 vs 0.7 gave identical outcomes; the
   matcher was the real issue.
5. **Academic breakage unexplained by lenient scoring** — open.
6. **Tiny fixable sets** — credit ~9, academic ~9 genuine fixes; curves are
   coarse and noisy. This is a proof-of-concept, not a smooth publication figure.

---

## 8. Design choices (and rationale)

- **Correction-with-context, not blind re-extraction.** Siblings give locating
  context; hiding the original avoids anchoring. Chosen after blind re-extraction
  returned null.
- **Fixability gate.** Only regenerate (and only fairly score) errors whose gold
  value is in the extracted text. Regeneration cannot recover what extraction
  lost (both read the same flattened text — the model never sees the PDF).
- **Two-phase (GPU generate → cache → CPU sweep).** The expensive generation runs
  once and is cached; the sweep/scoring is re-runnable for free, which is what
  let us re-score strict vs lenient without new GPU.
- **Report strict AND lenient.** Matcher leniency is a post-hoc, audit-justified
  change; showing both keeps it honest.
- **Selective, not wholesale.** The headline is the curve at small budget
  (regenerate the probe's top picks), not net-if-regenerate-everything.
- **Same model, same text as extraction.** Regeneration adds no new information
  source — it re-asks more focusedly on the same flattened document text.

---

## 9. Open items / next steps

- **Audit the 13 academic breakages** — matcher artefact vs genuine damage?
  Determines whether academic also flips net-positive.
- **Add the logprob baseline to the lenient curve** (rescore script currently
  shows probe/random/oracle; logprob comparison completes the story).
- **Per-field audit counts** for the matcher-leniency justification (record the
  92%-artefact figure properly, per the pre-registration protocol).
- **Restrict to canonical-valued fields** as the cleanest evaluation set; treat
  long clauses separately (they cannot be exact-match scored).
- **Larger fixable set** — combine domains, or add more documents, for a smoother
  curve.

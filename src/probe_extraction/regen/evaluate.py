"""What a regeneration actually did to a document, field by field.

Stage 7 priced selective regeneration under an assumption: a re-asked wrong
field becomes right with probability 0.7, a re-asked right field breaks with
probability 0.05. Those numbers were invented. This module is how they get
replaced by measurement -- it takes what the model actually said on a second
attempt (Stage 8), swaps it into the original record one field at a time, and
re-labels against gold to see what really happened.

It lives in the package rather than inside scripts/09_regen_evaluate.py for two
reasons: the accounting is the paper's central result and deserves direct unit
tests, and the per-document worker has to be importable by name so it survives
being shipped to a joblib subprocess.

The outcome vocabulary, used everywhere downstream:

    repaired     was wrong, is now right
    damaged      was right, is now wrong
    unchanged    the label did not move (including the case where the resample
                 returned the identical value -- tracked separately as the
                 `identical` status, because "the model repeated itself" is a
                 different finding from "the model changed its mind and was
                 still wrong")
    unavailable  the resample had no value at that path; nothing was swapped
    lost         the swap changed the record's shape and that leaf no longer
                 exists to be labeled
"""

from __future__ import annotations

import json
import sys
from typing import Any, Callable, Sequence

from probe_extraction.labeling.matcher import label_extraction
from probe_extraction.utils.jsonpath import json_get, with_replacements

Labeler = Callable[[Any], dict[str, int]]

# The matcher walks gold and extracted in parallel by recursion, and SOB records
# nest deeply enough to blow CPython's default 1,000-frame limit -- Stage 2
# raises it for the same reason.
#
# It is set HERE, at module scope, and not only in the calling script, because
# the per-document pass runs in joblib worker processes. Those are fresh
# interpreters: they do not inherit the parent's recursion limit, so a limit set
# in the script applies to the parent alone and every worker still runs at 1,000.
# A worker must import this module to unpickle `measure_document`, so setting it
# here guarantees the limit is in place before any labeling happens, in whatever
# process does it.
RECURSION_LIMIT = 20000
if sys.getrecursionlimit() < RECURSION_LIMIT:
    sys.setrecursionlimit(RECURSION_LIMIT)


# ---------------------------------------------------------------------------
# Which resamples are allowed to count
# ---------------------------------------------------------------------------

def usable_samples(payload: dict) -> tuple[list[Any], dict[str, int]]:
    """The resamples we may use, plus a tally of why the others were dropped.

    A sample that hit max_new_tokens is EXCLUDED. Its JSON exists only because
    the parser repaired a cut-off generation; treating that reconstruction as
    the model's answer would credit or blame the model for text it never
    produced.
    """
    stats = {"total": 0, "truncated": 0, "parse_error": 0, "no_json": 0, "usable": 0}
    out = []
    for s in payload.get("samples", []):
        stats["total"] += 1
        if s.get("finish_reason") == "length":
            stats["truncated"] += 1
            continue
        if s.get("parse_error") is not None:
            stats["parse_error"] += 1
            continue
        parsed = s.get("parsed_json")
        # Records are JSON objects; anything else cannot be walked against the
        # gold annotation and is not a usable extraction.
        if not isinstance(parsed, dict):
            stats["no_json"] += 1
            continue
        stats["usable"] += 1
        out.append(parsed)
    return out, stats


def select_value(samples: Sequence[Any], path: Sequence,
                 strategy: str) -> tuple[bool, Any]:
    """What the regeneration offers for one field, under one strategy.

    `first`        the first usable resample -- the honest k=1 deployment cost.
    `vote`         the plurality value across usable resamples (self-consistency).
    `vote_strict`  the same, but only when the resamples actually AGREE: at
                   least two of them, and a strict majority. Otherwise the field
                   is left alone.

    `first` fixes the sample and then reads the path: if that one resample has
    nothing there, the field is unavailable. It must not fall through to a
    later sample, or `first` would quietly become best-of-k and overstate what a
    single regeneration call buys.

    `vote_strict` exists because a resample that disagrees with itself is a poor
    reason to overwrite an answer that may well have been right. Requiring
    consensus spends less of the budget but should break fewer correct fields --
    whether it actually does is measured, not assumed.
    """
    if not samples:
        return False, None
    if strategy == "first":
        return json_get(samples[0], path)
    if strategy in ("vote", "vote_strict"):
        counts: dict[str, int] = {}
        seen: dict[str, tuple[int, Any]] = {}
        n_present = 0
        for s in samples:
            ok, v = json_get(s, path)
            if not ok:
                continue
            n_present += 1
            # Values can be unhashable (lists, dicts), so vote on a canonical
            # serialization rather than the value itself.
            k = json.dumps(v, sort_keys=True, default=str)
            counts[k] = counts.get(k, 0) + 1
            seen.setdefault(k, (len(seen), v))
        if not counts:
            return False, None
        # Most votes; ties go to whichever value appeared first, so the result
        # does not depend on dict iteration order.
        best = min(counts, key=lambda k: (-counts[k], seen[k][0]))
        if strategy == "vote_strict":
            # Two agreeing samples minimum, and a strict majority of those that
            # had anything to say. A lone sample can never form a consensus, so
            # it is left alone rather than treated as unanimous.
            if counts[best] < 2 or counts[best] * 2 <= n_present:
                return False, None
        return True, seen[best][1]
    raise ValueError(f"unknown strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------

def make_labeler(*, doc_id: str, domain: str, schema: dict, gold: dict,
                 fuzzy_threshold: float, number_tolerance: float,
                 mode_params: dict) -> Labeler:
    """A function mapping an extracted record to {path_str: is_error}.

    Built from the config's own labeling settings via Stage 2's mode table: the
    before and after labels must come from identical rules, or the measured
    difference would partly be a change of definition rather than a change of
    answer.
    """
    def labeler(extracted: Any) -> dict[str, int]:
        res = label_extraction(
            doc_id=doc_id, domain=domain, schema=schema, gold=gold,
            extracted=extracted,
            fuzzy_threshold=fuzzy_threshold, number_tolerance=number_tolerance,
            **mode_params,
        )
        return {lab.path_str: lab.is_error for lab in res.labels}
    return labeler


def errors_over_scored(after: dict[str, int], before: dict[str, int]) -> int:
    """Error count over the FIXED scored field set.

    Two conventions, both deliberate:
      * a scored field missing from `after` (the swap changed the record's
        shape) keeps its original label -- the swap is neither credited nor
        blamed for a field it destroyed. These are counted as `lost` and
        reported separately.
      * leaves the swap CREATED are ignored. The scored set is what the probe
        ranked; it has to stay fixed or budgets would not be comparable.
    """
    return sum(after.get(ps, y0) for ps, y0 in before.items())


# ---------------------------------------------------------------------------
# Per-document measurement
# ---------------------------------------------------------------------------

def evaluate_document(original: Any,
                      scored: list[tuple[str, list, int]],
                      samples: Sequence[Any],
                      strategy: str,
                      labeler: Labeler,
                      joint_sets: dict[str, list[str]] | None = None) -> dict:
    """Measure every single-field swap for one document, plus any joint checks.

    `scored` is (path_str, path, label_before) for the fields the probe ranked.
    Each field is swapped ALONE, against the untouched original, and the effect
    recorded as `net_delta`: the change in this document's error count over the
    scored set. Measuring one at a time is what lets any budget be priced by
    summing the fields it flags, instead of re-labeling the corpus once per
    budget -- and because net_delta is computed over the whole scored set, it
    already charges a swap for any collateral damage to its neighbours.

    `joint_sets` ({name: [path_str]}) additionally swaps a whole flagged set at
    once and re-labels it in one go, so the caller can check how far that
    additive shortcut drifts from the truth.

    Returns per-field records in the order given, so they can be concatenated
    across documents and indexed by the same masks the flagging produces.
    """
    before = {ps: y for ps, _p, y in scored}
    base_err = sum(before.values())
    paths = {ps: p for ps, p, _y in scored}

    recs = []
    chosen: dict[str, Any] = {}       # path_str -> the value actually swapped in
    for ps, path, y0 in scored:
        rec = {"path_str": ps, "y": y0, "status": None, "self": None,
               "net_delta": 0}
        found, val = select_value(samples, path, strategy)
        if not found:
            rec["status"], rec["self"] = "unavailable", "unavailable"
            recs.append(rec)
            continue

        ok0, orig_val = json_get(original, path)
        if ok0 and orig_val == val:
            # The resample said exactly what the original said. Nothing is
            # swapped and nothing can change -- but it is worth counting: this
            # is the share of the budget that buys literally nothing.
            rec["status"], rec["self"] = "identical", "unchanged"
            recs.append(rec)
            continue

        hybrid, failed = with_replacements(original, {tuple(path): val})
        if failed:
            # The path came from the original's own leaves, so this should not
            # happen; if it ever does it must not be silently counted as a
            # successful no-op.
            rec["status"], rec["self"] = "set_failed", "unavailable"
            recs.append(rec)
            continue

        try:
            after = labeler(hybrid)
        except Exception as e:
            # One unlabelable hybrid must not destroy a run that has already
            # paid for its GPU time. The field is recorded as unmeasured rather
            # than silently counted as a no-op, and the caller reports how many
            # there were -- so this shows up as lost coverage, never as a
            # flattering zero.
            rec["status"], rec["self"] = "label_failed", "unavailable"
            rec["error"] = type(e).__name__
            recs.append(rec)
            continue

        rec["status"] = "swapped"
        rec["net_delta"] = errors_over_scored(after, before) - base_err
        if ps not in after:
            rec["self"] = "lost"
        else:
            y1 = after[ps]
            rec["self"] = ("unchanged" if y1 == y0
                           else ("repaired" if y0 == 1 else "damaged"))
        chosen[ps] = val
        recs.append(rec)

    out = {"base_errors": base_err, "n_scored": len(scored), "fields": recs}

    if joint_sets:
        joint = {}
        for name, flagged_paths in joint_sets.items():
            repl = {tuple(paths[ps]): chosen[ps]
                    for ps in flagged_paths if ps in chosen}
            if not repl:
                joint[name] = {"n_swapped": 0, "errors": base_err}
                continue
            hybrid, failed = with_replacements(original, repl)
            try:
                after = labeler(hybrid)
            except Exception as e:
                # The joint pass is a CHECK on the per-field measurement, not
                # the measurement itself, so a failure here is reported and
                # skipped rather than allowed to fail the run.
                joint[name] = {"n_swapped": len(repl) - len(failed),
                               "errors": None, "error": type(e).__name__}
                continue
            joint[name] = {"n_swapped": len(repl) - len(failed),
                           "errors": errors_over_scored(after, before)}
        out["joint"] = joint
    return out


def measure_document(doc_id, domain, original, gold, schema, scored, samples,
                     strategy, joint_sets, fuzzy_threshold, number_tolerance,
                     mode_params):
    """Importable entry point for the parallel pass.

    Takes only plain data and builds the labeler on the far side, so it can be
    shipped to a joblib worker process without carrying a closure along.
    """
    labeler = make_labeler(doc_id=doc_id, domain=domain, schema=schema,
                           gold=gold, fuzzy_threshold=fuzzy_threshold,
                           number_tolerance=number_tolerance,
                           mode_params=mode_params)
    return doc_id, evaluate_document(original, scored, samples, strategy,
                                     labeler, joint_sets)

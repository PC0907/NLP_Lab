"""Tests for resume + shard selection (probe_extraction.utils.resume).

These exist because the scaled SOB extraction is long enough to hit a SLURM
time limit. If `--resume` wrongly judges a document to be complete, Stage 7
silently drops it for lack of a .rtokens.json sidecar and the corpus quietly
shrinks -- an error that would only surface as a puzzling document count much
later, after the GPU time is already spent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from probe_extraction.utils.resume import already_extracted, parse_shard


# --- shard selection -------------------------------------------------------

def test_no_shard_spec_means_the_whole_corpus():
    assert parse_shard(None) == (0, 1)
    assert parse_shard("") == (0, 1)


def test_shard_spec_is_one_based_on_the_cli_zero_based_internally():
    assert parse_shard("1/4") == (0, 4)
    assert parse_shard("4/4") == (3, 4)


@pytest.mark.parametrize("bad", ["0/4", "5/4", "4", "a/b", "-1/3", "1/0", "1/2/3"])
def test_bad_shard_specs_are_rejected(bad):
    with pytest.raises(ValueError):
        parse_shard(bad)


def test_shards_partition_the_corpus_exactly_once():
    """Concurrent GPU jobs must write disjoint files -- no document extracted
    twice, none missed."""
    n_docs, n_shards = 103, 4
    seen = []
    for shard in range(1, n_shards + 1):
        i, n = parse_shard(f"{shard}/{n_shards}")
        seen += [d for d in range(n_docs) if d % n == i]
    assert sorted(seen) == list(range(n_docs))


def test_a_single_shard_of_one_covers_everything():
    i, n = parse_shard("1/1")
    assert [d for d in range(10) if d % n == i] == list(range(10))


# --- resume ----------------------------------------------------------------

def _touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}")


def test_resume_requires_both_extraction_and_activations(tmp_path):
    ext, act = tmp_path / "extractions", tmp_path / "activations"
    ext.mkdir()
    act.mkdir()

    assert not already_extracted("d1", ext, act, False)
    _touch(ext / "d1.json")
    assert not already_extracted("d1", ext, act, False)  # activations still missing
    _touch(act / "d1.npz")
    assert already_extracted("d1", ext, act, False)


def test_resume_demands_the_rtokens_sidecar_when_capture_is_on(tmp_path):
    """The exact trap this guards: the first SOB run wrote extractions and
    activations but no per-token reasoning states. An attribution run resuming
    over those artifacts must re-extract them, not skip them."""
    ext, act = tmp_path / "extractions", tmp_path / "activations"
    _touch(ext / "d1.json")
    _touch(act / "d1.npz")

    assert already_extracted("d1", ext, act, require_rtokens=False)
    assert not already_extracted("d1", ext, act, require_rtokens=True)

    _touch(act / "d1.rtokens.json")
    assert already_extracted("d1", ext, act, require_rtokens=True)


def test_unknown_document_is_never_considered_done(tmp_path):
    ext, act = tmp_path / "extractions", tmp_path / "activations"
    ext.mkdir()
    act.mkdir()
    assert not already_extracted("never_seen", ext, act, True)
    assert not already_extracted("never_seen", ext, act, False)

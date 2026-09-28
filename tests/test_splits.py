"""Tests for the dev / held-out partition.

The held-out claim is only worth something if the partition is (a) stable as the
corpus grows, so a document cannot drift from held-out into dev between runs,
and (b) independent of dataset ordering, so it cannot inherit whatever ordering
the benchmark was assembled with. Both are tested here.
"""

from __future__ import annotations

import pytest

from probe_extraction.utils.splits import describe, partition, select


def ids(n: int, prefix: str = "sob__doc") -> list[str]:
    return [f"{prefix}{i:04d}" for i in range(n)]


# --- basic shape -----------------------------------------------------------

def test_halves_are_roughly_balanced_and_disjoint():
    """Balanced in expectation, not exactly -- assignment is by threshold on
    each document's own hash, which is what buys stability under growth."""
    p = partition(ids(1000))
    assert 450 <= len(p["dev"]) <= 550, len(p["dev"])
    assert len(p["dev"]) + len(p["heldout"]) == 1000
    assert not (p["dev"] & p["heldout"])


def test_every_document_lands_somewhere():
    everything = set(ids(377))
    p = partition(everything)
    assert p["dev"] | p["heldout"] == everything


def test_dev_fraction_is_respected():
    p = partition(ids(2000), dev_fraction=0.3)
    assert 540 <= len(p["dev"]) <= 660, len(p["dev"])   # 30% of 2000, +/- 10%
    assert len(p["dev"]) + len(p["heldout"]) == 2000


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.2, 1.5])
def test_nonsense_fractions_are_rejected(bad):
    with pytest.raises(ValueError):
        partition(ids(10), dev_fraction=bad)


def test_tiny_corpora_do_not_crash():
    """Threshold assignment cannot guarantee both halves are non-empty at tiny
    n. That is acceptable for a corpus of thousands, and describe() surfaces the
    realised sizes so an empty half is never silent."""
    for n in (1, 2, 5):
        p = partition(ids(n))
        assert len(p["dev"]) + len(p["heldout"]) == n, n


# --- the properties that make the claim defensible -------------------------

def test_partition_is_independent_of_input_order():
    forward = partition(ids(200))
    backward = partition(list(reversed(ids(200))))
    assert forward == backward


def test_partition_is_deterministic_across_calls():
    assert partition(ids(150)) == partition(ids(150))


def test_growing_the_corpus_never_moves_an_existing_document():
    """Extract 1,000 documents today and 3,000 next month: no document may
    change sides, or a held-out result is silently contaminated."""
    small = partition(ids(1000))
    large = partition(ids(3000))
    for d in ids(1000):
        in_small_dev = d in small["dev"]
        in_large_dev = d in large["dev"]
        assert in_small_dev == in_large_dev, d


def test_partition_does_not_track_document_numbering():
    """If the split followed position, dev would be dominated by low indices.
    A hash partition should scatter them."""
    p = partition(ids(1000))
    low = sum(1 for d in p["dev"] if int(d[-4:]) < 500)
    assert 200 < low < 300, low  # ~250 expected if scattered; 500 if positional


# --- select ----------------------------------------------------------------

def test_select_returns_the_matching_half():
    every = ids(400)
    p = partition(every)
    assert select(every, "dev") == p["dev"]
    assert select(every, "heldout") == p["heldout"]


def test_select_all_returns_everything():
    every = ids(40)
    assert select(every, "all") == set(every)


def test_select_rejects_an_unknown_split_name():
    with pytest.raises(ValueError):
        select(ids(10), "train")  # type: ignore[arg-type]


def test_select_ignores_duplicates_in_the_input():
    assert select(ids(50) + ids(50), "all") == set(ids(50))


def test_describe_reports_the_realised_sizes():
    d = describe(ids(100))
    assert "100 documents" in d
    assert "dev " in d and "heldout " in d
    # The realised sizes must appear, so an unbalanced or empty half is visible
    # in the log rather than silent.
    p = partition(ids(100))
    assert f"dev {len(p['dev'])}" in d
    assert f"heldout {len(p['heldout'])}" in d

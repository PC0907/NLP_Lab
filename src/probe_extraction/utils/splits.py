"""Deterministic development / held-out partition of a document set.

Why this exists. Once we start comparing feature variants -- window pooling,
multi-layer blocks, alternative trace pooling, regularisation strength -- we are
running a *search*. Choosing the winner on the same documents we report it on
inflates the result, and a reviewer is right to discount it. So the corpus is
split once:

  dev       every variant is developed and chosen here
  heldout   touched ONCE, at the end, with the method already frozen

The partition is by a hash of the document id, not by position in the dataset.
Position would correlate with however the benchmark was assembled (topic
ordering, difficulty ordering, scrape order); a hash cannot.

Each document is assigned by comparing its OWN hash against a fixed threshold,
rather than by its rank among the ids present. That distinction matters: with
rank-based cutting the midpoint moves as the corpus grows, so documents near the
boundary change sides between runs and a held-out result computed today could
quietly include documents that were dev yesterday. Thresholding depends only on
the document itself, so extracting 1,000 today and 3,000 next month leaves every
earlier assignment untouched.

The cost is that the halves are balanced only in expectation (about +/- 1% at
n = 3,000), not exactly. Stability is worth more than exact balance here, and
describe() prints the realised sizes so an unbalanced or empty half is visible.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Literal

SplitName = Literal["dev", "heldout", "all"]

# Changing this re-randomises the partition and invalidates any held-out claim
# already made. Treat it as frozen.
_SALT = "sob-reasoning-track-v1"


def _unit(doc_id: str) -> float:
    """Map a document id to a stable, uniformly distributed value in [0, 1)."""
    h = hashlib.sha256(f"{_SALT}:{doc_id}".encode()).hexdigest()
    return int(h[:16], 16) / 2 ** 64


def partition(doc_ids: Iterable[str], *, dev_fraction: float = 0.5
              ) -> dict[str, set[str]]:
    """Split document ids into {'dev': ..., 'heldout': ...}.

    Each id is assigned independently of the others, so the result is stable
    under corpus growth and independent of input order.
    """
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError(f"dev_fraction must be in (0, 1), got {dev_fraction}")
    dev, heldout = set(), set()
    for d in set(doc_ids):
        (dev if _unit(d) < dev_fraction else heldout).add(d)
    return {"dev": dev, "heldout": heldout}


def select(doc_ids: Iterable[str], split: SplitName,
           *, dev_fraction: float = 0.5) -> set[str]:
    """The subset of `doc_ids` belonging to `split`. 'all' returns everything."""
    ids = set(doc_ids)
    if split == "all":
        return ids
    if split not in ("dev", "heldout"):
        raise ValueError(f"split must be 'dev', 'heldout' or 'all', got {split!r}")
    return partition(ids, dev_fraction=dev_fraction)[split]


def describe(doc_ids: Iterable[str], *, dev_fraction: float = 0.5) -> str:
    """One-line summary for logging, so every run records which half it used."""
    p = partition(doc_ids, dev_fraction=dev_fraction)
    n = len(p["dev"]) + len(p["heldout"])
    return (f"{n} documents -> dev {len(p['dev'])} / heldout {len(p['heldout'])} "
            f"(salt {_SALT!r}, hash-partitioned)")

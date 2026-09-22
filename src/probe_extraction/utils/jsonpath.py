"""Read and write a single leaf of a parsed JSON document by its path.

Stage 9 builds a "hybrid" record: the original extraction everywhere, with the
flagged fields -- and only those -- replaced by the regenerated values. That
needs exact addressing of one leaf at a time, using the same `path` lists the
extraction metadata already stores (e.g. ["actress", "name"] or
["items", 0, "name"]).

Kept separate and dependency-free so the addressing can be unit tested on its
own. A silent mis-set here would corrupt the measured repair rate without
raising anything, so every operation reports success explicitly rather than
guessing.
"""

from __future__ import annotations

import copy
from typing import Any, Sequence

PathElem = str | int
Path = Sequence[PathElem]

_MISSING = object()


def _step(container: Any, key: PathElem) -> Any:
    """One level down, or _MISSING if the step does not apply."""
    if isinstance(key, bool):
        # bool is a subclass of int; treating True as index 1 would be a bug.
        return _MISSING
    if isinstance(container, dict):
        # A JSON object key is always a string; tolerate an int path element
        # by trying its string form too.
        if key in container:
            return container[key]
        if isinstance(key, int) and str(key) in container:
            return container[str(key)]
        return _MISSING
    if isinstance(container, list):
        if isinstance(key, int) and -len(container) <= key < len(container):
            return container[key]
        return _MISSING
    return _MISSING


def json_get(obj: Any, path: Path) -> tuple[bool, Any]:
    """Return (found, value) for `path` in `obj`.

    `found` is False when any step of the path does not exist, so a legitimately
    stored None is distinguishable from an absent path -- the difference matters:
    a regenerated null is a real answer, a missing path means the regeneration
    had nothing to offer for that field.
    """
    cur = obj
    for key in path:
        cur = _step(cur, key)
        if cur is _MISSING:
            return False, None
    return True, cur


def json_set(obj: Any, path: Path, value: Any) -> bool:
    """Set `path` to `value` in `obj`, in place. Returns False and changes
    nothing if the path does not already exist.

    Refusing to create missing paths is deliberate. Stage 9 replaces fields the
    original extraction actually produced; inventing a new key would fabricate a
    field the model never emitted, which is exactly what this analysis must not
    do.
    """
    if not path:
        return False
    parent = obj
    for key in path[:-1]:
        parent = _step(parent, key)
        if parent is _MISSING:
            return False
    last = path[-1]
    if isinstance(last, bool):
        return False
    if isinstance(parent, dict):
        if last in parent:
            parent[last] = value
            return True
        if isinstance(last, int) and str(last) in parent:
            parent[str(last)] = value
            return True
        return False
    if isinstance(parent, list):
        if isinstance(last, int) and -len(parent) <= last < len(parent):
            parent[last] = value
            return True
        return False
    return False


def with_replacements(original: Any, replacements: dict[tuple, Any]) -> tuple[Any, list[tuple]]:
    """Deep-copy `original` and apply `replacements` ({path tuple: value}).

    Returns (new_object, failed_paths). The original is never mutated, so a
    caller can build one hybrid per budget from the same source without the
    budgets contaminating each other.
    """
    out = copy.deepcopy(original)
    failed = [p for p, v in replacements.items() if not json_set(out, p, v)]
    return out, failed

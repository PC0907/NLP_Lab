"""Resume + shard selection for long extraction runs.

Scaling the SOB extraction past a few hundred documents makes a single Stage-1
run longer than a SLURM partition's time limit, so it needs to survive being
killed and restarted, and to be splittable across concurrent GPU jobs. Both are
just document-selection questions, kept here (torch-free) so they can be unit
tested without the model stack.
"""

from __future__ import annotations

from pathlib import Path


def parse_shard(spec: str | None) -> tuple[int, int]:
    """Parse a 1-based "I/N" shard spec into a 0-based (index, count).

    None or "" means "the whole corpus" -> (0, 1).
    """
    if not spec:
        return (0, 1)
    parts = str(spec).split("/")
    if len(parts) != 2:
        raise ValueError(f"--shard must look like I/N, got {spec!r}")
    try:
        i, n = int(parts[0]), int(parts[1])
    except ValueError as e:
        raise ValueError(f"--shard must look like I/N, got {spec!r}") from e
    if n < 1 or not (1 <= i <= n):
        raise ValueError(f"--shard I must be in 1..N, got {spec!r}")
    return (i - 1, n)


def already_extracted(
    doc_id: str,
    extractions_dir: Path,
    activations_dir: Path,
    require_rtokens: bool,
) -> bool:
    """True when this document's artifacts are complete FOR THE CURRENT capture
    settings.

    The `require_rtokens` arm matters: the first SOB run wrote extractions and
    activations but no per-token reasoning states. Resuming an attribution run
    over those artifacts must re-extract them rather than skip them, or Stage 7
    would silently drop the documents for want of a sidecar and the corpus would
    quietly shrink.
    """
    if not (extractions_dir / f"{doc_id}.json").exists():
        return False
    if not (activations_dir / f"{doc_id}.npz").exists():
        return False
    if require_rtokens and not (activations_dir / f"{doc_id}.rtokens.json").exists():
        return False
    return True

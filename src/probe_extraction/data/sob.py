"""SOB (Structured Output Benchmark) loader — text / multi-hop subset.

SOB (interfaze-ai/sob, arXiv:2604.25359) pairs a multi-hop question + source
context with a per-record JSON Schema and a human-verified ground-truth JSON
answer. The text subset is derived from HotpotQA, so extraction genuinely
requires multi-hop REASONING — the property that makes it a good fit for probing
a reasoning model's (DeepSeek-R1) reasoning trace.

Key differences from ExtractBench (handled here + a one-line Stage-2 change):
  - Schema is PER RECORD (not one schema per domain). Each Document carries its
    own schema; get_schema(domain) is therefore not meaningful and returns {}.
  - Input is already text (Wikipedia passages) — no PDF extraction. The question
    is folded into Document.text so the existing extraction prompt elicits the
    answer without any prompt change.

Offline use on the cluster: run scripts/00_download_sob.py on the LOGIN node to
cache the dataset to disk (compute nodes have no internet). Point
data.benchmark_path at that directory; this loader uses datasets.load_from_disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

from probe_extraction.data.base import Benchmark, Document

logger = logging.getLogger(__name__)


# ============================================================================
# Pure record -> Document mapping (no datasets dependency, unit-testable)
# ============================================================================

def record_to_document(rec: dict[str, Any]) -> Document:
    """Map one SOB record (a dict) to a pipeline Document.

    The question is prepended to the context so the standard extraction prompt
    ("extract per the schema from the document below") sees what to answer.
    """
    question = (rec.get("question") or "").strip()
    context = (rec.get("context") or "").strip()
    text = f"Question: {question}\n\nContext:\n{context}" if question else context

    source_dataset = rec.get("source_dataset") or "text"
    domain = f"sob/{source_dataset}"

    rec_id = str(rec.get("record_id") or "").strip()
    doc_id = _safe_doc_id(rec_id) if rec_id else _safe_doc_id(str(abs(hash(text))))

    return Document(
        doc_id=doc_id,
        domain=domain,
        text=text,
        schema=rec.get("json_schema") or {},
        gold=rec.get("ground_truth") or {},
        source_path=Path(rec_id or doc_id),
        metadata={
            "schema_complexity": rec.get("schema_complexity"),
            "question_type": rec.get("question_type"),
            "question_difficulty": rec.get("question_difficulty"),
            "source_dataset": source_dataset,
        },
        extraction_error=None,  # text is already available; nothing can fail here
    )


def _safe_doc_id(rec_id: str) -> str:
    """SOB record_ids are sha256 hex (filesystem-safe). Guard anyway."""
    stem = rec_id
    for bad in [" ", "/", "\\", ":", "*", "?", '"', "<", ">", "|"]:
        stem = stem.replace(bad, "_")
    return f"sob__{stem}"


# ============================================================================
# Benchmark loader
# ============================================================================

class SOB(Benchmark):
    """Loader for the SOB text subset via a locally-cached HF dataset.

    Args:
        benchmark_path: directory produced by `datasets.save_to_disk` (see
            scripts/00_download_sob.py). May hold a DatasetDict (train/
            validation/test) or a single split.
        split: which split to use (default "test", 5,000 records).
        domains: optional list of "sob/<source>" domains to keep.
        max_documents: cap on records (useful for a small first run).
    """

    def __init__(
        self,
        benchmark_path: str | Path,
        *,
        split: str = "test",
        domains: list[str] | None = None,
        max_documents: int | None = None,
    ) -> None:
        try:
            from datasets import load_from_disk  # lazy: only needed for SOB
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The `datasets` package is required for the SOB benchmark. "
                "Add `datasets` to requirements.txt and pip install it."
            ) from e

        self.benchmark_path = Path(benchmark_path)
        if not self.benchmark_path.exists():
            raise FileNotFoundError(
                f"SOB dataset dir not found: {self.benchmark_path}. "
                f"Run scripts/00_download_sob.py on the login node first."
            )

        ds = load_from_disk(str(self.benchmark_path))
        # DatasetDict vs single Dataset.
        if hasattr(ds, "keys") and not hasattr(ds, "features"):
            if split not in ds:
                raise KeyError(f"Split {split!r} not in dataset; have {list(ds.keys())}.")
            ds = ds[split]
        self._split = split

        self._domain_filter = set(domains) if domains else None
        if max_documents is not None:
            ds = ds.select(range(min(max_documents, len(ds))))
        self._ds = ds

        # Precompute the domain set (cheap column scan).
        src = self._ds["source_dataset"] if "source_dataset" in self._ds.column_names else None
        if src is not None:
            all_domains = sorted({f"sob/{s or 'text'}" for s in src})
        else:
            all_domains = ["sob/text"]
        self._domains = ([d for d in all_domains if d in self._domain_filter]
                         if self._domain_filter else all_domains)

        logger.info("SOB initialized: split=%s, %d records, domains=%s",
                    split, len(self._ds), self._domains)

    # ---- Benchmark interface ------------------------------------------------

    @property
    def name(self) -> str:
        return "sob"

    @property
    def domains(self) -> list[str]:
        return list(self._domains)

    def __len__(self) -> int:
        return len(self._ds)

    def __iter__(self) -> Iterator[Document]:
        for rec in self._ds:
            doc = record_to_document(rec)
            if self._domain_filter and doc.domain not in self._domain_filter:
                continue
            yield doc

    def get_schema(self, domain: str) -> dict[str, Any]:
        # SOB schemas are per-record; there is no single domain-level schema.
        # Stage 2 uses each Document's own schema, so this is intentionally {}.
        return {}

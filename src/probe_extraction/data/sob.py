"""SOB (Structured Output Benchmark) loader — text / multi-hop subset.

SOB (interfaze-ai/sob, arXiv:2604.25359) pairs a multi-hop question + source
context with a per-record JSON Schema and a human-verified ground-truth JSON
answer. The text subset is derived from HotpotQA, so extraction genuinely
requires multi-hop REASONING.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

from probe_extraction.data.base import Benchmark, Document

logger = logging.getLogger(__name__)


def _as_dict(v: Any) -> dict[str, Any]:
    """SOB stores json_schema / ground_truth as JSON *strings* in the parquet.
    Parse to a dict; tolerate an already-parsed dict or an unparseable value."""
    if isinstance(v, dict):
        return v
    if isinstance(v, str) and v.strip():
        try:
            parsed = json.loads(v)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, TypeError):
            return {}
    return {}


def record_to_document(rec: dict[str, Any]) -> Document:
    """Map one SOB record (a dict) to a pipeline Document."""
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
        schema=_as_dict(rec.get("json_schema")),
        gold=_as_dict(rec.get("ground_truth")),
        source_path=Path(rec_id or doc_id),
        metadata={
            "schema_complexity": rec.get("schema_complexity"),
            "question_type": rec.get("question_type"),
            "question_difficulty": rec.get("question_difficulty"),
            "source_dataset": source_dataset,
        },
        extraction_error=None,
    )


def _safe_doc_id(rec_id: str) -> str:
    stem = rec_id
    for bad in [" ", "/", "\\", ":", "*", "?", '"', "<", ">", "|"]:
        stem = stem.replace(bad, "_")
    return f"sob__{stem}"


class SOB(Benchmark):
    """Loader for the SOB text subset via a locally-cached HF dataset."""

    def __init__(
        self,
        benchmark_path: str | Path,
        *,
        split: str = "test",
        domains: list[str] | None = None,
        max_documents: int | None = None,
    ) -> None:
        try:
            from datasets import load_from_disk
        except ImportError as e:  # pragma: no cover
            raise ImportError("The `datasets` package is required for SOB.") from e

        self.benchmark_path = Path(benchmark_path)
        if not self.benchmark_path.exists():
            raise FileNotFoundError(
                f"SOB dataset dir not found: {self.benchmark_path}. "
                f"Run scripts/00_download_sob.py on the login node first."
            )

        ds = load_from_disk(str(self.benchmark_path))
        if hasattr(ds, "keys") and not hasattr(ds, "features"):
            if split not in ds:
                raise KeyError(f"Split {split!r} not in dataset; have {list(ds.keys())}.")
            ds = ds[split]
        self._split = split

        self._domain_filter = set(domains) if domains else None
        if max_documents is not None:
            ds = ds.select(range(min(max_documents, len(ds))))
        self._ds = ds

        src = self._ds["source_dataset"] if "source_dataset" in self._ds.column_names else None
        if src is not None:
            all_domains = sorted({f"sob/{s or 'text'}" for s in src})
        else:
            all_domains = ["sob/text"]
        self._domains = ([d for d in all_domains if d in self._domain_filter]
                         if self._domain_filter else all_domains)

        logger.info("SOB initialized: split=%s, %d records, domains=%s",
                    split, len(self._ds), self._domains)

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
        return {}

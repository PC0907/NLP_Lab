"""SOB (Structured Output Benchmark) loader -- text modality.

JigsawStack's SOB (https://huggingface.co/datasets/interfaze-ai/sob) measures
VALUE-correctness of structured JSON output. Each record is a QA-style task:
    context      : source text (HotpotQA Wikipedia paragraphs, for text split)
    question     : the question to answer in structured form
    json_schema  : per-record JSON Schema for the answer
    ground_truth : raw gold answer
    validated_output : schema-aligned gold (what we score against)

This is a DIFFERENT task shape from ExtractBench: context + question -> answer,
vs document -> extract-all-fields. We use it as a CROSS-TASK-TYPE generalization
test -- does the probe's trust signal hold on structured QA, not just document
extraction? (SOB errors are partly multi-hop *reasoning* errors, so this shades
toward hallucination detection; note that when reporting.)

DESIGN: the per-record question is folded into `doc.text` with a clear template,
so the existing `text + schema -> extract` path works unchanged (no Extractor or
prompt-builder edits). The model processes question+context+schema in-context
either way; the probe reads the same hidden states. If SOB proves valuable, a
proper `question` prompt field is the clean follow-up.

Per-record schemas: unlike a fixed-schema benchmark, each row carries its own
json_schema; the loader reads it per document into doc.schema.

Usage: set `benchmark: sob` in the config.
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Any, Iterator

from probe_extraction.data.base import Benchmark, Document

logger = logging.getLogger(__name__)

HF_DATASET = "interfaze-ai/sob"
DOMAIN = "sob/text"

# Column-name candidates (be tolerant -- HF datasets sometimes rename).
_CONTEXT_KEYS = ("context", "input", "document", "text", "passage")
_QUESTION_KEYS = ("question", "query", "prompt")
_SCHEMA_KEYS = ("json_schema", "schema", "output_schema")
_GOLD_KEYS = ("validated_output", "ground_truth", "answer", "output")


def _first_present(row: dict, keys) -> Any:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def _as_obj(raw: Any) -> Any:
    """Parse a field that may be a dict already, a JSON string, or a py-repr."""
    if raw is None or isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(s)
            except (ValueError, SyntaxError):
                return raw  # leave as string if unparseable
    return raw


class SOBench(Benchmark):
    """Loader for JigsawStack SOB, text modality."""

    def __init__(
        self,
        benchmark_path: str | Path | None = None,
        *,
        domains: list[str] | None = None,
        max_documents: int | None = None,
        hf_config: str = "default",    # text / image / audio -- we build for text
        hf_split: str = "train",
        **kwargs: Any,
    ) -> None:
        self.max_documents = max_documents
        self.hf_config = hf_config
        self.hf_split = hf_split

        if domains:
            unknown = set(domains) - {DOMAIN}
            if unknown:
                raise ValueError(
                    f"Unknown domains {sorted(unknown)}; this loader provides {DOMAIN!r}."
                )

        from datasets import load_dataset
        # SOB is multi-config (text/image/audio); load the text config.
        try:
            ds = load_dataset(HF_DATASET, hf_config)[hf_split]
        except (ValueError, KeyError):
            # fall back: some datasets expose splits without named configs
            ds = load_dataset(HF_DATASET)[hf_split]

        self._rows = []
        skipped = 0
        for i, row in enumerate(ds):
            context = _first_present(row, _CONTEXT_KEYS)
            question = _first_present(row, _QUESTION_KEYS)
            schema = _as_obj(_first_present(row, _SCHEMA_KEYS))
            gold = _as_obj(_first_present(row, _GOLD_KEYS))
            # need context, schema, and gold to be usable
            if context is None or schema is None or gold is None:
                skipped += 1
                continue
            self._rows.append({
                "index": i,
                "context": str(context),
                "question": "" if question is None else str(question),
                "schema": schema,
                "gold": gold,
            })
        if max_documents is not None:
            self._rows = self._rows[:max_documents]

        logger.info("SOBench initialized: %d usable rows (skipped %d) from %s[%s/%s]",
                    len(self._rows), skipped, HF_DATASET, hf_config, hf_split)
        if self._rows:
            ex = self._rows[0]
            logger.info("  example: context=%d chars, question=%d chars, schema_keys=%s",
                        len(ex["context"]), len(ex["question"]),
                        list(ex["schema"].get("properties", {}).keys())[:6]
                        if isinstance(ex["schema"], dict) else "n/a")

    @property
    def name(self) -> str:
        return "sob"

    @property
    def domains(self) -> list[str]:
        return [DOMAIN]

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[Document]:
        for row in self._rows:
            yield self._build(row)

    def get_schema(self, domain: str) -> dict[str, Any]:
        # SOB schemas are per-record; no single fixed schema. Return the first
        # row's schema as a representative (callers should prefer doc.schema).
        if domain != DOMAIN:
            raise KeyError(f"Schema for domain {domain!r} not available.")
        return self._rows[0]["schema"] if self._rows else {}

    def _build(self, row: dict) -> Document:
        idx = row["index"]
        q = row["question"].strip()
        ctx = row["context"]
        # Fold question into text with a clear template so the model (and a human
        # reading a dumped doc) can tell them apart.
        if q:
            text = f"Question: {q}\n\nContext:\n{ctx}"
        else:
            text = ctx

        doc_id = f"sob__text__{idx:05d}"
        gold = row["gold"] if isinstance(row["gold"], dict) else {"answer": row["gold"]}

        return Document(
            doc_id=doc_id,
            domain=DOMAIN,
            text=text,
            schema=row["schema"],
            gold=gold,
            source_path=None,
            metadata={"hf_index": idx, "has_question": bool(q)},
            extraction_error=None,
        )
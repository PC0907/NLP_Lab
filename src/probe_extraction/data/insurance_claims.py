"""Insurance Claims benchmark loader (CONSTRUCT / Cleanlab).

Loads the Cleanlab insurance-claims-extraction dataset
(https://huggingface.co/datasets/Cleanlab/insurance-claims-extraction),
one of four clean-gold structured-output benchmarks introduced by
CONSTRUCT (Goh & Mueller, arXiv:2603.18014). Used as an INDEPENDENT
generalization benchmark: the probe was developed on ExtractBench; testing it
on separately-curated gold checks the trust signal is not an ExtractBench
artifact.

Why a dedicated loader (not the ExtractBench loader): insurance claims are
TEXT-native (email-style claim descriptions), not PDFs. The ExtractBench loader
hard-requires a `.pdf` per document and runs PyMuPDF on it; feeding it fake
PDFs would round-trip text -> PDF -> text and reintroduce normalization
artifacts. This loader mirrors the RealKIE pattern: read text inline, yield
ordinary Documents, no pdf_utils.

Data (HF `train` split, n=30):
    claim_text     : str  -- the claim document (~2.5k chars)
    ground_truth   : nested JSON (str or dict): header, policy_details,
                     insured_objects[], incident_description

A smoke test confirmed Qwen3.5-4B reproduces this schema's nested keys 100%
literally, so the path-based matcher aligns (no Llama-style key divergence).

Usage: set `benchmark: insurance_claims` in the config.
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Any, Iterator

from probe_extraction.data.base import Benchmark, Document

logger = logging.getLogger(__name__)


HF_DATASET = "Cleanlab/insurance-claims-extraction"
DOMAIN = "insurance/claims"

# Insurance schema (plain JSON Schema form). Derived from CONSTRUCT's gold
# structure; confirmed by the key smoke test.
INSURANCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "header": {
            "type": "object",
            "properties": {
                "claim_id": {"type": "string"},
                "report_date": {"type": "string", "description": "YYYY-MM-DD"},
                "incident_date": {"type": "string", "description": "YYYY-MM-DD"},
                "reported_by": {"type": "string"},
                "channel": {"type": "string"},
            },
        },
        "policy_details": {
            "type": "object",
            "properties": {
                "policy_number": {"type": "string"},
                "policyholder_name": {"type": "string"},
                "coverage_type": {"type": "string"},
                "effective_date": {"type": "string"},
                "expiration_date": {"type": "string"},
            },
        },
        "insured_objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "object_id": {"type": "string"},
                    "object_type": {"type": "string"},
                    "make_model": {"type": "string"},
                    "year": {"type": "integer"},
                    "location_address": {"type": "string"},
                    "estimated_value": {"type": "number"},
                },
            },
        },
        "incident_description": {
            "type": "object",
            "properties": {
                "incident_type": {"type": "string"},
                "location_type": {"type": "string"},
                "estimated_damage_amount": {"type": "number"},
                "police_report_number": {"type": "string"},
            },
        },
    },
}


def _parse_gold(raw: Any) -> dict[str, Any]:
    """Gold may be a dict, a JSON string, or a python-repr string."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return ast.literal_eval(raw)
    raise ValueError(f"Unexpected gold type: {type(raw)}")


class InsuranceClaims(Benchmark):
    """Loader for the Cleanlab insurance-claims-extraction dataset."""

    def __init__(
        self,
        benchmark_path: str | Path | None = None,
        *,
        domains: list[str] | None = None,
        max_documents: int | None = None,
        hf_split: str = "train",
        **kwargs: Any,  # tolerate extra dispatcher kwargs
    ) -> None:
        self.benchmark_path = Path(benchmark_path) if benchmark_path else None
        self.max_documents = max_documents
        self.hf_split = hf_split

        if domains:
            unknown = set(domains) - {DOMAIN}
            if unknown:
                raise ValueError(
                    f"Unknown domains requested: {sorted(unknown)}. "
                    f"This loader only provides {DOMAIN!r}."
                )

        from datasets import load_dataset  # local import: heavy dependency
        ds = load_dataset(HF_DATASET)[hf_split]

        self._rows: list[dict[str, Any]] = []
        for i, row in enumerate(ds):
            self._rows.append({"index": i, **row})
        if self.max_documents is not None:
            self._rows = self._rows[: self.max_documents]

        logger.info(
            "InsuranceClaims initialized: %d document(s) from %s[%s]",
            len(self._rows), HF_DATASET, hf_split,
        )

    @property
    def name(self) -> str:
        return "insurance_claims"

    @property
    def domains(self) -> list[str]:
        return [DOMAIN]

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[Document]:
        for row in self._rows:
            yield self._build_document(row)

    def get_schema(self, domain: str) -> dict[str, Any]:
        if domain != DOMAIN:
            raise KeyError(f"Schema for domain {domain!r} not available.")
        return INSURANCE_SCHEMA

    def _build_document(self, row: dict[str, Any]) -> Document:
        idx = row["index"]
        text = row["claim_text"]
        gold = _parse_gold(row["ground_truth"])

        claim_id = ""
        try:
            claim_id = str(gold.get("header", {}).get("claim_id", "") or "")
        except Exception:
            claim_id = ""
        stem = claim_id if claim_id else f"claim{idx:03d}"
        doc_id = self._make_doc_id(DOMAIN, stem)

        return Document(
            doc_id=doc_id,
            domain=DOMAIN,
            text=text,
            schema=INSURANCE_SCHEMA,
            gold=gold,
            source_path=None,
            metadata={"hf_index": idx},
            extraction_error=None,
        )

    @staticmethod
    def _make_doc_id(domain: str, stem: str) -> str:
        domain_part = domain.replace("/", "__")
        for bad in [" ", "/", "\\", ":", "*", "?", '"', "<", ">", "|"]:
            stem = stem.replace(bad, "_")
        return f"{domain_part}__{stem}"
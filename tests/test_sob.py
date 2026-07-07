"""Tests for the SOB loader mapping and SOB-style labeling.

record_to_document is pure (no `datasets` dependency), so these run on CPU.
Uses the actual record shape from interfaze-ai/sob (record_id, context,
question, json_schema, ground_truth, source_dataset, ...).
"""
from __future__ import annotations

from probe_extraction.data.sob import record_to_document
from probe_extraction.labeling.matcher import label_extraction
from probe_extraction.labeling.value_compare import ComparisonStrategy


_REC = {
    "record_id": "c3c68e6bb53539968ebb71cb42f022fe",
    "question": "what percussion instrument called zils was used?",
    "context": "Xylophone: a wooden instrument... Tambourine: a frame drum with zils...",
    "json_schema": {
        "type": "object",
        "properties": {
            "instrument_name": {"type": "string"},
            "instrument_family": {"type": "string"},
            "main_components": {"type": "array", "items": {"type": "string"}},
        },
    },
    "ground_truth": {
        "instrument_name": "Tambourine",
        "instrument_family": "percussion",
        "main_components": ["frame", "zils"],
    },
    "source_dataset": "hotpotqa",
    "schema_complexity": "hard",
}


def test_record_to_document_mapping():
    doc = record_to_document(_REC)
    assert doc.doc_id == "sob__c3c68e6bb53539968ebb71cb42f022fe"
    assert doc.domain == "sob/hotpotqa"
    assert doc.text.startswith("Question: what percussion")
    assert "Context:" in doc.text
    assert doc.schema == _REC["json_schema"]
    assert doc.gold == _REC["ground_truth"]
    assert doc.extraction_error is None
    assert doc.metadata["schema_complexity"] == "hard"


def test_record_to_document_id_is_filesystem_safe():
    doc = record_to_document({"record_id": "a/b:c", "context": "x", "question": "q"})
    for bad in '/\\:*?"<>|':
        assert bad not in doc.doc_id


def test_sob_labeling_correct_answer_structure_aware():
    doc = record_to_document(_REC)
    extracted = {
        "instrument_name": "tambourine",           # case differs -> AUTO matches
        "instrument_family": "Percussion",
        "main_components": ["zils", "frame"],       # set-equal
    }
    res = label_extraction(
        doc_id=doc.doc_id, domain=doc.domain, schema=doc.schema, gold=doc.gold,
        extracted=extracted,
        leaf_default=ComparisonStrategy.AUTO, structure_aware=True,
    )
    assert res.n_errors == 0


def test_sob_labeling_flags_wrong_value():
    doc = record_to_document(_REC)
    extracted = {
        "instrument_name": "Xylophone",            # wrong
        "instrument_family": "percussion",
        "main_components": ["frame", "zils"],
    }
    res = label_extraction(
        doc_id=doc.doc_id, domain=doc.domain, schema=doc.schema, gold=doc.gold,
        extracted=extracted,
        leaf_default=ComparisonStrategy.AUTO, structure_aware=True,
    )
    assert res.n_errors == 1

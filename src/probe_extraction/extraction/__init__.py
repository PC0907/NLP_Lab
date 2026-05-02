"""Extraction stage: prompts, parsing, and orchestration.

The extraction stage runs the LLM on each document, captures the JSON output
plus per-field hidden-state activations, and saves both to disk for
downstream stages (labeling, probe training).
"""

from probe_extraction.extraction.extractor import Extractor, ExtractionResult
from probe_extraction.extraction.parser import (
    FieldLocation,
    locate_fields,
    parse_json_output,
)
from probe_extraction.extraction.prompts import build_extraction_prompt

__all__ = [
    "Extractor",
    "ExtractionResult",
    "FieldLocation",
    "build_extraction_prompt",
    "locate_fields",
    "parse_json_output",
]
"""Labeling: produce per-field correctness labels by comparing extracted
values against gold annotations.

A Labeler walks the gold JSON and the extracted JSON in parallel and
produces a binary label per leaf field: 0 = correct, 1 = error. These
labels are the targets the probe is trained on.
"""

from probe_extraction.labeling.matcher import (
    FieldLabel,
    LabelingResult,
    Matcher,
    label_extraction,
)
from probe_extraction.labeling.value_compare import (
    ComparisonStrategy,
    compare_values,
    normalize_email,
    normalize_url,
)

__all__ = [
    "ComparisonStrategy",
    "FieldLabel",
    "LabelingResult",
    "Matcher",
    "compare_values",
    "label_extraction",
    "normalize_email",
    "normalize_url",
]
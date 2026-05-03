"""Gold-vs-extracted matcher.

Walks two parallel JSON trees (gold annotation and model extraction),
deciding for each leaf field whether the model got it right or wrong.
Produces a list of FieldLabel records keyed by path_str — the same
identifier the extractor used when saving activations, so labels and
activations can be joined downstream.

Key design decisions (discussed and confirmed before implementation):
  - Arrays of objects: positional alignment (model[i] ↔ gold[i])
  - Arrays of primitives: compared as sets, treated as a single field
  - Hallucinations (model has value, gold empty) and omissions (model
    empty, gold has value) both count as errors
  - Comparison strategy per field comes from the schema's
    `evaluation_config`, with reasonable fallbacks

What this module does NOT do:
  - I/O (reading extractions or writing labels — that's the script's job)
  - Field localization (already done by the extractor)
  - Probe training (Stage 3)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from probe_extraction.labeling.value_compare import (
    ComparisonStrategy,
    compare_values,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Public types
# ============================================================================

@dataclass
class FieldLabel:
    """Label for a single extracted field.

    Attributes:
        path: JSON path as a list of keys/indices, e.g.
            ["authors", 0, "name"]. Same path the extractor used.
        path_str: Dotted-string form, used as the lookup key when joining
            with activations. E.g. "authors.0.name".
        gold_value: The reference value from the gold annotation. May be
            None if gold has no value at this path (hallucination case).
        extracted_value: The model's value at this path. May be None if
            the model omitted this field (omission case).
        is_error: 1 if the field is wrong, 0 if correct.
        error_type: Categorization of the error, useful for analysis:
            - "match" (no error)
            - "value_mismatch" (both present, different values)
            - "hallucination" (model has value, gold empty/missing)
            - "omission" (gold has value, model empty/missing)
            - "type_mismatch" (e.g., gold list, model string)
        comparison_strategy: Which strategy was used to compare. String form
            of ComparisonStrategy.
        gold_present: Whether gold has a non-empty value at this path.
        extracted_present: Whether extracted has a non-empty value at this
            path.
    """

    path: list[str | int]
    path_str: str
    gold_value: Any
    extracted_value: Any
    is_error: int
    error_type: str
    comparison_strategy: str
    gold_present: bool
    extracted_present: bool


@dataclass
class LabelingResult:
    """All field labels for a single document, plus aggregate stats.

    Attributes:
        doc_id: Identifier of the source document.
        domain: Domain identifier (e.g., "academic/research").
        labels: One FieldLabel per labeled field.
        n_total: Number of fields labeled.
        n_errors: Number of fields with is_error=1.
        n_hallucinations: Subset of errors that are hallucinations.
        n_omissions: Subset of errors that are omissions.
        n_value_mismatches: Subset that are genuine value disagreements.
        n_type_mismatches: Subset that are type-level disagreements.
        unmatched_gold_paths: Gold leaves with no corresponding extracted
            value (these still appear as omission errors in `labels`,
            but are surfaced separately for analysis).
        unmatched_extracted_paths: Extracted leaves with no corresponding
            gold value (hallucinations; same — surfaced separately).
    """

    doc_id: str
    domain: str
    labels: list[FieldLabel]
    n_total: int
    n_errors: int
    n_hallucinations: int
    n_omissions: int
    n_value_mismatches: int
    n_type_mismatches: int
    unmatched_gold_paths: list[list[str | int]] = field(default_factory=list)
    unmatched_extracted_paths: list[list[str | int]] = field(default_factory=list)

    @property
    def error_rate(self) -> float:
        return self.n_errors / self.n_total if self.n_total else 0.0


# ============================================================================
# Public entry point
# ============================================================================

def label_extraction(
    *,
    doc_id: str,
    domain: str,
    schema: dict[str, Any],
    gold: dict[str, Any],
    extracted: dict[str, Any],
    fuzzy_threshold: float = 0.85,
    number_tolerance: float = 0.01,
) -> LabelingResult:
    """Convenience function: build a Matcher and label one extraction.

    Most callers should use this rather than the Matcher class directly.
    """
    matcher = Matcher(
        schema=schema,
        fuzzy_threshold=fuzzy_threshold,
        number_tolerance=number_tolerance,
    )
    return matcher.label(doc_id=doc_id, domain=domain, gold=gold, extracted=extracted)


# ============================================================================
# Matcher
# ============================================================================

class Matcher:
    """Walks gold + extracted in parallel, producing FieldLabels.

    The matcher is constructed once per schema (cheap), then used to label
    one or more documents that share that schema. The schema is used to
    look up per-field `evaluation_config` values.
    """

    # Map ExtractBench's evaluation_config strings to our strategies.
    # Schemas in ExtractBench can declare any of:
    #   string_exact, string_case_insensitive, string_fuzzy, string_semantic,
    #   integer_exact, number_tolerance, boolean_exact, array_llm.
    # We translate string_semantic and array_llm to FUZZY for now — both
    # ideally use an LLM judge; we approximate.
    _STRATEGY_MAP: dict[str, ComparisonStrategy] = {
        "string_exact": ComparisonStrategy.EXACT,
        "string_case_insensitive": ComparisonStrategy.CASE_INSENSITIVE,
        "string_fuzzy": ComparisonStrategy.FUZZY,
        "string_semantic": ComparisonStrategy.FUZZY,  # LLM judge approx
        "integer_exact": ComparisonStrategy.EXACT,
        "number_tolerance": ComparisonStrategy.NUMBER,
        "boolean_exact": ComparisonStrategy.EXACT,
        "array_llm": ComparisonStrategy.FUZZY,  # used for primitive arrays
    }

    def __init__(
        self,
        *,
        schema: dict[str, Any],
        fuzzy_threshold: float = 0.85,
        number_tolerance: float = 0.01,
    ) -> None:
        self.schema = schema
        self.fuzzy_threshold = fuzzy_threshold
        self.number_tolerance = number_tolerance

    # ------------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------------

    def label(
        self,
        *,
        doc_id: str,
        domain: str,
        gold: dict[str, Any],
        extracted: dict[str, Any],
    ) -> LabelingResult:
        """Produce labels for a single (gold, extracted) pair."""
        labels: list[FieldLabel] = []
        unmatched_gold: list[list[str | int]] = []
        unmatched_extracted: list[list[str | int]] = []

        self._walk(
            schema=self.schema,
            gold=gold,
            extracted=extracted,
            path=[],
            labels=labels,
            unmatched_gold=unmatched_gold,
            unmatched_extracted=unmatched_extracted,
        )

        # Aggregates
        n_total = len(labels)
        n_errors = sum(lab.is_error for lab in labels)
        n_hallucinations = sum(1 for lab in labels if lab.error_type == "hallucination")
        n_omissions = sum(1 for lab in labels if lab.error_type == "omission")
        n_value_mismatches = sum(1 for lab in labels if lab.error_type == "value_mismatch")
        n_type_mismatches = sum(1 for lab in labels if lab.error_type == "type_mismatch")

        return LabelingResult(
            doc_id=doc_id,
            domain=domain,
            labels=labels,
            n_total=n_total,
            n_errors=n_errors,
            n_hallucinations=n_hallucinations,
            n_omissions=n_omissions,
            n_value_mismatches=n_value_mismatches,
            n_type_mismatches=n_type_mismatches,
            unmatched_gold_paths=unmatched_gold,
            unmatched_extracted_paths=unmatched_extracted,
        )

    # ------------------------------------------------------------------------
    # Recursive walker
    # ------------------------------------------------------------------------

    def _walk(
        self,
        *,
        schema: dict[str, Any] | None,
        gold: Any,
        extracted: Any,
        path: list[str | int],
        labels: list[FieldLabel],
        unmatched_gold: list[list[str | int]],
        unmatched_extracted: list[list[str | int]],
    ) -> None:
        """Recursively walk gold + extracted in parallel.

        At each node:
          - If both are dicts: recurse into matching keys, union of all keys.
          - If both are lists of objects: pair positionally, recurse.
          - If both are lists of primitives: emit a single field label
            (set comparison).
          - If types disagree at a node: emit a type_mismatch label.
          - If we hit a leaf: compare values, emit a label.
        """
        # Resolve any anyOf in the schema to the most specific applicable
        # variant given the actual gold value (best effort).
        sub_schema = _resolve_schema_for_value(schema, gold)

        # ------ Both dicts ------
        if isinstance(gold, dict) and isinstance(extracted, dict):
            properties = (sub_schema or {}).get("properties", {})
            keys = set(gold.keys()) | set(extracted.keys())
            for key in sorted(keys):  # deterministic order
                self._walk(
                    schema=properties.get(key),
                    gold=gold.get(key),
                    extracted=extracted.get(key),
                    path=path + [key],
                    labels=labels,
                    unmatched_gold=unmatched_gold,
                    unmatched_extracted=unmatched_extracted,
                )
            return

        # Gold is dict, extracted isn't (or vice versa)
        # Gold is dict, extracted isn't (or vice versa)
        if isinstance(gold, dict) or isinstance(extracted, dict):
            # If the missing side is None/empty, this is a structural omission
            # or hallucination. Recurse into the present side to emit per-leaf
            # labels so each field of the sub-object gets a label rather than
            # collapsing into a single "type_mismatch" at the array index.
            if gold is None or (isinstance(gold, dict) and len(gold) == 0):
                self._walk(
                    schema=sub_schema,
                    gold=_make_empty_like(extracted),
                    extracted=extracted,
                    path=path,
                    labels=labels,
                    unmatched_gold=unmatched_gold,
                    unmatched_extracted=unmatched_extracted,
                )
                return
            if extracted is None or (isinstance(extracted, dict) and len(extracted) == 0):
                self._walk(
                    schema=sub_schema,
                    gold=gold,
                    extracted=_make_empty_like(gold),
                    path=path,
                    labels=labels,
                    unmatched_gold=unmatched_gold,
                    unmatched_extracted=unmatched_extracted,
                )
                return
            # Both present but types disagree → genuine type mismatch.
            self._emit_type_mismatch(
                path=path, gold=gold, extracted=extracted, labels=labels,
            )
            return
        # ------ Both lists ------
        if isinstance(gold, list) and isinstance(extracted, list):
            self._handle_list_pair(
                schema=sub_schema,
                gold_list=gold,
                extracted_list=extracted,
                path=path,
                labels=labels,
                unmatched_gold=unmatched_gold,
                unmatched_extracted=unmatched_extracted,
            )
            return

        # Gold is list, extracted isn't (or vice versa)
        # Gold is list, extracted isn't (or vice versa)
        if isinstance(gold, list) or isinstance(extracted, list):
            # Same structural treatment as for dicts above: if the other side
            # is None/empty, recurse with an empty list as the substitute so
            # we emit per-element labels.
            if gold is None or (isinstance(gold, list) and len(gold) == 0):
                self._walk(
                    schema=sub_schema,
                    gold=[],
                    extracted=extracted,
                    path=path,
                    labels=labels,
                    unmatched_gold=unmatched_gold,
                    unmatched_extracted=unmatched_extracted,
                )
                return
            if extracted is None or (isinstance(extracted, list) and len(extracted) == 0):
                self._walk(
                    schema=sub_schema,
                    gold=gold,
                    extracted=[],
                    path=path,
                    labels=labels,
                    unmatched_gold=unmatched_gold,
                    unmatched_extracted=unmatched_extracted,
                )
                return
            self._emit_type_mismatch(
                path=path, gold=gold, extracted=extracted, labels=labels,
            )
            return
        
        # ------ Leaf comparison ------
        self._emit_leaf_label(
            schema=sub_schema,
            gold=gold,
            extracted=extracted,
            path=path,
            labels=labels,
        )

    # ------------------------------------------------------------------------
    # List handling
    # ------------------------------------------------------------------------

    def _handle_list_pair(
        self,
        *,
        schema: dict[str, Any] | None,
        gold_list: list[Any],
        extracted_list: list[Any],
        path: list[str | int],
        labels: list[FieldLabel],
        unmatched_gold: list[list[str | int]],
        unmatched_extracted: list[list[str | int]],
    ) -> None:
        """Two cases: arrays of objects (positional) vs primitives (set)."""
        items_schema = (schema or {}).get("items", {}) if schema else {}
        is_object_array = _is_object_array(items_schema, gold_list, extracted_list)

        if is_object_array:
            # Positional alignment.
            n = max(len(gold_list), len(extracted_list))
            for i in range(n):
                g = gold_list[i] if i < len(gold_list) else None
                e = extracted_list[i] if i < len(extracted_list) else None
                if g is None:
                    # Extracted has an extra object at this index → hallucinated
                    # sub-tree. Walk it with an empty gold to surface
                    # hallucination labels.
                    unmatched_extracted.append(path + [i])
                if e is None:
                    unmatched_gold.append(path + [i])
                self._walk(
                    schema=items_schema,
                    gold=g,
                    extracted=e,
                    path=path + [i],
                    labels=labels,
                    unmatched_gold=unmatched_gold,
                    unmatched_extracted=unmatched_extracted,
                )
        else:
            # Primitives → single label, set comparison.
            self._emit_primitive_array_label(
                schema=items_schema,
                gold_list=gold_list,
                extracted_list=extracted_list,
                path=path,
                labels=labels,
            )

    def _emit_primitive_array_label(
        self,
        *,
        schema: dict[str, Any] | None,
        gold_list: list[Any],
        extracted_list: list[Any],
        path: list[str | int],
        labels: list[FieldLabel],
    ) -> None:
        """Compare two arrays of primitives as sets."""
        strategy = self._strategy_for(schema, default=ComparisonStrategy.FUZZY)

        # Treat as case-insensitive sets for robustness across capitalization.
        gold_set = {_normalize_for_set(v) for v in gold_list}
        ext_set = {_normalize_for_set(v) for v in extracted_list}

        gold_present = bool(gold_set)
        extracted_present = bool(ext_set)

        if not gold_present and not extracted_present:
            # Both empty: trivially correct, emit a label so the field is
            # represented in training data. Probe stage may filter.
            is_error = 0
            error_type = "match"
        elif gold_present and not extracted_present:
            is_error = 1
            error_type = "omission"
        elif extracted_present and not gold_present:
            is_error = 1
            error_type = "hallucination"
        else:
            # Both non-empty: set equality.
            is_error = 0 if gold_set == ext_set else 1
            error_type = "match" if is_error == 0 else "value_mismatch"

        labels.append(
            FieldLabel(
                path=list(path),
                path_str=_path_to_string(path),
                gold_value=list(gold_list),
                extracted_value=list(extracted_list),
                is_error=is_error,
                error_type=error_type,
                comparison_strategy=f"set_{strategy.value}",
                gold_present=gold_present,
                extracted_present=extracted_present,
            )
        )

    # ------------------------------------------------------------------------
    # Leaf label emission
    # ------------------------------------------------------------------------

    def _emit_leaf_label(
        self,
        *,
        schema: dict[str, Any] | None,
        gold: Any,
        extracted: Any,
        path: list[str | int],
        labels: list[FieldLabel],
    ) -> None:
        """Compare two primitive values and emit a FieldLabel."""
        gold_present = _has_content(gold)
        extracted_present = _has_content(extracted)

        # Choose strategy from schema.
        strategy = self._strategy_for(schema, default=ComparisonStrategy.EXACT)

        if not gold_present and not extracted_present:
            # Both empty/null. Correct.
            is_error = 0
            error_type = "match"
        elif gold_present and not extracted_present:
            is_error = 1
            error_type = "omission"
        elif extracted_present and not gold_present:
            is_error = 1
            error_type = "hallucination"
        else:
            # Both present: invoke value comparison.
            same = compare_values(
                gold,
                extracted,
                strategy=strategy,
                fuzzy_threshold=self.fuzzy_threshold,
                number_tolerance=self.number_tolerance,
            )
            is_error = 0 if same else 1
            error_type = "match" if same else "value_mismatch"

        labels.append(
            FieldLabel(
                path=list(path),
                path_str=_path_to_string(path),
                gold_value=gold,
                extracted_value=extracted,
                is_error=is_error,
                error_type=error_type,
                comparison_strategy=strategy.value,
                gold_present=gold_present,
                extracted_present=extracted_present,
            )
        )

    def _emit_type_mismatch(
        self,
        *,
        path: list[str | int],
        gold: Any,
        extracted: Any,
        labels: list[FieldLabel],
    ) -> None:
        labels.append(
            FieldLabel(
                path=list(path),
                path_str=_path_to_string(path),
                gold_value=gold,
                extracted_value=extracted,
                is_error=1,
                error_type="type_mismatch",
                comparison_strategy="n/a",
                gold_present=_has_content(gold),
                extracted_present=_has_content(extracted),
            )
        )

    # ------------------------------------------------------------------------
    # Strategy lookup
    # ------------------------------------------------------------------------

    def _strategy_for(
        self,
        schema: dict[str, Any] | None,
        *,
        default: ComparisonStrategy,
    ) -> ComparisonStrategy:
        """Map a schema node's evaluation_config to a ComparisonStrategy."""
        if not schema:
            return default
        ec = schema.get("evaluation_config")
        if not ec:
            return default
        mapped = self._STRATEGY_MAP.get(ec)
        if mapped is None:
            logger.debug(
                "Unknown evaluation_config %r; falling back to %s.",
                ec, default.value,
            )
            return default
        return mapped


# ============================================================================
# Helpers
# ============================================================================

def _has_content(v: Any) -> bool:
    """A value is 'present' if it isn't None, empty string, or empty list/dict."""
    if v is None:
        return False
    if isinstance(v, str) and v.strip() == "":
        return False
    if isinstance(v, (list, dict)) and len(v) == 0:
        return False
    return True

def _make_empty_like(template: Any) -> Any:
    """Return an empty container matching the type of `template`.

    Used when one side of the comparison is missing a sub-object — we
    substitute an empty placeholder of the matching type so the recursion
    can produce per-leaf labels rather than collapsing into a structural
    type_mismatch.
    """
    if isinstance(template, dict):
        return {}
    if isinstance(template, list):
        return []
    return None

def _path_to_string(path: list[str | int]) -> str:
    """Same convention as the extractor uses."""
    return ".".join(str(p) for p in path)


def _normalize_for_set(v: Any) -> str:
    """Normalize a primitive for set-based array comparison.

    Cast to string, strip whitespace, lowercase. Crude but consistent.
    """
    return str(v).strip().lower() if v is not None else ""


def _is_object_array(
    items_schema: dict[str, Any] | None,
    gold_list: list[Any],
    extracted_list: list[Any],
) -> bool:
    """Decide whether to treat a list as an array-of-objects or array-of-primitives.

    Schema says it best, but at the leaf level, both lists may be empty in
    which case we fall back to schema. If schema says items are objects,
    treat as object array. Otherwise treat as primitive array.
    """
    if items_schema:
        item_type = items_schema.get("type")
        if item_type == "object":
            return True
        if item_type in ("string", "number", "integer", "boolean"):
            return False
    # Schema unclear or absent: peek at actual values.
    sample = next(
        (x for x in gold_list + extracted_list if x is not None),
        None,
    )
    return isinstance(sample, dict)


def _resolve_schema_for_value(
    schema: dict[str, Any] | None,
    value: Any,
) -> dict[str, Any] | None:
    """If schema has an `anyOf`, pick the variant that matches the value's type.

    This handles patterns like:
      "skills": {"anyOf": [{"type": "array"}, {"type": "object"}, {"type": "null"}]}

    If we can't pick (no anyOf, or ambiguous), return the schema as-is.
    """
    if not schema or "anyOf" not in schema:
        return schema

    variants = schema["anyOf"]

    # Map Python type → JSON Schema type name.
    if value is None:
        target = "null"
    elif isinstance(value, bool):
        target = "boolean"
    elif isinstance(value, int):
        target = "integer"
    elif isinstance(value, float):
        target = "number"
    elif isinstance(value, str):
        target = "string"
    elif isinstance(value, list):
        target = "array"
    elif isinstance(value, dict):
        target = "object"
    else:
        target = None

    for variant in variants:
        if variant.get("type") == target:
            return variant

    # No exact match; return first variant as best-effort fallback.
    return variants[0] if variants else schema
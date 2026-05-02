"""Extraction orchestrator.

Glues the model wrapper, prompt builder, and parser into a single per-document
operation. The Extractor:

  1. Builds the prompt from a Document's text and schema.
  2. Runs the model with activation capture.
  3. Parses the generated text as JSON.
  4. Locates each leaf field's token positions in the generated output.
  5. Slices the captured hidden states to per-field activations.
  6. Returns an ExtractionResult ready to be saved to disk.

This module does NOT do disk I/O — that's the responsibility of
`scripts/01_extract.py`. Keeping I/O at the boundary makes the Extractor
easy to test and reuse (e.g., from a Jupyter notebook).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from probe_extraction.data import Document
from probe_extraction.extraction.parser import (
    FieldLocation,
    locate_fields,
    parse_json_output,
)
from probe_extraction.extraction.prompts import build_extraction_prompt
from probe_extraction.models.base import LLM

logger = logging.getLogger(__name__)


# ============================================================================
# Result container
# ============================================================================

@dataclass
class FieldExtraction:
    """A single extracted field with its activations.

    Attributes:
        path: JSON path (list of keys/indices). Same as FieldLocation.path.
        path_str: Dotted string form, e.g. "workExperience.0.employer".
            Convenient for use as a dict key, filename, or log line.
        value: The extracted value (primitive: str/int/float/bool/None).
        is_empty: Whether the value was empty/null in the model's output.
        token_span: (start, end) token indices, end-exclusive.
        activations: {layer_index: array of shape (hidden_dim,)}.
            One vector per layer, taken at the position specified by the
            extractor's `position` strategy (e.g., last token of the value).
    """

    path: list[str | int]
    path_str: str
    value: str | int | float | bool | None
    is_empty: bool
    token_span: tuple[int, int]
    activations: dict[int, np.ndarray]


@dataclass
class ExtractionResult:
    """The complete result of running the model on one document.

    Attributes:
        doc_id: Identifier of the source Document.
        domain: Domain of the source Document.
        prompt_token_count: Number of tokens in the prompt (input).
        generated_token_count: Number of tokens generated.
        finish_reason: Why generation stopped ("stop", "length", "error").
        elapsed_seconds: Wall-clock time for generation.
        raw_generated_text: The full string the model generated, before any
            stripping. Saved for debugging and analysis.
        parsed_json: The parsed JSON object (None if parsing failed).
        parse_error: Error message if parsing failed, else None.
        token_logprobs: Per-generated-token log-probabilities. Used by the
            token-logprob baseline.
        fields: Per-field extractions with activations. Empty if parsing
            failed.
        captured_layers: The layer indices at which activations were
            captured. Stored so downstream stages know what's available.
    """

    doc_id: str
    domain: str
    prompt_token_count: int
    generated_token_count: int
    finish_reason: str
    elapsed_seconds: float
    raw_generated_text: str
    parsed_json: dict[str, Any] | None
    parse_error: str | None
    token_logprobs: list[float] | None
    fields: list[FieldExtraction] = field(default_factory=list)
    captured_layers: list[int] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        """True if the model produced parseable JSON with at least one field."""
        return self.parse_error is None and len(self.fields) > 0


# ============================================================================
# Extractor
# ============================================================================

class Extractor:
    """Run extraction on documents using a given model.

    Args:
        llm: An LLM implementation (e.g., HuggingFaceLLM).
        layers: Layer indices to capture activations from.
        position: Which position within a field's token span to read the
            activation from. Currently supports:
              - "last_token": the activation at the field's last token
                  (most common, captures the model's "final" representation
                  of the value)
              - "mean": elementwise mean across all tokens in the field's span
        max_new_tokens: Cap on tokens per generation.
        temperature: Sampling temperature (0 = greedy).
        top_p: Nucleus sampling parameter (used only if temperature > 0).
        include_schema: Whether to include the schema in the prompt.

    Notes:
        - The Extractor does NOT save anything to disk. It returns
          ExtractionResult objects for the caller to persist.
        - One LLM is shared across calls; do not construct a new Extractor
          for each document (model reload is expensive).
    """

    SUPPORTED_POSITIONS = ("last_token", "mean")

    def __init__(
        self,
        llm: LLM,
        *,
        layers: list[int],
        position: str = "last_token",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        include_schema: bool = True,
    ) -> None:
        if position not in self.SUPPORTED_POSITIONS:
            raise ValueError(
                f"Unsupported position {position!r}. "
                f"Must be one of {self.SUPPORTED_POSITIONS}."
            )

        self.llm = llm
        self.layers = sorted(set(layers))
        self.position = position
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.include_schema = include_schema

        # Sanity-check requested layers against the model up-front so we fail
        # before the first expensive forward pass.
        for ℓ in self.layers:
            if not (1 <= ℓ <= self.llm.num_layers):
                raise ValueError(
                    f"Layer {ℓ} out of range for model {self.llm.name} "
                    f"(has {self.llm.num_layers} layers)"
                )

    # ------------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------------

    def extract(self, doc: Document) -> ExtractionResult:
        """Run extraction on a single document.

        Returns an ExtractionResult regardless of success/failure. Failures
        produce a result with parse_error set and an empty fields list, so
        the caller can save partial state and report consistently.
        """
        if doc.extraction_error is not None:
            # Document had no usable text (image-only PDF, etc.). We don't
            # run the model in this case — there's nothing to extract from.
            logger.info(
                "Skipping %s (no document text: %s)",
                doc.doc_id, doc.extraction_error,
            )
            return ExtractionResult(
                doc_id=doc.doc_id,
                domain=doc.domain,
                prompt_token_count=0,
                generated_token_count=0,
                finish_reason="error",
                elapsed_seconds=0.0,
                raw_generated_text="",
                parsed_json=None,
                parse_error=f"document has no text: {doc.extraction_error}",
                token_logprobs=None,
                captured_layers=list(self.layers),
            )

        # ------ Build prompt ------
        system_msg, user_msg = build_extraction_prompt(
            schema=doc.schema,
            document_text=doc.text,
            include_schema=self.include_schema,
        )
        prompt = self.llm.format_chat(system_msg, user_msg)

        # ------ Run model ------
        logger.info("Extracting %s (text=%d chars)", doc.doc_id, len(doc.text))
        start = time.perf_counter()
        gen_output = self.llm.generate_with_activations(
            prompt=prompt,
            layers=self.layers,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            return_logprobs=True,
        )
        elapsed = time.perf_counter() - start

        logger.info(
            "Generated %d tokens in %.1fs (%.1f tok/s, finish=%s)",
            len(gen_output.generated_token_ids),
            elapsed,
            len(gen_output.generated_token_ids) / max(elapsed, 1e-6),
            gen_output.finish_reason,
        )

        # ------ Parse JSON ------
        parsed_json, parse_error, json_text = parse_json_output(gen_output.text)
        if parsed_json is None:
            logger.warning(
                "JSON parse failed for %s: %s", doc.doc_id, parse_error,
            )
            return ExtractionResult(
                doc_id=doc.doc_id,
                domain=doc.domain,
                prompt_token_count=gen_output.metadata["prompt_len"],
                generated_token_count=len(gen_output.generated_token_ids),
                finish_reason=gen_output.finish_reason,
                elapsed_seconds=elapsed,
                raw_generated_text=gen_output.text,
                parsed_json=None,
                parse_error=parse_error,
                token_logprobs=gen_output.token_logprobs,
                captured_layers=list(self.layers),
            )

        # ------ Locate fields in tokens ------
        per_token_strings = self.llm.decode_per_token(gen_output.generated_token_ids)
        try:
            field_locations = locate_fields(
                parsed_json=parsed_json,
                json_text=json_text,
                generated_token_ids=gen_output.generated_token_ids,
                per_token_strings=per_token_strings,
            )
        except Exception as e:
            # Localization is best-effort; if it crashes (e.g., due to a
            # weird JSON shape), we still return the parsed result without
            # field-level activations.
            logger.exception("Field localization failed for %s: %s", doc.doc_id, e)
            field_locations = []

        # ------ Slice activations per field ------
        fields = self._slice_activations(
            field_locations=field_locations,
            hidden_states=gen_output.hidden_states or {},
            num_generated=len(gen_output.generated_token_ids),
        )

        return ExtractionResult(
            doc_id=doc.doc_id,
            domain=doc.domain,
            prompt_token_count=gen_output.metadata["prompt_len"],
            generated_token_count=len(gen_output.generated_token_ids),
            finish_reason=gen_output.finish_reason,
            elapsed_seconds=elapsed,
            raw_generated_text=gen_output.text,
            parsed_json=parsed_json,
            parse_error=None,
            token_logprobs=gen_output.token_logprobs,
            fields=fields,
            captured_layers=list(self.layers),
        )

    # ------------------------------------------------------------------------
    # Internal: activation slicing
    # ------------------------------------------------------------------------

    def _slice_activations(
        self,
        *,
        field_locations: list[FieldLocation],
        hidden_states: dict[int, np.ndarray],
        num_generated: int,
    ) -> list[FieldExtraction]:
        """For each field, extract a single hidden-state vector per layer.

        Each layer in `hidden_states` has shape (num_generated, hidden_dim).
        For each field with token span [start, end), we select position(s)
        within that span according to self.position and reduce to a single
        vector per layer.
        """
        fields: list[FieldExtraction] = []
        for loc in field_locations:
            # Defensive bounds — a logic bug in the locator could produce
            # out-of-range indices; we'd rather log and skip than crash.
            start = max(0, min(loc.start_token_idx, num_generated - 1))
            end = max(start + 1, min(loc.end_token_idx, num_generated))

            per_layer_vec: dict[int, np.ndarray] = {}
            for ℓ, layer_array in hidden_states.items():
                # layer_array shape: (num_generated, hidden_dim)
                if layer_array.shape[0] != num_generated:
                    logger.warning(
                        "Layer %d activation length mismatch: got %d, "
                        "expected %d. Skipping field %s.",
                        ℓ, layer_array.shape[0], num_generated, loc.path,
                    )
                    continue

                span = layer_array[start:end]  # (span_len, hidden_dim)
                if span.size == 0:
                    continue

                if self.position == "last_token":
                    vec = span[-1]
                elif self.position == "mean":
                    # Cast to float32 for the mean to avoid fp16 underflow,
                    # then back to fp16 for storage consistency.
                    vec = span.astype(np.float32).mean(axis=0).astype(np.float16)
                else:
                    raise AssertionError(f"Unhandled position: {self.position}")

                per_layer_vec[ℓ] = vec

            if not per_layer_vec:
                # Couldn't produce any layer's activation; drop the field
                # rather than emit a bogus empty-activations record.
                logger.warning(
                    "No activations produced for field %s; skipping.", loc.path,
                )
                continue

            fields.append(
                FieldExtraction(
                    path=list(loc.path),
                    path_str=_path_to_string(loc.path),
                    value=loc.value,
                    is_empty=loc.is_empty,
                    token_span=(start, end),
                    activations=per_layer_vec,
                )
            )
        return fields


# ============================================================================
# Helpers
# ============================================================================

def _path_to_string(path: list[str | int]) -> str:
    """Convert a JSON path to a dotted string, e.g.,
    ['workExperience', 0, 'employer'] -> 'workExperience.0.employer'.
    """
    return ".".join(str(p) for p in path)
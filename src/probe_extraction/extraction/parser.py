"""Parsing the model's generated output and locating field values in tokens.

This module solves a small but important alignment problem. The model
generates a sequence of tokens that decodes to (approximately) JSON like:

    {"title": "Some Paper", "authors": ["A", "B"], "year": 2024}

We need:
  1. The parsed JSON value (for the labeling stage to compare to gold).
  2. For each leaf field in that JSON, the TOKEN POSITIONS in the generated
     output where that field's VALUE was produced. The probe is trained on
     activations at those positions.

The challenge: tokenization is opaque. The token boundaries don't align with
JSON value boundaries in general. A field's value might span multiple tokens,
or one token might contain part of a value plus part of a delimiter.

Strategy:
  - Parse JSON with json.loads (handles nested objects/arrays naturally).
  - Walk the parsed JSON, recording each leaf field's path and value.
  - For each leaf, find where its value appears in the GENERATED TEXT
    (string search), then map character offsets to token offsets using
    per-token decoded strings.
  - For each leaf, store (start_token_idx, end_token_idx) — both inclusive
    of the value, exclusive of surrounding delimiters/whitespace.

Edge cases handled:
  - Model wraps output in ```json ... ``` fences.
  - Model emits trailing text after the JSON.
  - Same value appears multiple times (we use sequential search to disambiguate).
  - Empty values (null, [], "", 0): given a synthetic single-position location
    so probe training has *something*; flagged via FieldLocation.is_empty.

Out-of-scope:
  - Repairing malformed JSON. If json.loads fails, we report it and the
    document is dropped from probe training. Quantifying this rate is itself
    informative for the analysis.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Iterator

logger = logging.getLogger(__name__)


# ============================================================================
# Public types
# ============================================================================

@dataclass
class FieldLocation:
    """Where a single leaf field's value lives in the generated tokens.

    Attributes:
        path: JSON path as a list of keys/indices. E.g.,
            ["personalInfo", "fullName"] or ["workExperience", 0, "employer"].
            Tuples could be more 'correct' but lists serialize cleanly to JSON.
        value: The parsed JSON value at this path. Primitive (str/int/float/
            bool/None) — never a dict or list. Containers are walked into.
        start_token_idx: Index into generated_token_ids where the value
            begins (inclusive). 0-indexed.
        end_token_idx: Index into generated_token_ids where the value ends
            (exclusive). For a value spanning tokens 5,6,7: start=5, end=8.
        is_empty: True if the value is null, empty string, or a numeric
            sentinel zero in a context where 0 is unlikely. We still keep
            a single-token location for the probe to read SOMETHING, but
            flag it so downstream stages can treat empties separately.
        char_start: Character offset of the value within the generated text.
            Useful for debugging/inspection.
        char_end: Character offset (exclusive) within the generated text.
    """

    path: list[str | int]
    value: str | int | float | bool | None
    start_token_idx: int
    end_token_idx: int
    is_empty: bool
    char_start: int
    char_end: int


@dataclass
class ParseResult:
    """Result of parsing a model's generated text.

    Attributes:
        parsed_json: The parsed JSON object (None if parsing failed).
        parse_error: Error message if parsing failed; else None.
        field_locations: Token-aligned locations for every leaf field. Empty
            list if parsing failed.
        json_text: The substring of the generated text that we treated as
            JSON (after stripping fences/preamble). Useful for debugging.
    """

    parsed_json: dict[str, Any] | None
    parse_error: str | None
    field_locations: list[FieldLocation]
    json_text: str


# ============================================================================
# Public entry points
# ============================================================================

def parse_json_output(generated_text: str) -> tuple[dict[str, Any] | None, str | None, str]:
    """Parse model output as JSON with defensive type handling."""
    if generated_text is None:
        return None, "model produced no text (None)", ""
    
    candidate = _strip_to_json(generated_text)
    
    if candidate is None or not isinstance(candidate, str):
        return None, f"strip_to_json returned non-string: {type(candidate).__name__}", ""
    
    if not candidate.strip():
        return None, "no JSON content found in output", candidate
    
    try:
        return json.loads(candidate), None, candidate
    except json.JSONDecodeError as e:
        return None, f"JSONDecodeError: {e}", candidate
    except (TypeError, ValueError) as e:
        return None, f"{type(e).__name__}: {e}", candidate

def locate_fields(
    *,
    parsed_json: dict[str, Any],
    json_text: str,
    generated_token_ids: list[int],
    per_token_strings: list[str],
) -> list[FieldLocation]:
    """Map each leaf JSON value to its (start, end) token positions.

    Args:
        parsed_json: The successfully-parsed JSON object.
        json_text: The substring of generated text that was parsed (used for
            character-offset finding). NOTE: this is the JSON content only,
            with any pre/post-amble stripped.
        generated_token_ids: Token IDs the model produced (full generation,
            including any pre/post-amble around the JSON).
        per_token_strings: Decoded surface form of each token, aligned 1-to-1
            with generated_token_ids.

    Returns:
        A FieldLocation for each leaf value found. May be shorter than the
        full set of leaves if some values can't be located (logged as warnings).
    """
    if len(generated_token_ids) != len(per_token_strings):
        raise ValueError(
            f"Token IDs and per-token strings must have same length: "
            f"got {len(generated_token_ids)} vs {len(per_token_strings)}"
        )

    # Reconstruct the full generated text from per-token strings, then find
    # where the JSON content sits within it. This gives a base offset to
    # add to per-value char offsets so they index into the *full* generated
    # text — which is what aligns to per_token_strings.
    full_text = "".join(per_token_strings)
    json_offset = full_text.find(json_text)
    if json_offset < 0:
        # Reconstruction mismatch (should be rare; happens with whitespace/
        # special-token quirks). Fall back to searching the whole stream.
        logger.warning(
            "Could not locate json_text within reconstructed full text; "
            "falling back to whole-stream search."
        )
        json_offset = 0
        full_text = json_text  # search inside json_text only

    # Build a char-offset → token-index lookup table.
    token_starts = _compute_token_char_starts(per_token_strings)

    locations: list[FieldLocation] = []
    cursor = 0  # advances as we find each leaf, ensuring sequential matching

    for path, value in _iter_leaves(parsed_json):
        loc = _locate_one_value(
            value=value,
            path=path,
            json_text=json_text,
            json_offset=json_offset,
            full_text=full_text,
            token_starts=token_starts,
            num_tokens=len(per_token_strings),
            search_from=cursor,
        )
        if loc is None:
            logger.warning(
                "Could not locate value for path %s (value=%r) in tokens",
                path, value,
            )
            continue

        locations.append(loc)
        cursor = loc.char_end  # next search starts after this value

    return locations


# ============================================================================
# Internal helpers
# ============================================================================

# Common ways models wrap JSON output. We try each in order.
_FENCE_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)


def _strip_to_json(text: str) -> str:
    """Best-effort extraction of a JSON object from raw model output.

    Handles, in priority order:
      0. Strips any <think>...</think> reasoning blocks (Qwen3+ thinking mode).
      1. ```json ... ``` fenced blocks.
      2. ``` ... ``` fenced blocks (no language tag).
      3. The first '{' to the matching closing '}', with brace-counting.
      4. Trim whitespace and return as-is (let json.loads decide).
    """
    text = text.strip()

    # 0: Strip any <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 1 & 2: fenced code block
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()

    # 3: brace-counted extraction
    first_brace = text.find("{")
    if first_brace >= 0:
        extracted = _extract_balanced_braces(text, first_brace)
        if extracted is not None:
            return extracted

    # 4: fall through
    return text
def _extract_balanced_braces(text: str, start: int) -> str | None:
    """Return the substring from `start` through the matching closing brace.

    Handles strings (so braces inside strings don't count) and escapes.
    Returns None if no balanced match found.
    """
    depth = 0
    i = start
    in_string = False
    escape = False
    while i < len(text):
        c = text[i]
        if escape:
            escape = False
        elif c == "\\":
            escape = True
        elif c == '"':
            in_string = not in_string
        elif not in_string:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        i += 1
    return None


def _iter_leaves(
    obj: Any,
    path: list[str | int] | None = None,
) -> Iterator[tuple[list[str | int], Any]]:
    """Walk a parsed JSON value, yielding (path, leaf_value) tuples.

    Leaves are primitives (str, int, float, bool, None). Dicts and lists are
    recursed into. Empty containers do NOT yield anything (no path leads to
    a leaf inside them).
    """
    if path is None:
        path = []

    if isinstance(obj, dict):
        for key, child in obj.items():
            yield from _iter_leaves(child, path + [key])
    elif isinstance(obj, list):
        for i, child in enumerate(obj):
            yield from _iter_leaves(child, path + [i])
    else:
        # Primitive: str, int, float, bool, None
        yield path, obj


def _compute_token_char_starts(per_token_strings: list[str]) -> list[int]:
    """For each token, the character offset where it starts in the
    concatenation of all token strings.

    Example: ["He", "llo", " world"] -> [0, 2, 5]
    """
    starts: list[int] = []
    cursor = 0
    for s in per_token_strings:
        starts.append(cursor)
        cursor += len(s)
    return starts


def _locate_one_value(
    *,
    value: Any,
    path: list[str | int],
    json_text: str,
    json_offset: int,
    full_text: str,
    token_starts: list[int],
    num_tokens: int,
    search_from: int,
) -> FieldLocation | None:
    """Locate a single leaf value's character span in `full_text`, then
    convert to token indices.

    Sequential search (`search_from`) ensures we match values in document
    order even when the same string appears multiple times. E.g., two work
    experiences with employer "KAIST" — we want the first match for the
    first job, the second match for the second job.
    """
    is_empty, search_str = _value_to_search_string(value)

    if search_str is None:
        # Truly empty (null / empty array). We can't search for anything;
        # synthesize a location at the current cursor so the probe sees
        # *something*. This is imperfect but better than dropping the field.
        synthetic_token = _char_offset_to_token_idx(
            search_from, token_starts, num_tokens,
        )
        return FieldLocation(
            path=path,
            value=value,  # type: ignore[arg-type]
            start_token_idx=synthetic_token,
            end_token_idx=min(synthetic_token + 1, num_tokens),
            is_empty=True,
            char_start=search_from,
            char_end=search_from,
        )

    # For non-empty values: search within the JSON text (so we don't match
    # accidental occurrences in pre/post-amble), then offset by json_offset
    # to land in full_text coordinates.
    local_idx = json_text.find(search_str, search_from - json_offset
                               if search_from >= json_offset else 0)
    if local_idx < 0:
        return None

    char_start = json_offset + local_idx
    char_end = char_start + len(search_str)

    start_token = _char_offset_to_token_idx(char_start, token_starts, num_tokens)
    end_token = _char_offset_to_token_idx(char_end, token_starts, num_tokens)
    # Ensure end > start (at least 1 token wide).
    if end_token <= start_token:
        end_token = min(start_token + 1, num_tokens)

    return FieldLocation(
        path=path,
        value=value,
        start_token_idx=start_token,
        end_token_idx=end_token,
        is_empty=is_empty,
        char_start=char_start,
        char_end=char_end,
    )


def _value_to_search_string(value: Any) -> tuple[bool, str | None]:
    """Convert a JSON leaf value to the string we expect to find verbatim
    in the generated JSON text.

    Returns (is_empty, search_string). search_string is None for values
    that have no surface form to search for (null, empty string).
    """
    if value is None:
        return True, None
    if isinstance(value, str):
        if value == "":
            return True, None
        return False, value  # raw string; we DON'T re-escape because we're
                              # searching the JSON text which is already
                              # pre-escape on our end. Imperfect for strings
                              # containing quotes, but rare in practice.
    if isinstance(value, bool):
        return False, "true" if value else "false"
    if isinstance(value, (int, float)):
        # Render as JSON would: integers without decimals, floats with.
        return False, json.dumps(value)
    return False, str(value)  # fallback


def _char_offset_to_token_idx(
    char_offset: int,
    token_starts: list[int],
    num_tokens: int,
) -> int:
    """Find the token index containing the given character offset.

    Linear search is fine for our scale (< 2000 tokens per document).
    Returns the LAST token whose start <= char_offset, or num_tokens-1 as
    a fallback if char_offset is past the end.
    """
    if not token_starts:
        return 0
    if char_offset >= token_starts[-1]:
        return num_tokens - 1

    # Find largest i such that token_starts[i] <= char_offset.
    last = 0
    for i, start in enumerate(token_starts):
        if start <= char_offset:
            last = i
        else:
            break
    return last
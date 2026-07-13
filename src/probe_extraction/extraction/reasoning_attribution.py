"""Field-localized reasoning attribution for reasoning-model trust signals.

Document-level reasoning pooling (reasoning_trace.reasoning_pooled_vectors) is
constant across a document's fields, so it cannot change *within-document* field
ranking (empirically ~null; see the reasoning-fusion LODO result). This module
localizes the reasoning to each *field*: it finds where a field's extracted
VALUE is mentioned inside the `<think>...</think>` trace and pools the hidden
states of exactly those tokens. The resulting vector VARIES across fields in a
document, so it is a legitimate within-document error signal.

Two complementary signals are produced per field:

  1. attr_vec  {layer: (hidden_dim,)} -- mean of the reasoning-token hidden
     states where the value is mentioned (the "how did the model reason about
     this value" representation). Zero vector if the value is never mentioned.

  2. features  {name: float} -- interpretable scalars, chiefly "was this value
     reasoned about at all, how often, and where". A value that appears in the
     final JSON but is ABSENT from the reasoning trace is a hallucination
     red-flag. These are cheap, model-agnostic, and human-readable.

Kept dependency-light (numpy + stdlib re only, no torch / no package imports) so
the matching + pooling logic is unit-testable on CPU without the model stack.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

# Scalar feature order is fixed so callers can build a stable (n_fields, k)
# matrix. Keep in sync with attribute_field()'s `features` dict.
FEATURE_NAMES: tuple[str, ...] = (
    "mentioned",        # 1.0 if the value (full or partial) is found in the trace
    "match_full",       # 1.0 if the FULL value string was found (vs partial word)
    "mention_count",    # number of matched occurrences
    "n_matched_tokens", # reasoning tokens covered by matches
    "first_frac",       # first mention start position / trace length in [0,1]
    "last_frac",        # last mention end position / trace length in [0,1]
    "value_char_len",   # length of the (normalized) value string (a control)
)


def _normalize(s: str) -> str:
    """Lowercase + collapse whitespace. Matching is done on normalized text so
    'Johann  Sebastian  Bach' and 'johann sebastian bach' align."""
    return re.sub(r"\s+", " ", s.strip().lower())


def value_to_text(value: Any) -> str:
    """Render an extracted leaf value as the string we search the trace for.

    Booleans/None render to '' (not searchable as a mention). Numbers render
    without trailing '.0' so an integer-valued float still matches '5' in text.
    """
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _char_to_token(token_strings: list[str]) -> tuple[str, np.ndarray]:
    """Concatenate token surface strings into the trace text and build a
    char-index -> token-index map (normalized to lowercase, whitespace kept as
    single spaces to match _normalize on both sides).

    Returns (norm_text, owner) where owner[c] is the token index owning char c.
    """
    pieces: list[str] = []
    owners: list[int] = []
    for i, tok in enumerate(token_strings):
        # Normalize each token the same way as the query, char-by-char, so the
        # owner map stays aligned with norm_text.
        norm = re.sub(r"\s+", " ", tok.lower())
        for ch in norm:
            pieces.append(ch)
            owners.append(i)
    # Collapse the boundary spaces the same way _normalize would (a token that
    # is pure whitespace becomes a single space; adjacent spaces stay adjacent).
    text = "".join(pieces)
    owner = np.array(owners, dtype=np.int64)
    return text, owner


def _find_spans(norm_text: str, needle: str) -> list[tuple[int, int]]:
    """All [start, end) char spans of `needle` in `norm_text`. Word-bounded
    when the needle's edges are word characters (avoids 'art' matching inside
    'Mozart'); plain substring otherwise (needle starts/ends with punctuation).
    """
    if not needle:
        return []
    esc = re.escape(needle)
    left = r"\b" if needle[0].isalnum() else ""
    right = r"\b" if needle[-1].isalnum() else ""
    pattern = f"{left}{esc}{right}"
    return [(m.start(), m.end()) for m in re.finditer(pattern, norm_text)]


def _longest_word(norm_value: str, min_len: int = 4) -> str:
    """The longest alphanumeric word in the value, for a partial fallback match
    (e.g. full 'lake tahoe, california' absent but 'california' present)."""
    words = re.findall(r"[a-z0-9]+", norm_value)
    words = [w for w in words if len(w) >= min_len]
    return max(words, key=len) if words else ""


def attribute_field(
    token_strings: list[str],
    reasoning_states: dict[int, np.ndarray],
    value: Any,
    *,
    hidden_dim_by_layer: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Attribute one field's value to the reasoning trace.

    Args:
        token_strings: surface strings of the reasoning-trace tokens, in order
            (the tokens BEFORE </think>). len == n_reasoning_tokens.
        reasoning_states: {layer: (n_reasoning_tokens, hidden_dim)} hidden
            states for those same tokens. Layers may be a subset.
        value: the field's extracted leaf value.
        hidden_dim_by_layer: optional {layer: dim} to size zero fallback vectors
            when nothing matches; inferred from reasoning_states if omitted.

    Returns:
        {"attr_vec": {layer: (hidden_dim,) float16},
         "features": {name: float for name in FEATURE_NAMES},
         "match_type": "full"|"partial"|"none"}
    """
    dims = hidden_dim_by_layer or {L: arr.shape[1] for L, arr in reasoning_states.items()}
    zero_vec = {L: np.zeros(d, dtype=np.float16) for L, d in dims.items()}
    empty_features = {name: 0.0 for name in FEATURE_NAMES}

    norm_value = _normalize(value_to_text(value))
    if not norm_value or not token_strings or not reasoning_states:
        return {"attr_vec": zero_vec, "features": empty_features, "match_type": "none"}

    norm_text, owner = _char_to_token(token_strings)
    n_tok = len(token_strings)
    trace_len = max(len(norm_text), 1)

    spans = _find_spans(norm_text, norm_value)
    match_type = "full"
    if not spans:
        word = _longest_word(norm_value)
        spans = _find_spans(norm_text, word) if word else []
        match_type = "partial" if spans else "none"

    if not spans:
        feats = dict(empty_features)
        feats["value_char_len"] = float(len(norm_value))
        return {"attr_vec": zero_vec, "features": feats, "match_type": "none"}

    # Collect the reasoning-token indices covered by any matched span.
    matched: set[int] = set()
    for c0, c1 in spans:
        c1c = min(c1, len(owner))
        if c1c > c0:
            matched.update(int(t) for t in owner[c0:c1c])
    matched_idx = sorted(t for t in matched if 0 <= t < n_tok)

    attr_vec: dict[int, np.ndarray] = {}
    for L, arr in reasoning_states.items():
        idx = [t for t in matched_idx if t < arr.shape[0]]
        if idx:
            attr_vec[L] = arr[idx].astype(np.float32).mean(axis=0).astype(np.float16)
        else:
            attr_vec[L] = zero_vec[L]

    first_start = min(s for s, _ in spans)
    last_end = max(e for _, e in spans)
    features = {
        "mentioned": 1.0,
        "match_full": 1.0 if match_type == "full" else 0.0,
        "mention_count": float(len(spans)),
        "n_matched_tokens": float(len(matched_idx)),
        "first_frac": first_start / trace_len,
        "last_frac": min(last_end, trace_len) / trace_len,
        "value_char_len": float(len(norm_value)),
    }
    return {"attr_vec": attr_vec, "features": features, "match_type": match_type}


def features_to_array(features: dict[str, float]) -> np.ndarray:
    """Pack a features dict into a fixed-order (len(FEATURE_NAMES),) vector."""
    return np.array([features.get(n, 0.0) for n in FEATURE_NAMES], dtype=np.float32)

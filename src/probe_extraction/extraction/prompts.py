"""Prompt construction for structured extraction.

The prompt's job is to make the model produce a JSON object conforming to a
provided JSON Schema, populated from a given document. We aim for:

  - Deterministic, JSON-only output (no preamble, no markdown fences).
  - Fields filled from the document; absent info → null/empty array.
  - No hallucinated values.

We don't use few-shot examples for now: the schemas are domain-specific and
crafting good in-context examples per domain adds complexity without obvious
benefit at sandbox scale. Easy to add later via the same module.
"""

from __future__ import annotations

import json
from typing import Any


# ============================================================================
# Templates
# ============================================================================

SYSTEM_MESSAGE = """You are a careful data extraction assistant. \
Given a document and a JSON Schema, you extract structured information from \
the document into JSON that conforms to the schema. You are precise and \
conservative: you do not invent information. If a field is not present in \
the document, you use null (for nullable fields) or an empty array (for \
arrays). Your output is ONLY a JSON object, with no surrounding text, \
explanation, or markdown fences."""


USER_TEMPLATE = """Extract structured information from the following document \
according to the JSON Schema below.

# JSON Schema

```json
{schema_json}
```

# Document

{document_text}

# Instructions

- Output ONLY a JSON object conforming to the schema.
- Do not include markdown fences, comments, or any text outside the JSON.
- Use null for missing nullable fields and [] for missing arrays.
- Do not invent information that is not in the document.

# Output

"""


# ============================================================================
# Public entry point
# ============================================================================

def build_extraction_prompt(
    *,
    schema: dict[str, Any],
    document_text: str,
    include_schema: bool = True,
) -> tuple[str, str]:
    """Build (system_message, user_message) for an extraction prompt.

    The model wrapper's `format_chat()` then applies the model's chat
    template to wrap these into a single prompt string.

    Args:
        schema: JSON Schema (dict) describing the extraction target.
        document_text: The text of the document to extract from.
        include_schema: If False, omits the schema from the prompt. Used for
            ablation studies; default True for normal operation.

    Returns:
        (system_message, user_message) tuple.
    """
    if include_schema:
        schema_json = json.dumps(schema, indent=2, ensure_ascii=False)
        user_message = USER_TEMPLATE.format(
            schema_json=schema_json,
            document_text=document_text,
        )
    else:
        # Schema-free variant: ablation only.
        user_message = (
            "Extract structured information from the following document "
            "into JSON.\n\n# Document\n\n"
            f"{document_text}\n\n# Output\n\n"
        )

    return SYSTEM_MESSAGE, user_message
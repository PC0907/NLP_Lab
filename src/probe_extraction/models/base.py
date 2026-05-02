"""Abstract LLM interface.

Defines the contract any backend must satisfy. The pipeline interacts only
through this interface, so swapping HuggingFace for vLLM (or any other
backend) requires only a new class implementing these methods.

The key capability is `generate_with_activations`: a single forward pass that
produces both the generated text AND the hidden states at requested layers,
positioned per generated token. Activations are the basis for probe training.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


# ============================================================================
# Output container
# ============================================================================

@dataclass
class GenerationOutput:
    """Result of a single generation call.

    Attributes:
        text: The decoded generated text (does NOT include the prompt).
        prompt_token_ids: Token IDs of the prompt (input). Useful for
            position-arithmetic when locating fields in the generated portion.
        generated_token_ids: Token IDs of just the generated portion.
        token_logprobs: Per-token log-probability of the chosen tokens
            (length = len(generated_token_ids)). Used by the
            `token_logprob` baseline. None if the backend doesn't expose them.
        hidden_states: Mapping {layer_index -> array of shape
            (generated_seq_len, hidden_dim)}. Populated only by
            generate_with_activations. The generated_seq_len axis aligns
            1-to-1 with generated_token_ids: hidden_states[ℓ][t] is the
            residual-stream vector at layer ℓ, at the position where
            generated_token_ids[t] was produced.
        finish_reason: Why generation stopped. One of:
            "stop" (EOS token), "length" (hit max_new_tokens), "error".
        metadata: Backend-specific extras (timing, memory, etc.). Not used by
            the pipeline; available for analysis.
    """

    text: str
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    token_logprobs: list[float] | None = None
    hidden_states: dict[int, np.ndarray] | None = None
    finish_reason: str = "stop"
    metadata: dict[str, Any] | None = None


# ============================================================================
# Abstract LLM
# ============================================================================

class LLM(ABC):
    """Abstract base class for language model wrappers.

    Subclasses implement the actual generation and activation-capture logic.
    The contract is intentionally minimal: a name, a tokenizer-like decode
    helper for converting token IDs back to strings (needed by the field
    locator), and the two generation methods.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier of the underlying model.

        Should match the HuggingFace repo ID where applicable
        (e.g., "Qwen/Qwen2.5-7B-Instruct").
        """
        ...

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Number of transformer layers (excluding embedding layer).

        For Qwen2.5-7B this is 28. Used to validate requested layer indices
        before construction time.
        """
        ...

    @property
    @abstractmethod
    def hidden_dim(self) -> int:
        """Dimensionality of the residual stream.

        For Qwen2.5-7B this is 3584. Used by the probe-training stage.
        """
        ...

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """Decode a list of token IDs back to a string.

        Used by the field locator to reconstruct text from tokens and align
        field positions to token positions.
        """
        ...

    @abstractmethod
    def decode_per_token(self, token_ids: list[int]) -> list[str]:
        """Decode each token individually to its surface form.

        Returns a list aligned 1-to-1 with `token_ids`. The concatenation of
        these strings should reproduce `decode(token_ids)`. Used for
        token-level position alignment when locating field boundaries.
        """
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        return_logprobs: bool = False,
    ) -> GenerationOutput:
        """Run generation without capturing hidden states.

        Lighter than generate_with_activations: avoids storing per-layer
        tensors. Use for baselines that need only the text and token
        logprobs (e.g., self-consistency).

        Args:
            prompt: The input string (already chat-template-formatted by the
                caller, if applicable; this method does NOT apply chat
                templates — see `format_chat` for that).
            max_new_tokens: Cap on generated tokens.
            temperature: 0 = greedy. > 0 enables sampling.
            top_p: Nucleus sampling parameter. Ignored when temperature == 0.
            return_logprobs: If True, populate `token_logprobs` in the output.

        Returns:
            GenerationOutput with hidden_states=None.
        """
        ...

    @abstractmethod
    def generate_with_activations(
        self,
        prompt: str,
        *,
        layers: list[int],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        return_logprobs: bool = True,
    ) -> GenerationOutput:
        """Run generation AND capture hidden states at requested layers.

        Single forward pass: text and activations come from the same
        generation. The cost over plain `generate` is the memory and copy
        time for storing per-layer tensors.

        Args:
            prompt: Input string.
            layers: Layer indices to capture (1-indexed; layer 0 is the
                embedding layer and is excluded by convention). Must all be
                in [1, num_layers].
            max_new_tokens: Cap on generated tokens.
            temperature: 0 = greedy. > 0 enables sampling.
            top_p: Nucleus sampling parameter.
            return_logprobs: If True, populate `token_logprobs`.

        Returns:
            GenerationOutput with hidden_states populated. Each entry is a
            (generated_seq_len, hidden_dim) numpy array.
        """
        ...

    @abstractmethod
    def format_chat(
        self,
        system_message: str | None,
        user_message: str,
    ) -> str:
        """Apply the model's chat template to (system, user) messages.

        Different model families use different chat templates (Qwen, Llama,
        Mistral all differ). Centralizing this here means callers don't need
        to know which template applies.
        """
        ...
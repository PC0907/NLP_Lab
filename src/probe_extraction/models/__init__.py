"""LLM wrappers.

Provides a uniform interface for running causal LMs and capturing hidden-state
activations during generation. Concrete implementations wrap specific backends
(currently HuggingFace transformers).
"""

from probe_extraction.models.base import GenerationOutput, LLM
from probe_extraction.models.hf_model import HuggingFaceLLM

__all__ = ["GenerationOutput", "LLM", "HuggingFaceLLM"]
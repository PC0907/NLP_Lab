"""HuggingFace causal-LM wrapper with activation capture.

Wraps a transformers AutoModelForCausalLM with:
  - Quantization support (4-bit, 8-bit via bitsandbytes; or none)
  - Chat template formatting (model-specific)
  - Generation with optional per-layer hidden-state capture
  - Per-token logprobs

Activation capture details (this is the part most likely to confuse later, so
read carefully):

  When we call `model.generate(..., output_hidden_states=True,
  return_dict_in_generate=True)`, transformers returns hidden states
  organized as a TUPLE OVER GENERATION STEPS, where each element is itself a
  tuple of layer tensors:

      hidden_states = (
          # Step 0: prompt forward pass
          (h0_layer0, h0_layer1, ..., h0_layerN),    # shape per tensor:
                                                      # (batch, prompt_len, hidden_dim)
          # Step 1: first generated token
          (h1_layer0, h1_layer1, ..., h1_layerN),    # shape per tensor:
                                                      # (batch, 1, hidden_dim)
          # Step 2: second generated token
          (h2_layer0, ...),                          # shape per tensor:
                                                      # (batch, 1, hidden_dim)
          ...
      )

  - The OUTER tuple has length 1 + num_generated_tokens.
  - Index 0 is special: it covers the entire prompt forward pass, so its
    inner tensors have seq_len = prompt_len.
  - Indices 1..N each cover ONE generated token, with seq_len = 1.
  - Layer 0 in the inner tuple is the embedding output; layer N is the last
    transformer block's output. We use 1-indexed layers in our config
    (layer 1 = first transformer block).

  To get a (generated_seq_len, hidden_dim) array for a given layer, we slice
  step indices 1..N at that layer and concatenate along the seq_len axis.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from probe_extraction.models.base import GenerationOutput, LLM

logger = logging.getLogger(__name__)


# ============================================================================
# Implementation
# ============================================================================

class HuggingFaceLLM(LLM):
    """LLM wrapper backed by HuggingFace transformers.

    Args:
        model_name: HF repo ID (e.g., "Qwen/Qwen2.5-7B-Instruct").
        dtype: Computation dtype. "bfloat16" recommended for modern GPUs;
            "float16" if bf16 is unsupported (rare on Volta+ but possible).
        quantization: "4bit", "8bit", or "none". 4bit uses bitsandbytes NF4.
        device_map: Passed to from_pretrained. "auto" lets accelerate place
            layers; explicit dict for manual placement.
        trust_remote_code: Required True for some models with custom code.
            Qwen2.5 doesn't need it; Qwen-VL does.
        hf_token: Optional HuggingFace token for gated models.

    Example:
        llm = HuggingFaceLLM(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            dtype="bfloat16",
            quantization="4bit",
        )
        out = llm.generate_with_activations(
            prompt="Extract the title from: ...",
            layers=[12, 16, 20, 24],
            max_new_tokens=200,
        )
        print(out.text)
        print(out.hidden_states[20].shape)  # (generated_len, 3584)
    """

    # ------------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------------

    def __init__(
        self,
        model_name: str,
        *,
        dtype: str = "bfloat16",
        quantization: str = "4bit",
        device_map: str | dict[str, Any] = "auto",
        trust_remote_code: bool = False,
        hf_token: str | None = None,
    ) -> None:
        self._model_name = model_name

        torch_dtype = self._resolve_dtype(dtype)
        quant_config = self._build_quant_config(quantization, torch_dtype)

        logger.info(
            "Loading %s (dtype=%s, quantization=%s)...",
            model_name, dtype, quantization,
        )

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            token=hf_token,
        )

        # Pad token: many causal LMs don't define one. We need it for
        # batched ops (even though we don't currently batch — defensive).
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            quantization_config=quant_config,
            trust_remote_code=trust_remote_code,
            token=hf_token,
        )
        self.model.eval()  # No training; ensures dropout off, etc.

        # Cache config-derived values so we don't re-fetch every call.
        self._num_layers: int = self.model.config.num_hidden_layers
        self._hidden_dim: int = self.model.config.hidden_size

        logger.info(
            "Model loaded: %d layers, hidden_dim=%d, device=%s",
            self._num_layers, self._hidden_dim,
            next(self.model.parameters()).device,
        )

    @staticmethod
    def _resolve_dtype(dtype: str) -> torch.dtype:
        mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype not in mapping:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return mapping[dtype]

    @staticmethod
    def _build_quant_config(
        quantization: str, compute_dtype: torch.dtype,
    ) -> BitsAndBytesConfig | None:
        if quantization == "none":
            return None
        if quantization == "4bit":
            # NF4 is the default and best-tested 4-bit variant.
            # double_quant=True saves a bit more memory at negligible quality cost.
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        if quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        raise ValueError(f"Unsupported quantization: {quantization}")

    # ------------------------------------------------------------------------
    # LLM interface: properties
    # ------------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    # ------------------------------------------------------------------------
    # LLM interface: tokenization helpers
    # ------------------------------------------------------------------------

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def decode_per_token(self, token_ids: list[int]) -> list[str]:
        # `convert_ids_to_tokens` returns the BPE token strings (with
        # leading-space markers). `decode([id])` returns the user-visible
        # surface form. We want surface forms aligned 1-to-1 with positions,
        # so we decode each token individually.
        return [self.tokenizer.decode([tid]) for tid in token_ids]

    def format_chat(
        self,
        system_message: str | None,
        user_message: str,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # appends the assistant header so the
                                         # model knows to start generating
        )

    # ------------------------------------------------------------------------
    # LLM interface: generation
    # ------------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        return_logprobs: bool = False,
    ) -> GenerationOutput:
        return self._generate_inner(
            prompt=prompt,
            layers=None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            return_logprobs=return_logprobs,
        )

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
        # Validate layer indices BEFORE the expensive forward pass.
        for ℓ in layers:
            if not (1 <= ℓ <= self._num_layers):
                raise ValueError(
                    f"Layer {ℓ} out of range [1, {self._num_layers}] for "
                    f"model {self._model_name}"
                )

        return self._generate_inner(
            prompt=prompt,
            layers=layers,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            return_logprobs=return_logprobs,
        )

    # ------------------------------------------------------------------------
    # Internal: shared generation path
    # ------------------------------------------------------------------------

    @torch.no_grad()
    def _generate_inner(
        self,
        *,
        prompt: str,
        layers: list[int] | None,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        return_logprobs: bool,
    ) -> GenerationOutput:
        """Core generation routine used by both public methods.

        Capture is conditioned on `layers`: if None, hidden states are not
        requested (faster, less memory). If a list, hidden states are
        requested for those specific layers.
        """
        # ------ Tokenize prompt ------
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded.input_ids.to(self.model.device)
        attention_mask = encoded.attention_mask.to(self.model.device)
        prompt_len = input_ids.shape[1]

        # ------ Generation kwargs ------
        do_sample = temperature > 0.0
        gen_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "return_dict_in_generate": True,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        # Hidden states / scores are tied: we need scores for logprobs,
        # hidden states for activations. Both add cost so we only enable
        # them when asked.
        capture_activations = layers is not None
        if capture_activations:
            gen_kwargs["output_hidden_states"] = True
        if return_logprobs:
            gen_kwargs["output_scores"] = True

        # ------ Generate ------
        outputs = self.model.generate(**gen_kwargs)

        # `outputs.sequences` shape: (batch=1, prompt_len + generated_len)
        full_sequence = outputs.sequences[0].tolist()
        generated_ids = full_sequence[prompt_len:]
        prompt_ids = full_sequence[:prompt_len]

        # ------ Determine finish reason ------
        finish_reason = "length"
        if generated_ids and generated_ids[-1] == self.tokenizer.eos_token_id:
            finish_reason = "stop"

        # ------ Decode generated text ------
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # ------ Token logprobs (optional) ------
        token_logprobs: list[float] | None = None
        if return_logprobs and outputs.scores is not None:
            token_logprobs = self._extract_token_logprobs(
                scores=outputs.scores,
                generated_ids=generated_ids,
            )

        # ------ Hidden states (optional) ------
        hidden_states: dict[int, np.ndarray] | None = None
        if capture_activations and outputs.hidden_states is not None:
            hidden_states = self._extract_hidden_states(
                hidden_states_tuple=outputs.hidden_states,
                layers=layers,  # type: ignore[arg-type]
                num_generated=len(generated_ids),
            )

        return GenerationOutput(
            text=text,
            prompt_token_ids=prompt_ids,
            generated_token_ids=generated_ids,
            token_logprobs=token_logprobs,
            hidden_states=hidden_states,
            finish_reason=finish_reason,
            metadata={"prompt_len": prompt_len},
        )

    # ------------------------------------------------------------------------
    # Internal: extracting per-token logprobs
    # ------------------------------------------------------------------------

    @staticmethod
    def _extract_token_logprobs(
        scores: tuple[torch.Tensor, ...],
        generated_ids: list[int],
    ) -> list[float]:
        """Compute log-probabilities of the actually-chosen tokens.

        `scores` is a tuple of length num_generated_tokens; each element is a
        (batch, vocab_size) tensor of logits at that generation step.
        The chosen token at step t is generated_ids[t]; its logprob is
        log_softmax(scores[t])[chosen_id].
        """
        logprobs: list[float] = []
        for t, step_logits in enumerate(scores):
            # step_logits shape: (1, vocab)
            log_probs = torch.log_softmax(step_logits[0], dim=-1)
            chosen = generated_ids[t]
            logprobs.append(log_probs[chosen].item())
        return logprobs

    # ------------------------------------------------------------------------
    # Internal: extracting hidden states (the tricky part)
    # ------------------------------------------------------------------------

    def _extract_hidden_states(
        self,
        hidden_states_tuple: tuple[tuple[torch.Tensor, ...], ...],
        layers: list[int],
        num_generated: int,
    ) -> dict[int, np.ndarray]:
        """Slice hidden states corresponding to GENERATED tokens only.

        See module docstring for the structure. Briefly:
          - hidden_states_tuple has length 1 + num_generated.
          - We skip index 0 (the prompt forward pass) — we only care about
            activations at positions where the model produced output tokens.
          - For each subsequent step (length-1 sequences), we grab the
            requested layers and stack into a (num_generated, hidden_dim)
            array per layer.

        Returns:
            {layer_index: array of shape (num_generated, hidden_dim)}.
            Stored as float16 numpy by default; if the source dtype is
            bfloat16 we cast through float32 first because numpy doesn't
            natively support bfloat16.
        """
        if len(hidden_states_tuple) != 1 + num_generated:
            logger.warning(
                "Unexpected hidden_states length: got %d, expected %d. "
                "Truncating to min.",
                len(hidden_states_tuple), 1 + num_generated,
            )

        # Skip step 0 (prompt forward pass). Take only generated steps.
        # Defensive truncation in case the lengths disagree.
        gen_steps = hidden_states_tuple[1 : 1 + num_generated]

        result: dict[int, np.ndarray] = {}
        for ℓ in layers:
            # For each generation step, grab the layer-ℓ tensor.
            # Each is shape (1, 1, hidden_dim) — batch=1, seq=1.
            per_step_vectors: list[np.ndarray] = []
            for step_layers in gen_steps:
                t = step_layers[ℓ]                # (1, 1, hidden_dim)
                vec = t[0, 0].to(torch.float32).cpu().numpy()  # (hidden_dim,)
                per_step_vectors.append(vec)

            stacked = np.stack(per_step_vectors, axis=0)        # (gen_len, hidden_dim)
            result[ℓ] = stacked.astype(np.float16)              # storage compaction
        return result
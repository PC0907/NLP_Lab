"""Configuration loading and validation.

Loads YAML configs into validated Pydantic models. Any typo or wrong type in
a config file is caught here at load time rather than crashing mid-pipeline.

Usage:
    from probe_extraction.config import load_config
    cfg = load_config("configs/default.yaml")
    print(cfg.model.name)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Sub-config models
# ============================================================================

class ExperimentConfig(BaseModel):
    name: str
    seed: int = 42
    artifacts_dir: str = "artifacts"


class ModelConfig(BaseModel):
    name: str
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    quantization: Literal["4bit", "8bit", "none"] = "4bit"
    device_map: str = "auto"
    trust_remote_code: bool = False
    max_new_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0


class ActivationsConfig(BaseModel):
    layers: list[int]
    position: Literal["last_token", "mean", "all"] = "last_token"
    dtype: Literal["float16", "float32"] = "float16"

    @field_validator("layers")
    @classmethod
    def layers_must_be_nonempty_and_sorted(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("activations.layers must contain at least one layer")
        if any(layer < 0 for layer in v):
            raise ValueError("activations.layers must be non-negative")
        return sorted(set(v))  # dedupe and sort for consistency


class DataConfig(BaseModel):
    benchmark: Literal["extract_bench"] = "extract_bench"
    benchmark_path: str
    domains: list[str] = Field(default_factory=list)
    max_documents: int | None = None
    pdf_extractor: Literal["pymupdf"] = "pymupdf"


class ExtractionConfig(BaseModel):
    prompt_template: str = "default"
    include_schema: bool = True
    include_examples: bool = False


class LabelingConfig(BaseModel):
    strategy: Literal["strict", "semantic"] = "strict"
    string_match: Literal["exact", "fuzzy"] = "fuzzy"
    fuzzy_threshold: float = 0.85
    number_tolerance: float = 0.01
    url_normalize: bool = True
    email_normalize: bool = True

    @field_validator("fuzzy_threshold", "number_tolerance")
    @classmethod
    def must_be_in_unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"value must be in [0, 1], got {v}")
        return v


class ProbeConfig(BaseModel):
    type: Literal["logistic_regression"] = "logistic_regression"
    C: float = 1.0
    max_iter: int = 1000
    class_weight: str | None = "balanced"
    test_size: float = 0.2
    cv_folds: int = 5


class BaselinesConfig(BaseModel):
    enabled: list[str] = Field(default_factory=list)
    self_consistency_samples: int = 5
    self_consistency_temperature: float = 0.7


class SelectiveRegenConfig(BaseModel):
    thresholds: list[float]
    regeneration_strategy: Literal["resample", "focused_prompt", "bigger_model"] = "resample"
    resample_temperature: float = 0.7
    resample_samples: int = 3


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_to_file: bool = True
    log_dir: str = "logs"


# ============================================================================
# Top-level config
# ============================================================================

class Config(BaseModel):
    """Top-level project configuration.

    Loaded from YAML. All sub-configs are validated on load: type errors,
    out-of-range values, and missing required fields fail immediately.
    """

    experiment: ExperimentConfig
    model: ModelConfig
    activations: ActivationsConfig
    data: DataConfig
    extraction: ExtractionConfig
    labeling: LabelingConfig
    probe: ProbeConfig
    baselines: BaselinesConfig
    selective_regen: SelectiveRegenConfig
    logging: LoggingConfig

    # ------------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------------

    @property
    def artifacts_path(self) -> Path:
        """Resolved Path object for the artifacts directory."""
        return Path(self.experiment.artifacts_dir)

    @property
    def benchmark_path(self) -> Path:
        """Resolved Path object for the benchmark dataset."""
        return Path(self.data.benchmark_path)


# ============================================================================
# Loaders
# ============================================================================

def load_config(path: str | Path) -> Config:
    """Load a YAML config file into a validated Config object.

    Also loads environment variables from .env (if present) so secrets like
    HF_TOKEN are available to downstream code via os.environ.

    Args:
        path: Path to the YAML config file.

    Returns:
        Validated Config instance.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        pydantic.ValidationError: If the config has invalid types or values.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    # Load .env into os.environ if present. Non-fatal if missing.
    load_dotenv(override=False)

    with path.open("r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping at the root")

    return Config.model_validate(raw)


def get_hf_token() -> str | None:
    """Get HuggingFace token from Kaggle Secrets, then env, then None.

    Tries sources in order:
      1. Kaggle UserSecretsClient (when running in Kaggle)
      2. HF_TOKEN environment variable (set via .env or shell)

    Returns None if no token found; downstream code should handle that
    appropriately (e.g., proceed for ungated models, fail for gated ones).
    """
    # Try Kaggle Secrets first (only available in Kaggle environment)
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]
        try:
            return UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            # Secret not set in Kaggle; fall through to env var
            pass
    except ImportError:
        # Not running on Kaggle
        pass

    return os.environ.get("HF_TOKEN")
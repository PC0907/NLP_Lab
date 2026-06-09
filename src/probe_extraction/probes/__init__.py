"""Probe implementations for trust-signal estimation from LLM activations."""

from probe_extraction.probes.linear import LinearProbe, ProbeMetrics, train_probe
from probe_extraction.probes.clap import (
    ClapMetrics,
    ClapProbeResult,
    train_clap,
    fit_clap_for_lodo,
    predict_clap_proba,
)

__all__ = [
    # Linear probe (logistic regression, per layer) — Stage 3
    "LinearProbe",
    "ProbeMetrics",
    "train_probe",
    # CLAP probe (cross-layer attention) — Stage 6
    "ClapMetrics",
    "ClapProbeResult",
    "train_clap",
    "fit_clap_for_lodo",
    "predict_clap_proba",
]
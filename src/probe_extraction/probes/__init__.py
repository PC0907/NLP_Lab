"""Probe implementations for trust-signal estimation from LLM activations."""

from probe_extraction.probes.linear import LinearProbe, ProbeMetrics, train_probe

__all__ = [
    # Linear probe (logistic regression, per layer) — Stage 3
    "LinearProbe",
    "ProbeMetrics",
    "train_probe",
]

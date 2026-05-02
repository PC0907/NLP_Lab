"""Benchmark loaders.

A Benchmark exposes documents, schemas, and gold annotations in a uniform
shape so the rest of the pipeline doesn't care which benchmark is loaded.
"""

from probe_extraction.data.base import Benchmark, Document

__all__ = ["Benchmark", "Document"]
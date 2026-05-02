"""Smoke tests for the ExtractBench loader.

These tests exercise the loader against real ExtractBench data. They're not
unit tests in the strict sense (they hit the filesystem, parse real PDFs);
they're integration sanity checks. Run them once after data is in place to
verify the data layer works before building stages on top of it.

Run from the repo root:
    pytest tests/test_extract_bench.py -v

Skip if ExtractBench isn't checked out:
    All tests are skipped automatically when data/extract-bench is missing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from probe_extraction.data.extract_bench import ExtractBench

# ============================================================================
# Test fixtures
# ============================================================================

# Allow override via env var for non-default checkout locations
BENCHMARK_PATH = Path(os.environ.get("EXTRACT_BENCH_PATH", "data/extract-bench"))

pytestmark = pytest.mark.skipif(
    not (BENCHMARK_PATH / "dataset").exists(),
    reason=f"ExtractBench not found at {BENCHMARK_PATH}. "
           f"Clone it or set EXTRACT_BENCH_PATH.",
)


@pytest.fixture(scope="module")
def loader_sandbox() -> ExtractBench:
    """Loader scoped to the sandbox domain (academic/research).

    We use academic/research as the sandbox because:
      - All 6 PDFs have real text layers (extracts cleanly)
      - hiring/resume has 5/7 image-only PDFs that fail PyMuPDF extraction
      - swimming/sport has tabular content that's harder to validate
    """
    return ExtractBench(BENCHMARK_PATH, domains=["academic/research"])


@pytest.fixture(scope="module")
def loader_all() -> ExtractBench:
    """Loader for all available domains."""
    return ExtractBench(BENCHMARK_PATH)


# ============================================================================
# Discovery & metadata
# ============================================================================

class TestDiscovery:
    def test_loader_constructs(self, loader_sandbox: ExtractBench) -> None:
        assert loader_sandbox is not None

    def test_sandbox_domain_found(self, loader_sandbox: ExtractBench) -> None:
        assert loader_sandbox.domains == ["academic/research"]

    def test_sandbox_has_expected_document_count(
        self, loader_sandbox: ExtractBench
    ) -> None:
        # Per the dataset README: 6 academic research documents.
        # (4 are listed in the directory tree we have, but the README claims 6;
        # we assert what's actually present.)
        assert len(loader_sandbox) >= 1
        assert len(loader_sandbox) <= 10  # sanity bound

    def test_all_domains_discovered(self, loader_all: ExtractBench) -> None:
        # Per the dataset README: 5 schemas, 35 documents.
        expected_domains = {
            "academic/research",
            "finance/10kq",
            "finance/credit_agreement",
            "hiring/resume",
            "sport/swimming",
        }
        assert set(loader_all.domains) == expected_domains
        assert len(loader_all) == 35

    def test_unknown_domain_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown domains"):
            ExtractBench(BENCHMARK_PATH, domains=["nonexistent/domain"])

    def test_missing_benchmark_path_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="dataset directory not found"):
            ExtractBench("/nonexistent/path/to/extract-bench")


# ============================================================================
# Schema loading
# ============================================================================

class TestSchema:
    def test_schema_loaded(self, loader_sandbox: ExtractBench) -> None:
        """Schema should be a valid JSON Schema (regardless of wrapper format)."""
        schema = loader_sandbox.get_schema("academic/research")
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
        # academic/research uses the unwrapped format — confirm we handle both
        assert "schema_definition" not in schema

    def test_schema_has_expected_research_fields(
        self, loader_sandbox: ExtractBench
    ) -> None:
        schema = loader_sandbox.get_schema("academic/research")
        properties = schema["properties"]
        # Research papers should have at least these conceptual fields.
        # We check for *some* of them existing rather than requiring all,
        # since the exact field set may vary.
        expected_any_of = {"title", "authors", "abstract", "ids"}
        assert expected_any_of & set(properties.keys()), (
            f"Expected at least one of {expected_any_of} in schema properties; "
            f"got {list(properties.keys())}"
        )

    def test_unknown_domain_schema_raises(
        self, loader_sandbox: ExtractBench
    ) -> None:
        with pytest.raises(KeyError):
            loader_sandbox.get_schema("finance/10kq")  # not loaded


# ============================================================================
# Document iteration
# ============================================================================

class TestDocuments:
    def test_iteration_yields_documents(
        self, loader_sandbox: ExtractBench
    ) -> None:
        docs = list(loader_sandbox)
        assert len(docs) >= 1
        for doc in docs:
            assert doc.doc_id
            assert doc.domain == "academic/research"
            assert doc.schema is not None
            assert doc.gold is not None

    def test_doc_ids_are_unique(self, loader_sandbox: ExtractBench) -> None:
        ids = [doc.doc_id for doc in loader_sandbox]
        assert len(ids) == len(set(ids))

    def test_doc_ids_are_filesystem_safe(
        self, loader_sandbox: ExtractBench
    ) -> None:
        bad_chars = set('/\\:*?"<>| ')
        for doc in loader_sandbox:
            assert not (set(doc.doc_id) & bad_chars), (
                f"doc_id contains unsafe characters: {doc.doc_id!r}"
            )

    def test_doc_ids_have_expected_prefix(
        self, loader_sandbox: ExtractBench
    ) -> None:
        for doc in loader_sandbox:
            assert doc.doc_id.startswith("academic__research__")

    def test_max_documents_caps_iteration(self) -> None:
        loader = ExtractBench(
            BENCHMARK_PATH, domains=["academic/research"], max_documents=2,
        )
        assert len(loader) == 2
        assert len(list(loader)) == 2

    def test_pdf_extraction_succeeds_for_research_papers(
        self, loader_sandbox: ExtractBench
    ) -> None:
        """Academic research PDFs are real digital documents and should
        extract cleanly with PyMuPDF.
        """
        docs = list(loader_sandbox)
        succeeded = [d for d in docs if d.extraction_error is None]
        failed = [d for d in docs if d.extraction_error is not None]

        # We expect ALL research PDFs to extract cleanly. If even one fails,
        # we want to know about it — likely indicates a problem.
        assert len(succeeded) == len(docs), (
            f"Unexpected extraction failures in academic/research: "
            f"{[(d.doc_id, d.extraction_error) for d in failed]}"
        )

        # Research papers should yield substantial text (more than a resume).
        for doc in succeeded:
            assert len(doc.text) >= 1000, (
                f"Document {doc.doc_id} has only {len(doc.text)} chars; "
                f"research papers should be longer."
            )


# ============================================================================
# Metadata
# ============================================================================

class TestMetadata:
    def test_metadata_populated(self, loader_sandbox: ExtractBench) -> None:
        for doc in loader_sandbox:
            assert "page_count" in doc.metadata
            assert doc.metadata["page_count"] >= 1
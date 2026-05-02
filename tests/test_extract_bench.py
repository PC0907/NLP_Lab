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
def loader_resume() -> ExtractBench:
    """Loader scoped to just the resume domain (fastest)."""
    return ExtractBench(BENCHMARK_PATH, domains=["hiring/resume"])


@pytest.fixture(scope="module")
def loader_all() -> ExtractBench:
    """Loader for all available domains."""
    return ExtractBench(BENCHMARK_PATH)


# ============================================================================
# Discovery & metadata
# ============================================================================

class TestDiscovery:
    def test_loader_constructs(self, loader_resume: ExtractBench) -> None:
        assert loader_resume is not None

    def test_resume_domain_found(self, loader_resume: ExtractBench) -> None:
        assert loader_resume.domains == ["hiring/resume"]

    def test_resume_has_seven_documents(self, loader_resume: ExtractBench) -> None:
        # Per the dataset README: 7 resume documents.
        assert len(loader_resume) == 7

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
    def test_schema_unwrapped(self, loader_resume: ExtractBench) -> None:
        """Schema should be the inner schema_definition, not the outer wrapper."""
        schema = loader_resume.get_schema("hiring/resume")
        # The unwrapped schema has 'type'/'properties' at the top, not 'name'.
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "name" not in schema  # outer wrapper key
        assert "schema_definition" not in schema  # outer wrapper key

    def test_schema_has_expected_resume_fields(
        self, loader_resume: ExtractBench
    ) -> None:
        schema = loader_resume.get_schema("hiring/resume")
        properties = schema["properties"]
        for required_field in [
            "personalInfo",
            "workExperience",
            "education",
            "skills",
        ]:
            assert required_field in properties

    def test_unknown_domain_schema_raises(
        self, loader_resume: ExtractBench
    ) -> None:
        with pytest.raises(KeyError):
            loader_resume.get_schema("finance/10kq")  # not loaded


# ============================================================================
# Document iteration
# ============================================================================

class TestDocuments:
    def test_iteration_yields_documents(
        self, loader_resume: ExtractBench
    ) -> None:
        docs = list(loader_resume)
        assert len(docs) == 7
        for doc in docs:
            assert doc.doc_id
            assert doc.domain == "hiring/resume"
            assert doc.schema is not None
            assert doc.gold is not None

    def test_doc_ids_are_unique(self, loader_resume: ExtractBench) -> None:
        ids = [doc.doc_id for doc in loader_resume]
        assert len(ids) == len(set(ids))

    def test_doc_ids_are_filesystem_safe(
        self, loader_resume: ExtractBench
    ) -> None:
        bad_chars = set('/\\:*?"<>| ')
        for doc in loader_resume:
            assert not (set(doc.doc_id) & bad_chars), (
                f"doc_id contains unsafe characters: {doc.doc_id!r}"
            )

    def test_doc_ids_have_expected_prefix(
        self, loader_resume: ExtractBench
    ) -> None:
        for doc in loader_resume:
            assert doc.doc_id.startswith("hiring__resume__")

    def test_max_documents_caps_iteration(self) -> None:
        loader = ExtractBench(
            BENCHMARK_PATH, domains=["hiring/resume"], max_documents=3,
        )
        assert len(loader) == 3
        assert len(list(loader)) == 3

    def test_pdf_extraction_mostly_succeeds(
        self, loader_resume: ExtractBench
    ) -> None:
        """Most resumes should extract cleanly. Allow a small number of
        image-only PDFs (known issue: some ExtractBench resumes are
        image-PDFs from Google Docs without embedded text).
        """
        docs = list(loader_resume)
        succeeded = [d for d in docs if d.extraction_error is None]
        failed = [d for d in docs if d.extraction_error is not None]

        # We expect at least most resumes to extract successfully.
        # If more than half fail, something is broken in the extractor.
        assert len(succeeded) >= len(docs) // 2 + 1, (
            f"Too many extraction failures: {len(failed)}/{len(docs)}. "
            f"Failures: {[(d.doc_id, d.extraction_error) for d in failed]}"
        )

        # All successful extractions should have non-trivial text.
        for doc in succeeded:
            assert len(doc.text) >= 50, (
                f"Document {doc.doc_id} reported success but has only "
                f"{len(doc.text)} characters of text"
            )

        # All failures should have an error message (no silent failures).
        for doc in failed:
            assert doc.extraction_error, (
                f"Document {doc.doc_id} has empty text but no error message"
            )

    def test_known_image_only_pdf_is_flagged(
        self, loader_resume: ExtractBench
    ) -> None:
        """Resume-Finance is known to be an image-only PDF from Google Docs.
        Verify our extractor correctly flags it rather than silently producing
        empty content.
        """
        target_id = "hiring__resume__Resume-Finance"
        doc = next((d for d in loader_resume if d.doc_id == target_id), None)
        if doc is None:
            pytest.skip(f"{target_id} not present in this checkout")

        assert doc.extraction_error is not None
        assert doc.text == ""
        assert "image" in doc.extraction_error.lower() or \
               "0 chars" in doc.extraction_error.lower()

    def test_gold_loaded_correctly_for_resume_it(
        self, loader_resume: ExtractBench
    ) -> None:
        """Sanity-check Resume-IT gold matches what we expect from the sample."""
        target_id = "hiring__resume__Resume-IT"
        doc = next(
            (d for d in loader_resume if d.doc_id == target_id), None
        )
        assert doc is not None, f"Resume-IT not found among loaded documents"

        # Spot-check known fields from the gold sample we reviewed.
        gold = doc.gold
        assert gold["personalInfo"]["fullName"] == "Marcus Chen"
        assert isinstance(gold["workExperience"], list)
        assert len(gold["workExperience"]) == 3
        assert gold["workExperience"][0]["employer"] == "NexaTech Solutions"
        assert gold["workExperience"][0]["isCurrent"] is True


# ============================================================================
# Metadata
# ============================================================================

class TestMetadata:
    def test_metadata_populated(self, loader_resume: ExtractBench) -> None:
        for doc in loader_resume:
            assert "page_count" in doc.metadata
            assert doc.metadata["page_count"] >= 1
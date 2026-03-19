"""
Tests for agentx.pipeline module.
Covers: PIIDetector, DataValidator, DataCleaner, RuntimePIIMasker,
        KnowledgeFreshnessManager, RetentionPolicy, FineTunePipeline,
        CitationManager, SourceReliability.
"""

import time

import pytest

from agentx.pipeline.ingestion import PIIDetector, DataValidator, DataCleaner
from agentx.pipeline.knowledge import (
    RuntimePIIMasker,
    KnowledgeFreshnessManager, KnowledgeSource,
    RetentionPolicy,
    FineTunePipeline, FineTuneConfig,
    CitationManager, SourceReliability,
)


# ─────────────────────────────────────────────
# PIIDetector
# ─────────────────────────────────────────────

class TestPIIDetector:
    def setup_method(self):
        self.detector = PIIDetector()

    def test_detect_email(self):
        findings = self.detector.detect("Contact me at john@example.com please")
        types = [f["type"] for f in findings]
        assert "email" in types

    def test_detect_phone(self):
        findings = self.detector.detect("Call me at (555) 123-4567")
        types = [f["type"] for f in findings]
        assert "phone" in types

    def test_detect_ssn(self):
        findings = self.detector.detect("SSN: 123-45-6789")
        types = [f["type"] for f in findings]
        assert "ssn" in types

    def test_detect_credit_card(self):
        findings = self.detector.detect("Card: 4111 1111 1111 1111")
        types = [f["type"] for f in findings]
        assert "credit_card" in types

    def test_detect_aadhaar(self):
        findings = self.detector.detect("Aadhaar: 1234 5678 9012")
        types = [f["type"] for f in findings]
        assert "aadhaar" in types

    def test_detect_pan(self):
        findings = self.detector.detect("PAN: ABCDE1234F")
        types = [f["type"] for f in findings]
        assert "pan" in types

    def test_redact_mask(self):
        detector = PIIDetector(method="mask")
        result = detector.redact("Email: john@example.com")
        assert "[EMAIL_REDACTED]" in result
        assert "john@example.com" not in result

    def test_redact_hash(self):
        detector = PIIDetector(method="hash")
        result = detector.redact("Email: john@example.com")
        assert "john@example.com" not in result

    def test_redact_remove(self):
        detector = PIIDetector(method="remove")
        result = detector.redact("Email: john@example.com")
        assert "john@example.com" not in result

    def test_has_pii_true(self):
        assert self.detector.has_pii("my email is test@test.com") is True

    def test_has_pii_false(self):
        assert self.detector.has_pii("no personal info here") is False

    def test_specific_fields(self):
        detector = PIIDetector(fields_to_detect=["email"])
        # Should detect email
        assert detector.has_pii("test@test.com") is True
        # Should NOT detect phone (not in fields)
        findings = detector.detect("Call (555) 123-4567")
        types = [f["type"] for f in findings]
        assert "phone" not in types


# ─────────────────────────────────────────────
# DataValidator
# ─────────────────────────────────────────────

class TestDataValidator:
    def test_valid_content(self):
        validator = DataValidator(min_length=5, max_length=1000)
        is_valid, errors = validator.validate("This is valid content that is long enough.")
        assert is_valid is True
        assert errors == []

    def test_too_short(self):
        validator = DataValidator(min_length=100)
        is_valid, errors = validator.validate("short")
        assert is_valid is False
        assert any("too short" in e.lower() for e in errors)

    def test_too_long(self):
        validator = DataValidator(max_length=10)
        is_valid, errors = validator.validate("x" * 100)
        assert is_valid is False
        assert any("too long" in e.lower() for e in errors)

    def test_blocked_content(self):
        validator = DataValidator(min_length=1, blocked_content=["spam", "forbidden"])
        is_valid, errors = validator.validate("This contains spam content")
        assert is_valid is False
        assert any("blocked" in e.lower() for e in errors)

    def test_required_fields(self):
        validator = DataValidator(min_length=1, required_fields=["source", "author"])
        is_valid, errors = validator.validate("content", metadata={"source": "web"})
        assert is_valid is False
        assert any("author" in e for e in errors)

    def test_empty_content(self):
        validator = DataValidator(min_length=0)
        is_valid, errors = validator.validate("   ")
        assert is_valid is False


# ─────────────────────────────────────────────
# DataCleaner
# ─────────────────────────────────────────────

class TestDataCleaner:
    def setup_method(self):
        self.cleaner = DataCleaner()

    def test_removes_html_tags(self):
        result = self.cleaner.clean("<p>Hello <b>world</b></p>")
        assert "<p>" not in result
        assert "<b>" not in result
        assert "Hello" in result

    def test_collapses_whitespace(self):
        result = self.cleaner.clean("hello     world")
        assert "  " not in result

    def test_collapses_newlines(self):
        result = self.cleaner.clean("a\n\n\n\n\nb")
        assert "\n\n\n" not in result

    def test_removes_control_chars(self):
        result = self.cleaner.clean("hello\x00\x01world")
        assert "\x00" not in result

    def test_strips(self):
        result = self.cleaner.clean("  text  ")
        assert result == "text"


# ─────────────────────────────────────────────
# RuntimePIIMasker
# ─────────────────────────────────────────────

class TestRuntimePIIMasker:
    def test_mask_and_unmask_roundtrip(self):
        masker = RuntimePIIMasker()
        original = "My email is john@example.com and phone is (555) 123-4567"
        masked = masker.mask(original)
        assert "john@example.com" not in masked
        assert "[EMAIL_MASKED_" in masked
        unmasked = masker.unmask(masked)
        assert unmasked == original

    def test_mask_no_pii(self):
        masker = RuntimePIIMasker()
        text = "No personal info here"
        masked = masker.mask(text)
        assert masked == text
        assert masker.has_pii is False

    def test_has_pii_property(self):
        masker = RuntimePIIMasker()
        masker.mask("email: a@b.com")
        assert masker.has_pii is True

    def test_mask_contexts(self):
        masker = RuntimePIIMasker()
        contexts = ["Contact john@test.com", "No PII here"]
        masked = masker.mask_contexts(contexts)
        assert "john@test.com" not in masked[0]
        assert masked[1] == "No PII here"

    def test_specific_fields(self):
        masker = RuntimePIIMasker(fields=["email"])
        masked = masker.mask("email: a@b.com, phone: (555) 123-4567")
        assert "a@b.com" not in masked
        # Phone should still be visible since we only mask email
        assert "555" in masked


# ─────────────────────────────────────────────
# KnowledgeFreshnessManager
# ─────────────────────────────────────────────

class TestKnowledgeFreshnessManager:
    def test_register_source(self):
        mgr = KnowledgeFreshnessManager()
        source = KnowledgeSource(name="docs", source_type="file", location="./docs")
        mgr.register_source(source)
        assert source.id != ""
        assert source.id in mgr._sources

    def test_check_freshness_not_indexed(self):
        mgr = KnowledgeFreshnessManager()
        source = KnowledgeSource(name="docs", location="./docs")
        mgr.register_source(source)
        info = mgr.check_freshness(source.id)
        assert info["is_stale"] is True  # Never indexed = stale

    def test_check_freshness_not_found(self):
        mgr = KnowledgeFreshnessManager()
        info = mgr.check_freshness("nonexistent")
        assert "error" in info

    def test_get_stale_sources_all_stale(self):
        mgr = KnowledgeFreshnessManager()
        source = KnowledgeSource(
            name="docs", location="./docs",
            refresh_interval_hours=0.001,  # very short
        )
        mgr.register_source(source)
        stale = mgr.get_stale_sources()
        assert len(stale) == 1

    def test_get_stale_sources_fresh(self):
        mgr = KnowledgeFreshnessManager()
        source = KnowledgeSource(
            name="docs", location="./docs",
            last_indexed=time.time(),
            refresh_interval_hours=24,
        )
        mgr.register_source(source)
        stale = mgr.get_stale_sources()
        assert len(stale) == 0

    def test_mark_refreshed(self):
        mgr = KnowledgeFreshnessManager()
        source = KnowledgeSource(name="docs", location="./docs")
        mgr.register_source(source)
        mgr.mark_refreshed(source.id, new_hash="abc123", doc_count=50)
        updated = mgr._sources[source.id]
        assert updated.content_hash == "abc123"
        assert updated.document_count == 50
        assert updated.is_stale is False
        assert updated.version == 2  # Incremented


# ─────────────────────────────────────────────
# RetentionPolicy
# ─────────────────────────────────────────────

class TestRetentionPolicy:
    def test_defaults(self):
        policy = RetentionPolicy()
        assert policy.conversations_days == 90
        assert policy.embeddings_days == 365
        assert policy.user_data_days == 730
        assert policy.audit_logs_days == 365
        assert policy.allow_deletion is True
        assert policy.archive_before_delete is True

    def test_custom(self):
        policy = RetentionPolicy(conversations_days=30, allow_deletion=False)
        assert policy.conversations_days == 30
        assert policy.allow_deletion is False


# ─────────────────────────────────────────────
# FineTunePipeline
# ─────────────────────────────────────────────

class TestFineTunePipeline:
    def test_collect(self):
        pipeline = FineTunePipeline()
        pipeline.collect(query="What is Python?", response="A programming language", quality_score=0.9)
        pipeline.collect(query="What is JS?", response="A scripting language", quality_score=0.5)
        assert len(pipeline._samples) == 2

    def test_curate_quality_filter(self):
        pipeline = FineTunePipeline(config=FineTuneConfig(quality_threshold=0.8))
        pipeline.collect(query="q1", response="r1", quality_score=0.9)
        pipeline.collect(query="q2", response="r2", quality_score=0.5)
        pipeline.collect(query="q3", response="r3", quality_score=0.85)
        stats = pipeline.curate()
        assert stats["total_collected"] == 3
        assert stats["after_quality_filter"] == 2
        assert stats["after_dedup"] == 2

    def test_curate_dedup(self):
        pipeline = FineTunePipeline(config=FineTuneConfig(quality_threshold=0.0))
        pipeline.collect(query="same q", response="same r", quality_score=0.9)
        pipeline.collect(query="same q", response="same r", quality_score=0.9)
        stats = pipeline.curate()
        assert stats["after_dedup"] == 1

    def test_stats(self):
        pipeline = FineTunePipeline()
        pipeline.collect(query="q", response="r", quality_score=0.95)
        stats = pipeline.stats()
        assert stats["total_collected"] == 1
        assert stats["quality_distribution"]["excellent"] == 1

    def test_export_jsonl(self, tmp_path):
        pipeline = FineTunePipeline(config=FineTuneConfig(output_dir=str(tmp_path)))
        pipeline.collect(query="q", response="r", quality_score=0.9)
        path = pipeline.export(format="jsonl")
        assert path.endswith(".jsonl")
        import json
        with open(path) as f:
            data = json.loads(f.readline())
        assert "messages" in data


# ─────────────────────────────────────────────
# CitationManager
# ─────────────────────────────────────────────

class TestCitationManager:
    def test_register_source(self):
        cm = CitationManager()
        cm.register_source("react_docs", name="React Docs", authority=0.95, recency=0.8, accuracy=0.9)
        assert "react_docs" in cm._sources

    def test_get_reliability(self):
        cm = CitationManager()
        cm.register_source("src1", authority=1.0, recency=1.0, accuracy=1.0)
        score = cm.get_reliability("src1")
        assert score == pytest.approx(1.0, abs=0.01)

    def test_get_reliability_unknown(self):
        cm = CitationManager()
        assert cm.get_reliability("unknown") == 0.5

    def test_weight_contexts(self):
        cm = CitationManager()
        cm.register_source("reliable", authority=1.0, recency=1.0, accuracy=1.0)
        cm.register_source("unreliable", authority=0.0, recency=0.0, accuracy=0.0)

        weighted = cm.weight_contexts(
            contexts=["ctx1", "ctx2"],
            source_ids=["reliable", "unreliable"],
            scores=[1.0, 1.0],
        )
        # Reliable source should be ranked higher
        assert weighted[0][0] == "ctx1"
        assert weighted[0][1] > weighted[1][1]

    def test_record_feedback_updates_accuracy(self):
        cm = CitationManager()
        cm.register_source("src1", authority=0.5, recency=0.5, accuracy=0.5)
        cm.record_feedback("src1", helpful=True)
        cm.record_feedback("src1", helpful=True)
        cm.record_feedback("src1", helpful=False)
        # Accuracy should now be 2/3
        assert cm._sources["src1"].accuracy_score == pytest.approx(2 / 3, abs=0.01)

    def test_format_citations_inline(self):
        cm = CitationManager()
        cm.register_source("s1", name="Source One")
        formatted = cm.format_citations(["s1"], format="inline")
        assert "Source One" in formatted
        assert "[1]" in formatted

    def test_format_citations_footnote(self):
        cm = CitationManager()
        cm.register_source("s1", name="Source One", source_type="official_docs")
        formatted = cm.format_citations(["s1"], format="footnote")
        assert "Sources:" in formatted
        assert "Source One" in formatted


class TestSourceReliability:
    def test_overall_score_formula(self):
        # overall = authority*0.4 + recency*0.3 + accuracy*0.3
        cm = CitationManager()
        cm.register_source("src", authority=1.0, recency=0.5, accuracy=0.5)
        src = cm._sources["src"]
        expected = 1.0 * 0.4 + 0.5 * 0.3 + 0.5 * 0.3
        assert src.overall_score == pytest.approx(expected, abs=0.01)

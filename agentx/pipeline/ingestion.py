"""
AgentX - Data Pipeline.
Phase 2: Ingestion, cleaning, validation, transformation, and PII detection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..rag.engine import Document

logger = logging.getLogger("agentx")


# --- PII Detection & Redaction ---

class PIIDetector:
    """Detect and redact Personally Identifiable Information."""

    PATTERNS: dict[str, str] = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone": r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        "ssn": r"\b\d{3}-?\d{2}-?\d{4}\b",
        "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
        "pan": r"\b[A-Z]{5}\d{4}[A-Z]\b",
        "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    }

    def __init__(self, fields_to_detect: list[str] | None = None, method: str = "mask"):
        self.fields = fields_to_detect or list(self.PATTERNS.keys())
        self.method = method

    def detect(self, text: str) -> list[dict[str, Any]]:
        """Find all PII in text."""
        findings = []
        for field_name in self.fields:
            pattern = self.PATTERNS.get(field_name)
            if not pattern:
                continue
            for match in re.finditer(pattern, text):
                findings.append({
                    "type": field_name,
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                })
        return findings

    def redact(self, text: str) -> str:
        """Remove or mask PII from text."""
        for field_name in self.fields:
            pattern = self.PATTERNS.get(field_name)
            if not pattern:
                continue
            if self.method == "mask":
                text = re.sub(pattern, f"[{field_name.upper()}_REDACTED]", text)
            elif self.method == "hash":
                def hash_replace(m: re.Match[str]) -> str:
                    return hashlib.sha256(m.group().encode()).hexdigest()[:12]
                text = re.sub(pattern, hash_replace, text)
            elif self.method == "remove":
                text = re.sub(pattern, "", text)
        return text

    def has_pii(self, text: str) -> bool:
        return len(self.detect(text)) > 0


# --- Data Validators ---

class DataValidator:
    """Validate data quality before ingestion."""

    def __init__(
        self,
        min_length: int = 50,
        max_length: int = 50000,
        required_fields: list[str] | None = None,
        blocked_content: list[str] | None = None,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.required_fields = required_fields or []
        self.blocked_content = blocked_content or []

    def validate(self, text: str, metadata: dict[str, Any] | None = None) -> tuple[bool, list[str]]:
        """Validate content. Returns (is_valid, list_of_errors)."""
        errors = []
        metadata = metadata or {}

        if len(text.strip()) < self.min_length:
            errors.append(f"Content too short: {len(text)} < {self.min_length}")

        if len(text) > self.max_length:
            errors.append(f"Content too long: {len(text)} > {self.max_length}")

        for field in self.required_fields:
            if field not in metadata:
                errors.append(f"Missing required metadata field: {field}")

        for blocked in self.blocked_content:
            if blocked.lower() in text.lower():
                errors.append(f"Blocked content found: {blocked}")

        if not text.strip():
            errors.append("Empty content")

        return len(errors) == 0, errors


# --- Data Cleaners ---

class DataCleaner:
    """Clean and normalize text data before ingestion."""

    def clean(self, text: str) -> str:
        # Remove excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\t+", " ", text)

        # Remove common artifacts
        text = re.sub(r"<[^>]+>", "", text)  # HTML tags
        text = re.sub(r"\[.*?\]\(.*?\)", lambda m: m.group(), text)  # Keep markdown links
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)  # Control chars

        return text.strip()


# --- Data Source Loaders ---

class BaseLoader(ABC):
    """Abstract data source loader."""

    @abstractmethod
    async def load(self) -> list[dict[str, Any]]:
        """Load data and return list of {content, metadata} dicts."""


class FileLoader(BaseLoader):
    """Load data from files (txt, md, json)."""

    def __init__(self, path: str, glob_pattern: str = "**/*.*"):
        self.path = Path(path)
        self.glob_pattern = glob_pattern

    async def load(self) -> list[dict[str, Any]]:
        results = []
        for file_path in self.path.glob(self.glob_pattern):
            if file_path.suffix in (".txt", ".md"):
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                results.append({
                    "content": content,
                    "metadata": {
                        "source": str(file_path),
                        "filename": file_path.name,
                        "type": file_path.suffix.lstrip("."),
                        "loaded_at": datetime.now(timezone.utc).isoformat(),
                    },
                })
            elif file_path.suffix == ".json":
                data = json.loads(file_path.read_text())
                if isinstance(data, list):
                    for item in data:
                        results.append({
                            "content": item.get("content", json.dumps(item)),
                            "metadata": {**item.get("metadata", {}), "source": str(file_path)},
                        })
                elif isinstance(data, dict):
                    results.append({
                        "content": data.get("content", json.dumps(data)),
                        "metadata": {**data.get("metadata", {}), "source": str(file_path)},
                    })
        return results


class URLLoader(BaseLoader):
    """Load data from URLs (web scraping)."""

    def __init__(self, urls: list[str]):
        self.urls = urls

    async def load(self) -> list[dict[str, Any]]:
        results = []
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                for url in self.urls:
                    try:
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                            text = await resp.text()
                            # Basic HTML stripping
                            clean = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
                            clean = re.sub(r"<style[^>]*>.*?</style>", "", clean, flags=re.DOTALL)
                            clean = re.sub(r"<[^>]+>", " ", clean)
                            clean = re.sub(r"\s+", " ", clean).strip()
                            results.append({
                                "content": clean,
                                "metadata": {"source": url, "type": "web", "loaded_at": datetime.now(timezone.utc).isoformat()},
                            })
                    except Exception as e:
                        logger.warning(f"Failed to load URL {url}: {e}")
        except ImportError:
            logger.error("Install aiohttp: pip install aiohttp")
        return results


# --- Main Pipeline ---

class IngestionPipeline:
    """
    Complete data ingestion pipeline.

    Flow: Load → Clean → Validate → Detect PII → Transform → Store

    Usage:
        pipeline = IngestionPipeline(rag_engine=rag)
        pipeline.add_loader(FileLoader("./docs"))
        stats = await pipeline.run()
    """

    def __init__(
        self,
        rag_engine: Any,
        pii_detector: PIIDetector | None = None,
        validator: DataValidator | None = None,
        cleaner: DataCleaner | None = None,
    ):
        self.rag_engine = rag_engine
        self.pii_detector = pii_detector or PIIDetector()
        self.validator = validator or DataValidator()
        self.cleaner = cleaner or DataCleaner()
        self.loaders: list[BaseLoader] = []
        self._stats: dict[str, int] = {"loaded": 0, "cleaned": 0, "validated": 0, "pii_redacted": 0, "ingested": 0, "errors": 0}

    def add_loader(self, loader: BaseLoader) -> IngestionPipeline:
        self.loaders.append(loader)
        return self

    async def run(self, detect_pii: bool = True, validate: bool = True) -> dict[str, int]:
        """Execute the full ingestion pipeline."""
        self._stats = {"loaded": 0, "cleaned": 0, "validated": 0, "pii_redacted": 0, "ingested": 0, "errors": 0}

        for loader in self.loaders:
            try:
                raw_data = await loader.load()
                self._stats["loaded"] += len(raw_data)

                for item in raw_data:
                    content = item["content"]
                    metadata = item.get("metadata", {})

                    # Step 1: Clean
                    content = self.cleaner.clean(content)
                    self._stats["cleaned"] += 1

                    # Step 2: Validate
                    if validate:
                        is_valid, errors = self.validator.validate(content, metadata)
                        if not is_valid:
                            logger.warning(f"Validation failed: {errors}")
                            self._stats["errors"] += 1
                            continue
                    self._stats["validated"] += 1

                    # Step 3: PII Detection & Redaction
                    if detect_pii:
                        if self.pii_detector.has_pii(content):
                            content = self.pii_detector.redact(content)
                            self._stats["pii_redacted"] += 1

                    # Step 4: Ingest into RAG
                    count = await self.rag_engine.ingest(content, metadata)
                    self._stats["ingested"] += count

            except Exception as e:
                logger.error(f"Pipeline error: {e}")
                self._stats["errors"] += 1

        logger.info(f"Pipeline complete: {self._stats}")
        return self._stats

    @property
    def stats(self) -> dict[str, int]:
        return self._stats

"""
AgentX - Knowledge Management Pipeline.

Addresses LLM Limitations:
1. Stale Knowledge — auto-refresh, versioning, freshness tracking
2. Data Retention — enforcement, archival, compliance
3. Fine-tuning — data collection, training pipeline, model versioning
4. PII at query time — runtime masking before sending to LLM
5. Citation & Source Reliability — source scoring, formatted citations
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Awaitable

from pydantic import BaseModel, Field

logger = logging.getLogger("agentx")


# ═══════════════════════════════════════════════════════════════
# 1. KNOWLEDGE FRESHNESS — Stale detection, auto-refresh, versioning
# ═══════════════════════════════════════════════════════════════

class KnowledgeSource(BaseModel):
    """A tracked knowledge source with freshness metadata."""

    id: str = ""
    name: str = ""
    source_type: str = "file"  # file, url, api, database
    location: str = ""         # file path, URL, etc.
    last_indexed: float = 0.0
    last_checked: float = 0.0
    content_hash: str = ""     # hash of content at last index
    version: int = 1
    document_count: int = 0
    is_stale: bool = False
    refresh_interval_hours: float = 24.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeFreshnessManager:
    """
    Tracks knowledge source freshness and triggers re-indexing when stale.

    Addresses LLM limitation: Knowledge cutoffs / stale knowledge.

    Usage:
        manager = KnowledgeFreshnessManager()

        # Register sources
        manager.register_source(KnowledgeSource(
            name="product_docs",
            source_type="file",
            location="./docs/products/",
            refresh_interval_hours=12,
        ))

        # Check freshness
        stale = manager.get_stale_sources()
        for source in stale:
            await manager.refresh_source(source.id, refresh_fn)

        # Version tracking
        manager.mark_refreshed("product_docs", new_hash="abc123", doc_count=50)
    """

    def __init__(
        self,
        stale_warning_days: int = 30,
        auto_refresh: bool = False,
        db: Any = None,
    ):
        self._sources: dict[str, KnowledgeSource] = {}
        self._refresh_history: list[dict[str, Any]] = []
        self._stale_warning_days = stale_warning_days
        self._auto_refresh = auto_refresh
        self._db = db

    def register_source(self, source: KnowledgeSource) -> None:
        """Register a knowledge source for freshness tracking."""
        if not source.id:
            source.id = hashlib.md5(f"{source.name}:{source.location}".encode()).hexdigest()[:12]
        self._sources[source.id] = source
        logger.info(f"Registered knowledge source: {source.name} ({source.source_type})")

    def get_stale_sources(self) -> list[KnowledgeSource]:
        """Get all sources that need refreshing."""
        now = time.time()
        stale = []
        for source in self._sources.values():
            age_hours = (now - source.last_indexed) / 3600 if source.last_indexed > 0 else float('inf')
            if age_hours > source.refresh_interval_hours:
                source.is_stale = True
                stale.append(source)
        return stale

    def check_freshness(self, source_id: str) -> dict[str, Any]:
        """Check freshness of a specific source."""
        source = self._sources.get(source_id)
        if not source:
            return {"error": f"Source {source_id} not found"}

        now = time.time()
        age_hours = (now - source.last_indexed) / 3600 if source.last_indexed > 0 else float('inf')
        age_days = age_hours / 24

        return {
            "source": source.name,
            "age_hours": round(age_hours, 1),
            "age_days": round(age_days, 1),
            "is_stale": age_hours > source.refresh_interval_hours,
            "refresh_needed_in_hours": max(0, source.refresh_interval_hours - age_hours),
            "version": source.version,
            "document_count": source.document_count,
            "warning": age_days > self._stale_warning_days,
        }

    async def refresh_source(
        self,
        source_id: str,
        refresh_fn: Callable[..., Awaitable[dict[str, Any]]],
    ) -> dict[str, Any]:
        """
        Refresh a knowledge source using the provided refresh function.

        The refresh_fn should:
        1. Load fresh data from the source
        2. Re-index into the vector store
        3. Return {"content_hash": "...", "document_count": N}
        """
        source = self._sources.get(source_id)
        if not source:
            return {"error": f"Source {source_id} not found"}

        try:
            result = await refresh_fn(source)
            self.mark_refreshed(
                source_id,
                new_hash=result.get("content_hash", ""),
                doc_count=result.get("document_count", 0),
            )
            self._refresh_history.append({
                "source_id": source_id,
                "source_name": source.name,
                "timestamp": time.time(),
                "version": source.version,
                "document_count": source.document_count,
                "success": True,
            })
            logger.info(f"Refreshed source: {source.name} (v{source.version})")
            return {"success": True, "version": source.version}
        except Exception as e:
            self._refresh_history.append({
                "source_id": source_id,
                "timestamp": time.time(),
                "success": False,
                "error": str(e),
            })
            logger.error(f"Failed to refresh source {source.name}: {e}")
            return {"success": False, "error": str(e)}

    def mark_refreshed(
        self, source_id: str, new_hash: str = "", doc_count: int = 0,
    ) -> None:
        """Mark a source as freshly indexed."""
        source = self._sources.get(source_id)
        if source:
            changed = new_hash != source.content_hash if new_hash else True
            if changed:
                source.version += 1
            source.content_hash = new_hash or source.content_hash
            source.last_indexed = time.time()
            source.last_checked = time.time()
            source.document_count = doc_count or source.document_count
            source.is_stale = False

    def report(self) -> dict[str, Any]:
        """Get freshness report for all sources."""
        stale = self.get_stale_sources()
        return {
            "total_sources": len(self._sources),
            "stale_sources": len(stale),
            "sources": {
                sid: self.check_freshness(sid) for sid in self._sources
            },
            "refresh_history_count": len(self._refresh_history),
        }


# ═══════════════════════════════════════════════════════════════
# 2. DATA RETENTION — Enforcement, archival, compliance
# ═══════════════════════════════════════════════════════════════

class RetentionPolicy(BaseModel):
    """Data retention policy configuration."""

    conversations_days: int = 90
    embeddings_days: int = 365
    user_data_days: int = 730
    audit_logs_days: int = 365
    cost_records_days: int = 365
    training_data_days: int = 730
    allow_deletion: bool = True       # GDPR right to be forgotten
    archive_before_delete: bool = True
    archive_path: str = "./data/archive"


class RetentionEnforcer:
    """
    Enforce data retention policies — automatic cleanup and archival.

    Addresses: GDPR compliance, data minimization, storage management.

    Usage:
        enforcer = RetentionEnforcer(policy=RetentionPolicy(), db=db)
        report = await enforcer.enforce()
        await enforcer.delete_user_data("user-123")  # GDPR right to be forgotten
    """

    def __init__(self, policy: RetentionPolicy | None = None, db: Any = None):
        self.policy = policy or RetentionPolicy()
        self._db = db
        self._archive_path = Path(self.policy.archive_path)
        self._enforcement_log: list[dict[str, Any]] = []

    async def enforce(self) -> dict[str, Any]:
        """Run retention enforcement — delete expired data."""
        if not self._db or not self._db.is_connected:
            return {"error": "Database not connected"}

        report = {"timestamp": time.time(), "actions": []}

        # Define tables and their retention periods
        tables = [
            ("agentx_conversations", self.policy.conversations_days, "created_at"),
            ("agentx_costs", self.policy.cost_records_days, "created_at"),
            ("agentx_audit", self.policy.audit_logs_days, "created_at"),
            ("agentx_learned_rules", self.policy.training_data_days, "created_at"),
        ]

        for table, retention_days, date_col in tables:
            cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
            cutoff_str = cutoff.isoformat()

            try:
                # Archive before delete if configured
                if self.policy.archive_before_delete:
                    await self._archive_table(table, date_col, cutoff_str)

                # Delete expired records
                result = await self._db.execute(
                    f"DELETE FROM {table} WHERE {date_col} < ?",
                    [cutoff_str],
                )
                action = {
                    "table": table,
                    "retention_days": retention_days,
                    "cutoff": cutoff_str,
                    "status": "enforced",
                }
                report["actions"].append(action)
            except Exception as e:
                report["actions"].append({
                    "table": table,
                    "status": "error",
                    "error": str(e),
                })

        self._enforcement_log.append(report)
        return report

    async def delete_user_data(self, user_id: str) -> dict[str, Any]:
        """
        GDPR Right to be Forgotten — delete all data for a user.
        """
        if not self.policy.allow_deletion:
            return {"error": "Data deletion not allowed by policy"}

        if not self._db or not self._db.is_connected:
            return {"error": "Database not connected"}

        tables_with_user = [
            ("agentx_users", "id"),
            ("agentx_conversations", "user_id"),
            ("agentx_memory", "user_id"),
            ("agentx_costs", "user_id"),
            ("agentx_audit", "user_id"),
            ("agentx_goals", "user_id"),
        ]

        deleted = {}
        for table, col in tables_with_user:
            try:
                # Archive first
                if self.policy.archive_before_delete:
                    rows = await self._db.fetch_all(
                        f"SELECT * FROM {table} WHERE {col} = ?", [user_id]
                    )
                    if rows:
                        await self._archive_rows(table, user_id, rows)

                await self._db.execute(
                    f"DELETE FROM {table} WHERE {col} = ?", [user_id]
                )
                deleted[table] = "deleted"
            except Exception as e:
                deleted[table] = f"error: {e}"

        logger.info(f"GDPR deletion completed for user: {user_id}")
        return {"user_id": user_id, "tables": deleted, "timestamp": time.time()}

    async def _archive_table(self, table: str, date_col: str, cutoff: str) -> None:
        """Archive expired rows to file before deletion."""
        try:
            rows = await self._db.fetch_all(
                f"SELECT * FROM {table} WHERE {date_col} < ?", [cutoff]
            )
            if rows:
                self._archive_path.mkdir(parents=True, exist_ok=True)
                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                path = self._archive_path / f"{table}_{ts}.jsonl"
                with open(path, "w") as f:
                    for row in rows:
                        f.write(json.dumps(dict(row), default=str) + "\n")
        except Exception as e:
            logger.warning(f"Archive failed for {table}: {e}")

    async def _archive_rows(self, table: str, user_id: str, rows: list) -> None:
        """Archive specific rows before deletion."""
        self._archive_path.mkdir(parents=True, exist_ok=True)
        path = self._archive_path / f"gdpr_{user_id}_{table}.jsonl"
        with open(path, "a") as f:
            for row in rows:
                f.write(json.dumps(dict(row), default=str) + "\n")

    def get_enforcement_log(self) -> list[dict[str, Any]]:
        return self._enforcement_log


# ═══════════════════════════════════════════════════════════════
# 3. FINE-TUNE PIPELINE — Collect, train, evaluate, deploy
# ═══════════════════════════════════════════════════════════════

class FineTuneConfig(BaseModel):
    """Fine-tuning pipeline configuration."""

    min_samples: int = 1000
    export_format: str = "jsonl"        # jsonl, json, csv
    output_dir: str = "./data/finetune"
    validation_split: float = 0.1
    model_provider: str = "anthropic"   # anthropic, openai, local
    base_model: str = "claude-sonnet-4-6"
    quality_threshold: float = 0.8      # Min quality score to include
    dedup_threshold: float = 0.95       # Dedup similar samples


class FineTuneSample(BaseModel):
    """A single fine-tuning sample."""

    query: str
    response: str
    system_prompt: str = ""
    quality_score: float = 0.0
    source: str = ""  # agent, feedback, manual
    validated: bool = False
    created_at: float = Field(default_factory=time.time)


class FineTunePipeline:
    """
    Fine-tuning data pipeline — collect, curate, export, track.

    Self-sufficiency path:
    1. Collect high-quality Q&A pairs during operation
    2. Curate: filter by quality, dedup, validate
    3. Export in training format (JSONL for API fine-tuning)
    4. Track model versions and compare performance

    Usage:
        pipeline = FineTunePipeline()

        # Collect during operation
        pipeline.collect(query="...", response="...", quality_score=0.95)

        # When ready to fine-tune
        stats = pipeline.curate()
        path = pipeline.export()

        # Track versions
        pipeline.register_model("ft-model-v1", metrics={"accuracy": 0.92})
    """

    def __init__(self, config: FineTuneConfig | None = None, db: Any = None):
        self.config = config or FineTuneConfig()
        self._samples: list[FineTuneSample] = []
        self._curated: list[FineTuneSample] = []
        self._models: list[dict[str, Any]] = []
        self._db = db
        self._output_dir = Path(self.config.output_dir)

    def collect(
        self,
        query: str,
        response: str,
        system_prompt: str = "",
        quality_score: float = 0.0,
        source: str = "agent",
        validated: bool = False,
    ) -> None:
        """Collect a fine-tuning sample."""
        sample = FineTuneSample(
            query=query, response=response,
            system_prompt=system_prompt,
            quality_score=quality_score,
            source=source, validated=validated,
        )
        self._samples.append(sample)

    def curate(self) -> dict[str, int]:
        """
        Curate collected samples:
        1. Filter by quality threshold
        2. Deduplicate similar samples
        3. Separate validation set
        """
        # Filter by quality
        quality_filtered = [
            s for s in self._samples
            if s.quality_score >= self.config.quality_threshold
        ]

        # Deduplicate
        seen_hashes: set[str] = set()
        deduped = []
        for sample in quality_filtered:
            h = hashlib.md5(f"{sample.query}:{sample.response[:100]}".encode()).hexdigest()
            if h not in seen_hashes:
                deduped.append(sample)
                seen_hashes.add(h)

        self._curated = deduped

        return {
            "total_collected": len(self._samples),
            "after_quality_filter": len(quality_filtered),
            "after_dedup": len(deduped),
            "ready_for_training": len(deduped) >= self.config.min_samples,
            "min_samples_needed": self.config.min_samples,
        }

    def export(self, format: str = "") -> str:
        """Export curated samples for fine-tuning."""
        fmt = format or self.config.export_format
        self._output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        samples = self._curated or self._samples

        if fmt == "jsonl":
            path = self._output_dir / f"training_{ts}.jsonl"
            with open(path, "w") as f:
                for sample in samples:
                    # Anthropic/OpenAI fine-tuning format
                    entry = {
                        "messages": [
                            {"role": "system", "content": sample.system_prompt} if sample.system_prompt else None,
                            {"role": "user", "content": sample.query},
                            {"role": "assistant", "content": sample.response},
                        ]
                    }
                    entry["messages"] = [m for m in entry["messages"] if m]
                    f.write(json.dumps(entry) + "\n")
        elif fmt == "json":
            path = self._output_dir / f"training_{ts}.json"
            with open(path, "w") as f:
                json.dump([s.model_dump() for s in samples], f, indent=2)
        else:
            path = self._output_dir / f"training_{ts}.csv"
            with open(path, "w") as f:
                f.write("query,response,quality_score,source\n")
                for s in samples:
                    q = s.query.replace('"', '""')
                    r = s.response.replace('"', '""')
                    f.write(f'"{q}","{r}",{s.quality_score},{s.source}\n')

        logger.info(f"Exported {len(samples)} training samples to {path}")
        return str(path)

    def register_model(
        self, model_name: str, metrics: dict[str, float] | None = None,
    ) -> None:
        """Register a fine-tuned model version for tracking."""
        self._models.append({
            "model_name": model_name,
            "version": len(self._models) + 1,
            "training_samples": len(self._curated or self._samples),
            "metrics": metrics or {},
            "created_at": time.time(),
            "base_model": self.config.base_model,
        })

    def get_best_model(self, metric: str = "accuracy") -> dict[str, Any] | None:
        """Get the best fine-tuned model by a specific metric."""
        if not self._models:
            return None
        return max(self._models, key=lambda m: m.get("metrics", {}).get(metric, 0))

    def stats(self) -> dict[str, Any]:
        return {
            "total_collected": len(self._samples),
            "curated": len(self._curated),
            "models_trained": len(self._models),
            "ready_for_training": len(self._curated) >= self.config.min_samples,
            "quality_distribution": self._quality_distribution(),
        }

    def _quality_distribution(self) -> dict[str, int]:
        """Get quality score distribution."""
        dist = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        for s in self._samples:
            if s.quality_score >= 0.9:
                dist["excellent"] += 1
            elif s.quality_score >= 0.7:
                dist["good"] += 1
            elif s.quality_score >= 0.5:
                dist["fair"] += 1
            else:
                dist["poor"] += 1
        return dist


# ═══════════════════════════════════════════════════════════════
# 4. RUNTIME PII MASKING — Mask PII before sending to LLM
# ═══════════════════════════════════════════════════════════════

class RuntimePIIMasker:
    """
    Mask PII at query time — before data reaches the LLM.

    Unlike ingestion-time PII detection (which redacts stored data),
    this masks PII in real-time queries and contexts.

    Usage:
        masker = RuntimePIIMasker()
        masked_query = masker.mask("My email is john@example.com")
        # "My email is [EMAIL_REDACTED]"

        # Send masked to LLM, then unmask response if needed
        response = await llm.generate(masked_query)
        unmasked = masker.unmask(response)
    """

    PATTERNS: dict[str, str] = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone": r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        "ssn": r"\b\d{3}-?\d{2}-?\d{4}\b",
        "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
        "pan": r"\b[A-Z]{5}\d{4}[A-Z]\b",
        "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    }

    def __init__(self, fields: list[str] | None = None):
        self.fields = fields or list(self.PATTERNS.keys())
        self._mask_map: dict[str, str] = {}  # placeholder → original
        self._counter = 0

    def mask(self, text: str) -> str:
        """Mask PII in text, preserving a mapping for unmasking."""
        self._mask_map.clear()
        self._counter = 0
        masked = text

        for field in self.fields:
            pattern = self.PATTERNS.get(field)
            if not pattern:
                continue

            def replacer(m: re.Match) -> str:
                self._counter += 1
                placeholder = f"[{field.upper()}_MASKED_{self._counter}]"
                self._mask_map[placeholder] = m.group()
                return placeholder

            masked = re.sub(pattern, replacer, masked)

        return masked

    def unmask(self, text: str) -> str:
        """Restore masked PII in a response."""
        unmasked = text
        for placeholder, original in self._mask_map.items():
            unmasked = unmasked.replace(placeholder, original)
        return unmasked

    def mask_contexts(self, contexts: list[str]) -> list[str]:
        """Mask PII in all context passages."""
        return [self.mask(ctx) for ctx in contexts]

    @property
    def has_pii(self) -> bool:
        return len(self._mask_map) > 0


# ═══════════════════════════════════════════════════════════════
# 5. CITATION & SOURCE RELIABILITY — Scoring, formatting, verification
# ═══════════════════════════════════════════════════════════════

class SourceReliability(BaseModel):
    """Source reliability metadata."""

    source_id: str = ""
    name: str = ""
    authority_score: float = 0.5  # 0-1: how authoritative is this source
    recency_score: float = 0.5   # 0-1: how recent is the information
    accuracy_score: float = 0.5  # 0-1: historical accuracy rating
    overall_score: float = 0.5
    source_type: str = ""        # official_docs, community, blog, academic
    last_verified: float = 0.0


class CitationManager:
    """
    Source reliability scoring and citation generation.

    Features:
    - Rate source reliability (authority, recency, accuracy)
    - Generate formatted citations
    - Track which sources lead to good/bad answers
    - Prioritize reliable sources in context assembly

    Usage:
        citations = CitationManager()
        citations.register_source("react_docs", authority=0.95, source_type="official_docs")

        # Get reliability-weighted context
        weighted = citations.weight_contexts(contexts, source_ids)

        # Generate citations for response
        formatted = citations.format_citations(source_ids, format="inline")
    """

    def __init__(self):
        self._sources: dict[str, SourceReliability] = {}
        self._feedback: dict[str, list[bool]] = {}  # source_id → [helpful?]

    def register_source(
        self,
        source_id: str,
        name: str = "",
        authority: float = 0.5,
        recency: float = 0.5,
        accuracy: float = 0.5,
        source_type: str = "general",
    ) -> None:
        """Register a source with reliability scores."""
        overall = (authority * 0.4) + (recency * 0.3) + (accuracy * 0.3)
        self._sources[source_id] = SourceReliability(
            source_id=source_id,
            name=name or source_id,
            authority_score=authority,
            recency_score=recency,
            accuracy_score=accuracy,
            overall_score=overall,
            source_type=source_type,
            last_verified=time.time(),
        )

    def get_reliability(self, source_id: str) -> float:
        """Get overall reliability score for a source."""
        source = self._sources.get(source_id)
        if not source:
            return 0.5  # Unknown source default
        return source.overall_score

    def weight_contexts(
        self,
        contexts: list[str],
        source_ids: list[str],
        scores: list[float] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Weight contexts by source reliability.
        Returns [(context, weighted_score)] sorted by score descending.
        """
        base_scores = scores or [1.0] * len(contexts)
        weighted = []

        for ctx, sid, score in zip(contexts, source_ids, base_scores):
            reliability = self.get_reliability(sid)
            weighted_score = score * (0.5 + 0.5 * reliability)  # reliability adjusts score
            weighted.append((ctx, weighted_score))

        weighted.sort(key=lambda x: x[1], reverse=True)
        return weighted

    def record_feedback(self, source_id: str, helpful: bool) -> None:
        """Record whether a source led to a helpful answer."""
        if source_id not in self._feedback:
            self._feedback[source_id] = []
        self._feedback[source_id].append(helpful)

        # Update accuracy score based on feedback
        if source_id in self._sources:
            fb = self._feedback[source_id]
            positive = sum(1 for h in fb if h)
            accuracy = positive / len(fb) if fb else 0.5
            self._sources[source_id].accuracy_score = accuracy
            self._sources[source_id].overall_score = (
                self._sources[source_id].authority_score * 0.4
                + self._sources[source_id].recency_score * 0.3
                + accuracy * 0.3
            )

    def format_citations(
        self,
        source_ids: list[str],
        format: str = "inline",  # inline, footnote, bibliography
    ) -> str:
        """Generate formatted citations."""
        if format == "inline":
            parts = []
            for i, sid in enumerate(source_ids, 1):
                source = self._sources.get(sid)
                name = source.name if source else sid
                reliability = source.overall_score if source else 0.5
                parts.append(f"[{i}] {name} (reliability: {reliability:.0%})")
            return "\n".join(parts)

        elif format == "footnote":
            parts = []
            for i, sid in enumerate(source_ids, 1):
                source = self._sources.get(sid)
                name = source.name if source else sid
                stype = source.source_type if source else "unknown"
                parts.append(f"  {i}. {name} — {stype}")
            return "Sources:\n" + "\n".join(parts)

        elif format == "bibliography":
            parts = []
            for sid in source_ids:
                source = self._sources.get(sid)
                if source:
                    verified = datetime.fromtimestamp(source.last_verified).strftime("%Y-%m-%d") if source.last_verified else "N/A"
                    parts.append(
                        f"- {source.name} ({source.source_type}). "
                        f"Authority: {source.authority_score:.0%}, "
                        f"Last verified: {verified}"
                    )
            return "Bibliography:\n" + "\n".join(parts)

        return ""

    def report(self) -> dict[str, Any]:
        """Get source reliability report."""
        return {
            "total_sources": len(self._sources),
            "sources": {
                sid: {
                    "name": s.name,
                    "reliability": round(s.overall_score, 2),
                    "type": s.source_type,
                    "feedback_count": len(self._feedback.get(sid, [])),
                }
                for sid, s in self._sources.items()
            },
        }

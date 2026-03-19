"""
AgentX - Job Scheduler.

Cron-like scheduling for autonomous agent operations.
Supports: interval, cron expression, one-shot delayed, and event-triggered jobs.

Usage:
    scheduler = JobScheduler()

    # Run every 5 minutes
    scheduler.add_interval("refresh_knowledge", minutes=5, handler=refresh_fn)

    # Run at specific cron time
    scheduler.add_cron("daily_report", hour=9, minute=0, handler=report_fn)

    # Run once after delay
    scheduler.add_delayed("warmup", delay_seconds=10, handler=warmup_fn)

    await scheduler.start()
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Awaitable

from pydantic import BaseModel, Field

logger = logging.getLogger("agentx.daemon")


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    INTERVAL = "interval"
    CRON = "cron"
    DELAYED = "delayed"
    EVENT = "event"


class JobConfig(BaseModel):
    """Configuration for a scheduled job."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    job_type: JobType = JobType.INTERVAL
    enabled: bool = True

    # Interval scheduling
    interval_seconds: float = 0

    # Cron-like scheduling (simplified)
    cron_hour: int = -1       # 0-23, -1 = any
    cron_minute: int = -1     # 0-59, -1 = any
    cron_day_of_week: int = -1  # 0=Mon, 6=Sun, -1 = any
    cron_day_of_month: int = -1  # 1-31, -1 = any

    # Delayed (one-shot)
    delay_seconds: float = 0

    # Event-triggered
    trigger_event: str = ""

    # Execution settings
    max_retries: int = 3
    retry_delay_seconds: float = 5
    timeout_seconds: float = 300  # 5 min default
    run_immediately: bool = False  # run once at startup
    max_runs: int = 0  # 0 = unlimited
    overlap_policy: str = "skip"  # skip, queue, allow

    # Metadata
    description: str = ""
    tags: list[str] = Field(default_factory=list)


class JobRun(BaseModel):
    """Record of a single job execution."""

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    job_id: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0
    duration_ms: float = 0.0
    status: JobStatus = JobStatus.PENDING
    result: Any = None
    error: str = ""
    retry_count: int = 0


class Job:
    """A scheduled job with its handler and execution history."""

    def __init__(
        self,
        config: JobConfig,
        handler: Callable[..., Awaitable[Any]],
    ):
        self.config = config
        self.handler = handler
        self.status = JobStatus.PENDING
        self.run_count = 0
        self.last_run: float = 0.0
        self.next_run: float = 0.0
        self.history: list[JobRun] = []
        self._is_running = False
        self._max_history = 100

        # Calculate initial next_run
        self._schedule_next()

    def _schedule_next(self) -> None:
        """Calculate the next run time."""
        now = time.time()

        if self.config.job_type == JobType.INTERVAL:
            if self.config.run_immediately and self.run_count == 0:
                self.next_run = now
            else:
                self.next_run = now + self.config.interval_seconds

        elif self.config.job_type == JobType.DELAYED:
            if self.run_count == 0:
                self.next_run = now + self.config.delay_seconds
            else:
                self.next_run = 0  # one-shot, don't reschedule

        elif self.config.job_type == JobType.CRON:
            self.next_run = self._next_cron_time()

        elif self.config.job_type == JobType.EVENT:
            self.next_run = 0  # triggered externally

    def _next_cron_time(self) -> float:
        """Calculate next cron execution time."""
        now = datetime.now()
        candidate = now.replace(second=0, microsecond=0)

        # Set minute
        if self.config.cron_minute >= 0:
            candidate = candidate.replace(minute=self.config.cron_minute)
        else:
            candidate = candidate.replace(minute=0)

        # Set hour
        if self.config.cron_hour >= 0:
            candidate = candidate.replace(hour=self.config.cron_hour)

        # If candidate is in the past, advance
        if candidate <= now:
            if self.config.cron_hour >= 0:
                candidate += timedelta(days=1)
            elif self.config.cron_minute >= 0:
                candidate += timedelta(hours=1)
            else:
                candidate += timedelta(minutes=1)

        # Check day of week
        if self.config.cron_day_of_week >= 0:
            while candidate.weekday() != self.config.cron_day_of_week:
                candidate += timedelta(days=1)

        # Check day of month
        if self.config.cron_day_of_month >= 0:
            while candidate.day != self.config.cron_day_of_month:
                candidate += timedelta(days=1)

        return candidate.timestamp()

    def should_run(self) -> bool:
        """Check if this job should run now."""
        if not self.config.enabled:
            return False
        if self.status == JobStatus.PAUSED:
            return False
        if self.config.max_runs > 0 and self.run_count >= self.config.max_runs:
            return False
        if self._is_running and self.config.overlap_policy == "skip":
            return False
        if self.config.job_type == JobType.EVENT:
            return False  # triggered externally
        return self.next_run > 0 and time.time() >= self.next_run

    async def execute(self) -> JobRun:
        """Execute the job handler."""
        run = JobRun(job_id=self.config.id, started_at=time.time())
        self._is_running = True
        self.status = JobStatus.RUNNING

        try:
            result = await asyncio.wait_for(
                self.handler(),
                timeout=self.config.timeout_seconds,
            )
            run.status = JobStatus.COMPLETED
            run.result = result
            self.status = JobStatus.COMPLETED
        except asyncio.TimeoutError:
            run.status = JobStatus.FAILED
            run.error = f"Timeout after {self.config.timeout_seconds}s"
            self.status = JobStatus.FAILED
            logger.error(f"Job '{self.config.name}' timed out")
        except Exception as e:
            run.status = JobStatus.FAILED
            run.error = str(e)
            self.status = JobStatus.FAILED
            logger.error(f"Job '{self.config.name}' failed: {e}")
        finally:
            run.completed_at = time.time()
            run.duration_ms = (run.completed_at - run.started_at) * 1000
            self._is_running = False
            self.run_count += 1
            self.last_run = run.completed_at
            self.history.append(run)
            if len(self.history) > self._max_history:
                self.history = self.history[-self._max_history:]
            self._schedule_next()

        return run


class JobScheduler:
    """
    Async job scheduler for AgentX autonomous operations.

    Manages periodic, cron, delayed, and event-triggered jobs.
    Thread-safe, supports pause/resume, and tracks execution history.
    """

    def __init__(self, tick_interval: float = 1.0):
        self.tick_interval = tick_interval
        self._jobs: dict[str, Job] = {}
        self._running = False
        self._task: asyncio.Task | None = None
        self._event_handlers: dict[str, list[str]] = {}  # event -> [job_ids]
        self._stats = {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "total_runtime_ms": 0.0,
        }

    # --- Job Registration ---

    def add_interval(
        self,
        name: str,
        handler: Callable[..., Awaitable[Any]],
        seconds: float = 0,
        minutes: float = 0,
        hours: float = 0,
        run_immediately: bool = False,
        max_retries: int = 3,
        timeout_seconds: float = 300,
        description: str = "",
        tags: list[str] | None = None,
    ) -> str:
        """Add a job that runs at a fixed interval."""
        total_seconds = seconds + (minutes * 60) + (hours * 3600)
        if total_seconds <= 0:
            raise ValueError("Interval must be > 0")

        config = JobConfig(
            name=name,
            job_type=JobType.INTERVAL,
            interval_seconds=total_seconds,
            run_immediately=run_immediately,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            description=description,
            tags=tags or [],
        )
        job = Job(config, handler)
        self._jobs[config.id] = job
        logger.info(f"Scheduled interval job '{name}' every {total_seconds}s")
        return config.id

    def add_cron(
        self,
        name: str,
        handler: Callable[..., Awaitable[Any]],
        hour: int = -1,
        minute: int = 0,
        day_of_week: int = -1,
        day_of_month: int = -1,
        timeout_seconds: float = 600,
        description: str = "",
        tags: list[str] | None = None,
    ) -> str:
        """Add a job with cron-like scheduling."""
        config = JobConfig(
            name=name,
            job_type=JobType.CRON,
            cron_hour=hour,
            cron_minute=minute,
            cron_day_of_week=day_of_week,
            cron_day_of_month=day_of_month,
            timeout_seconds=timeout_seconds,
            description=description,
            tags=tags or [],
        )
        job = Job(config, handler)
        self._jobs[config.id] = job
        logger.info(f"Scheduled cron job '{name}'")
        return config.id

    def add_delayed(
        self,
        name: str,
        handler: Callable[..., Awaitable[Any]],
        delay_seconds: float = 0,
        description: str = "",
    ) -> str:
        """Add a one-shot delayed job."""
        config = JobConfig(
            name=name,
            job_type=JobType.DELAYED,
            delay_seconds=delay_seconds,
            max_runs=1,
            description=description,
        )
        job = Job(config, handler)
        self._jobs[config.id] = job
        logger.info(f"Scheduled delayed job '{name}' in {delay_seconds}s")
        return config.id

    def add_event_triggered(
        self,
        name: str,
        handler: Callable[..., Awaitable[Any]],
        trigger_event: str = "",
        description: str = "",
    ) -> str:
        """Add a job triggered by an event."""
        config = JobConfig(
            name=name,
            job_type=JobType.EVENT,
            trigger_event=trigger_event,
            description=description,
        )
        job = Job(config, handler)
        self._jobs[config.id] = job
        if trigger_event:
            if trigger_event not in self._event_handlers:
                self._event_handlers[trigger_event] = []
            self._event_handlers[trigger_event].append(config.id)
        logger.info(f"Registered event job '{name}' on '{trigger_event}'")
        return config.id

    # --- Event Triggering ---

    async def trigger_event(self, event: str, data: dict[str, Any] | None = None) -> list[JobRun]:
        """Trigger all jobs listening to an event."""
        runs = []
        job_ids = self._event_handlers.get(event, [])
        for job_id in job_ids:
            job = self._jobs.get(job_id)
            if job and job.config.enabled:
                logger.info(f"Event '{event}' triggered job '{job.config.name}'")
                run = await job.execute()
                self._record_run(run)
                runs.append(run)
        return runs

    # --- Control ---

    async def start(self) -> None:
        """Start the scheduler loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop())
        logger.info(f"Scheduler started with {len(self._jobs)} jobs")

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Scheduler stopped")

    def pause_job(self, job_id: str) -> bool:
        """Pause a job."""
        job = self._jobs.get(job_id)
        if job:
            job.status = JobStatus.PAUSED
            job.config.enabled = False
            return True
        return False

    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job."""
        job = self._jobs.get(job_id)
        if job:
            job.status = JobStatus.PENDING
            job.config.enabled = True
            job._schedule_next()
            return True
        return False

    def remove_job(self, job_id: str) -> bool:
        """Remove a job from the scheduler."""
        if job_id in self._jobs:
            del self._jobs[job_id]
            return True
        return False

    # --- Scheduler Loop ---

    async def _scheduler_loop(self) -> None:
        """Main scheduler tick loop."""
        while self._running:
            try:
                for job in list(self._jobs.values()):
                    if job.should_run():
                        # Run job in background to not block scheduler
                        asyncio.create_task(self._run_job(job))
                await asyncio.sleep(self.tick_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(self.tick_interval)

    async def _run_job(self, job: Job) -> None:
        """Execute a job with retry logic."""
        for attempt in range(job.config.max_retries + 1):
            run = await job.execute()
            self._record_run(run)

            if run.status == JobStatus.COMPLETED:
                break

            if attempt < job.config.max_retries:
                run.retry_count = attempt + 1
                logger.warning(
                    f"Job '{job.config.name}' failed, retry {attempt + 1}/{job.config.max_retries}"
                )
                await asyncio.sleep(job.config.retry_delay_seconds * (attempt + 1))

    def _record_run(self, run: JobRun) -> None:
        """Update global stats."""
        self._stats["total_runs"] += 1
        self._stats["total_runtime_ms"] += run.duration_ms
        if run.status == JobStatus.COMPLETED:
            self._stats["successful_runs"] += 1
        elif run.status == JobStatus.FAILED:
            self._stats["failed_runs"] += 1

    # --- Info ---

    def get_job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def get_jobs_by_tag(self, tag: str) -> list[Job]:
        return [j for j in self._jobs.values() if tag in j.config.tags]

    def list_jobs(self) -> list[dict[str, Any]]:
        """List all jobs with their status."""
        result = []
        for job in self._jobs.values():
            next_run_str = ""
            if job.next_run > 0:
                next_run_str = datetime.fromtimestamp(job.next_run).isoformat()
            result.append({
                "id": job.config.id,
                "name": job.config.name,
                "type": job.config.job_type.value,
                "status": job.status.value,
                "enabled": job.config.enabled,
                "run_count": job.run_count,
                "last_run": datetime.fromtimestamp(job.last_run).isoformat() if job.last_run else None,
                "next_run": next_run_str,
                "description": job.config.description,
                "tags": job.config.tags,
            })
        return result

    def stats(self) -> dict[str, Any]:
        return {
            **self._stats,
            "active_jobs": sum(1 for j in self._jobs.values() if j.config.enabled),
            "paused_jobs": sum(1 for j in self._jobs.values() if j.status == JobStatus.PAUSED),
            "total_jobs": len(self._jobs),
            "running": self._running,
        }

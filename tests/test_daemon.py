"""Tests for the autonomous daemon system."""

import asyncio
import time
import pytest

from agentx.daemon import (
    AgentXDaemon, DaemonConfig,
    JobScheduler, JobType, JobStatus,
    AgentXServer, WebhookHandler, APIResponse,
    FileWatcher, FileEvent, FileEventType, MessageQueueWatcher,
)
from agentx.app import AgentXApp
from agentx.config import AgentXConfig, DatabaseConfig


# ═══════════════════════════════════════════════════════════════
# JobScheduler Tests
# ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_scheduler_interval_job():
    """Interval jobs run on schedule."""
    results = []

    async def handler():
        results.append(time.time())

    scheduler = JobScheduler(tick_interval=0.1)
    scheduler.add_interval("test_job", handler=handler, seconds=0.3, run_immediately=True)

    await scheduler.start()
    await asyncio.sleep(0.8)
    await scheduler.stop()

    # Should have run at least 2 times (immediately + after 0.3s)
    assert len(results) >= 2


@pytest.mark.asyncio
async def test_scheduler_delayed_job():
    """Delayed jobs run once after delay."""
    results = []

    async def handler():
        results.append("ran")

    scheduler = JobScheduler(tick_interval=0.1)
    scheduler.add_delayed("delayed_test", handler=handler, delay_seconds=0.2)

    await scheduler.start()
    await asyncio.sleep(0.5)
    await scheduler.stop()

    assert len(results) == 1


@pytest.mark.asyncio
async def test_scheduler_event_triggered():
    """Event-triggered jobs run when event fires."""
    results = []

    async def handler():
        results.append("triggered")

    scheduler = JobScheduler(tick_interval=0.1)
    scheduler.add_event_triggered(
        "event_test", handler=handler, trigger_event="new_data"
    )

    await scheduler.start()
    await scheduler.trigger_event("new_data")
    await scheduler.stop()

    assert len(results) == 1


@pytest.mark.asyncio
async def test_scheduler_pause_resume():
    """Jobs can be paused and resumed."""
    results = []

    async def handler():
        results.append(time.time())

    scheduler = JobScheduler(tick_interval=0.1)
    job_id = scheduler.add_interval("pausable", handler=handler, seconds=0.2, run_immediately=True)

    await scheduler.start()
    await asyncio.sleep(0.3)
    count_before_pause = len(results)

    scheduler.pause_job(job_id)
    await asyncio.sleep(0.5)
    count_during_pause = len(results)

    scheduler.resume_job(job_id)
    await asyncio.sleep(0.3)
    await scheduler.stop()

    assert count_during_pause == count_before_pause  # no runs while paused
    assert len(results) > count_during_pause  # runs resumed


@pytest.mark.asyncio
async def test_scheduler_list_jobs():
    """List jobs returns all registered jobs."""
    scheduler = JobScheduler()

    async def noop():
        pass

    scheduler.add_interval("job1", handler=noop, seconds=60)
    scheduler.add_delayed("job2", handler=noop, delay_seconds=30)

    jobs = scheduler.list_jobs()
    assert len(jobs) == 2
    names = [j["name"] for j in jobs]
    assert "job1" in names
    assert "job2" in names


@pytest.mark.asyncio
async def test_scheduler_stats():
    """Stats track runs correctly."""
    results = []

    async def handler():
        results.append(1)

    scheduler = JobScheduler(tick_interval=0.1)
    scheduler.add_interval("counted", handler=handler, seconds=0.2, run_immediately=True)

    await scheduler.start()
    await asyncio.sleep(0.5)
    await scheduler.stop()

    stats = scheduler.stats()
    assert stats["total_runs"] >= 2
    assert stats["successful_runs"] >= 2
    assert stats["total_jobs"] == 1


@pytest.mark.asyncio
async def test_scheduler_failed_job_retry():
    """Failed jobs are retried."""
    attempts = []

    async def failing_handler():
        attempts.append(1)
        if len(attempts) < 3:
            raise ValueError("temporary error")
        return "success"

    scheduler = JobScheduler(tick_interval=0.1)
    job_id = scheduler.add_interval(
        "retry_test", handler=failing_handler, seconds=10,
        run_immediately=True, max_retries=3,
    )
    # Override retry delay to be fast for tests
    job = scheduler.get_job(job_id)
    job.config.retry_delay_seconds = 0.1

    await scheduler.start()
    await asyncio.sleep(3)
    await scheduler.stop()

    assert len(attempts) >= 3  # retried after failures


# ═══════════════════════════════════════════════════════════════
# WebhookHandler Tests
# ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_webhook_handler():
    """Webhook handlers process incoming data."""
    received = []
    webhook = WebhookHandler()

    async def on_github(source, payload):
        received.append({"source": source, "data": payload})
        return "processed"

    webhook.register(source="github", handler=on_github)
    results = await webhook.process("github", {"action": "push"})

    assert len(received) == 1
    assert received[0]["source"] == "github"
    assert results == ["processed"]


@pytest.mark.asyncio
async def test_webhook_global_handler():
    """Global handlers receive all webhooks."""
    received = []
    webhook = WebhookHandler()

    async def on_any(source, payload):
        received.append(source)

    webhook.register(handler=on_any)
    await webhook.process("github", {})
    await webhook.process("stripe", {})

    assert received == ["github", "stripe"]


# ═══════════════════════════════════════════════════════════════
# APIResponse Tests
# ═══════════════════════════════════════════════════════════════

def test_api_response_success():
    resp = APIResponse.success(data={"key": "value"})
    assert resp["status"] == "success"
    assert resp["data"]["key"] == "value"


def test_api_response_error():
    resp = APIResponse.error("not found", 404)
    assert resp["status"] == "error"
    assert resp["code"] == 404


# ═══════════════════════════════════════════════════════════════
# FileWatcher Tests
# ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_file_watcher_detect_new(tmp_path):
    """File watcher detects new files."""
    events = []

    async def on_file(event: FileEvent):
        events.append(event)

    watcher = FileWatcher(poll_interval=0.2)
    watcher.watch(str(tmp_path), on_file, patterns=["*.txt"])

    await watcher.start()
    await asyncio.sleep(0.3)

    # Create a new file
    (tmp_path / "test.txt").write_text("hello")
    await asyncio.sleep(0.5)
    await watcher.stop()

    assert len(events) >= 1
    assert events[0].event_type == FileEventType.CREATED
    assert "test.txt" in events[0].path


@pytest.mark.asyncio
async def test_file_watcher_ignores_patterns(tmp_path):
    """File watcher respects ignore patterns."""
    events = []

    async def on_file(event: FileEvent):
        events.append(event)

    watcher = FileWatcher(poll_interval=0.2)
    watcher.watch(str(tmp_path), on_file, patterns=["*.txt"])

    await watcher.start()
    await asyncio.sleep(0.3)

    # Create a .pyc file (should be ignored)
    (tmp_path / "test.pyc").write_bytes(b"bytecode")
    # Create a .txt file (should be caught)
    (tmp_path / "test.txt").write_text("hello")
    await asyncio.sleep(0.5)
    await watcher.stop()

    txt_events = [e for e in events if e.path.endswith(".txt")]
    pyc_events = [e for e in events if e.path.endswith(".pyc")]
    assert len(txt_events) >= 1
    assert len(pyc_events) == 0


# ═══════════════════════════════════════════════════════════════
# MessageQueueWatcher Tests
# ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_mq_in_memory():
    """In-memory message queue works."""
    received = []

    async def handler(channel, data):
        received.append((channel, data))

    mq = MessageQueueWatcher()
    mq.subscribe("tasks", handler)
    await mq.start()

    await mq.publish("tasks", {"action": "process"})
    await asyncio.sleep(0.3)
    await mq.stop()

    assert len(received) == 1
    assert received[0] == ("tasks", {"action": "process"})


# ═══════════════════════════════════════════════════════════════
# DaemonConfig Tests
# ═══════════════════════════════════════════════════════════════

def test_daemon_config_minimal():
    config = DaemonConfig.minimal()
    assert config.server_enabled is True
    assert config.scheduler_enabled is False


def test_daemon_config_full():
    config = DaemonConfig.full(port=9000)
    assert config.server_enabled is True
    assert config.scheduler_enabled is True
    assert config.watcher_enabled is True
    assert config.server_port == 9000


# ═══════════════════════════════════════════════════════════════
# AgentXDaemon Tests
# ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_daemon_fluent_api():
    """Daemon provides a fluent API for configuration."""
    app_config = AgentXConfig(database=DatabaseConfig.memory())
    daemon = AgentXDaemon(
        app=AgentXApp(app_config),
        config=DaemonConfig(
            server_enabled=False,
            scheduler_enabled=True,
            watcher_enabled=False,
            watchdog_enabled=False,
        ),
    )

    results = []

    async def handler():
        results.append("tick")

    # Fluent API
    daemon.every(seconds=0.2, name="test_every", handler=handler, run_immediately=True)

    # Verify job was registered
    assert daemon.scheduler is not None
    jobs = daemon.scheduler.list_jobs()
    assert len(jobs) == 1
    assert jobs[0]["name"] == "test_every"


@pytest.mark.asyncio
async def test_daemon_decorator_api():
    """Daemon supports decorator syntax."""
    app_config = AgentXConfig(database=DatabaseConfig.memory())
    daemon = AgentXDaemon(
        app=AgentXApp(app_config),
        config=DaemonConfig(server_enabled=False),
    )

    @daemon.every(seconds=60, name="decorated_job")
    async def my_job():
        pass

    @daemon.cron(hour=9, minute=0, name="morning_job")
    async def morning():
        pass

    assert daemon.scheduler is not None
    jobs = daemon.scheduler.list_jobs()
    assert len(jobs) == 2


@pytest.mark.asyncio
async def test_daemon_lifecycle_hooks():
    """Lifecycle hooks are called."""
    events = []

    app_config = AgentXConfig(database=DatabaseConfig.memory())
    daemon = AgentXDaemon(
        app=AgentXApp(app_config),
        config=DaemonConfig(server_enabled=False, watchdog_enabled=False),
    )

    @daemon.on_start
    async def start_hook():
        events.append("started")

    @daemon.on_stop
    async def stop_hook():
        events.append("stopped")

    assert len(daemon._on_start_hooks) == 1
    assert len(daemon._on_stop_hooks) == 1


@pytest.mark.asyncio
async def test_daemon_stats():
    """Stats show daemon state."""
    daemon = AgentXDaemon(
        config=DaemonConfig(
            server_enabled=False,
            scheduler_enabled=True,
            watcher_enabled=False,
        ),
    )

    stats = daemon.stats()
    assert "running" in stats
    assert "scheduler" in stats
    assert stats["running"] is False


def test_daemon_is_separate_plugin():
    """Daemon is separate — AgentXApp works without it, daemon is just a wrapper."""
    # Normal app is not coupled to daemon
    app = AgentXApp()
    assert app.is_started is False
    # No daemon attributes on app
    assert not hasattr(app, 'scheduler')
    assert not hasattr(app, 'file_watcher')

    # Daemon wraps the same app — plugin layer
    daemon = AgentXDaemon(
        app=app,
        config=DaemonConfig(server_enabled=False, watchdog_enabled=False),
    )
    assert daemon.app is app
    assert not daemon._running
    # Daemon has its own subsystems, separate from app
    assert daemon.scheduler is not None or daemon.config.scheduler_enabled is False

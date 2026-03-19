"""
AgentX - Autonomous Daemon Runner.

The 24/7 autonomous agent system. Combines all components into a
self-running, self-healing, continuously operating daemon.

Features:
- Continuous event loop with graceful shutdown (SIGINT, SIGTERM)
- Job scheduling (interval, cron, delayed, event-triggered)
- HTTP/WebSocket API server for external communication
- File system watchers for auto-ingestion
- Message queue integration (Redis pub/sub)
- Process watchdog with auto-restart
- Health monitoring with self-healing
- Configurable per project (interview bot, trading bot, etc.)

Usage:
    # Minimal — just run the app 24/7 with an API
    daemon = AgentXDaemon(app)
    await daemon.run_forever()

    # Full autonomous setup
    daemon = AgentXDaemon(
        app=app,
        config=DaemonConfig(
            server_enabled=True,
            server_port=8080,
            scheduler_enabled=True,
            watcher_enabled=True,
        ),
    )

    # Add scheduled jobs
    daemon.every(minutes=5, name="refresh_knowledge", handler=refresh_fn)
    daemon.every(hours=24, name="daily_report", handler=report_fn)
    daemon.cron(hour=9, minute=0, name="morning_briefing", handler=briefing_fn)

    # Watch files for auto-ingestion
    daemon.watch("/data/uploads", handler=ingest_fn, patterns=["*.pdf"])

    # Webhooks
    daemon.on_webhook("github", handler=github_handler)
    daemon.on_webhook("stripe", handler=stripe_handler)

    # Run forever
    await daemon.run_forever()

CLI:
    python -m agentx.daemon --config config.yaml
    python -m agentx.daemon --port 8080 --workers 4
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from typing import Any, Callable, Awaitable

from pydantic import BaseModel, Field

from ..app import AgentXApp
from ..config import AgentXConfig
from .scheduler import JobScheduler, JobStatus
from .server import AgentXServer, WebhookHandler
from .watcher import FileWatcher, FileEvent, MessageQueueWatcher

logger = logging.getLogger("agentx.daemon")


class DaemonConfig(BaseModel):
    """Configuration for the autonomous daemon."""

    # Server
    server_enabled: bool = True
    server_host: str = "0.0.0.0"
    server_port: int = 8080
    server_api_key: str = ""  # empty = no auth
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # Scheduler
    scheduler_enabled: bool = True
    scheduler_tick_interval: float = 1.0

    # File watcher
    watcher_enabled: bool = False
    watcher_poll_interval: float = 2.0
    watch_paths: list[dict[str, Any]] = Field(default_factory=list)

    # Message queue
    mq_enabled: bool = False
    mq_redis_url: str = ""
    mq_channels: list[str] = Field(default_factory=list)

    # Watchdog
    watchdog_enabled: bool = True
    watchdog_interval_seconds: float = 30.0
    auto_restart_on_failure: bool = True
    max_restart_attempts: int = 5
    restart_cooldown_seconds: float = 10.0

    # Health monitoring
    health_check_interval_seconds: float = 60.0
    unhealthy_threshold: int = 3  # consecutive failures before restart

    # Process
    graceful_shutdown_timeout: float = 30.0
    pid_file: str = ""  # write PID for process management

    # Logging
    log_level: str = "INFO"
    log_file: str = ""  # empty = stdout only

    @classmethod
    def minimal(cls) -> DaemonConfig:
        """Minimal config — just API server."""
        return cls(
            server_enabled=True,
            scheduler_enabled=False,
            watcher_enabled=False,
            mq_enabled=False,
        )

    @classmethod
    def full(cls, port: int = 8080) -> DaemonConfig:
        """Full autonomous config — everything enabled."""
        return cls(
            server_enabled=True,
            server_port=port,
            scheduler_enabled=True,
            watcher_enabled=True,
            mq_enabled=False,
            watchdog_enabled=True,
        )

    @classmethod
    def from_env(cls) -> DaemonConfig:
        """Load from environment variables."""
        return cls(
            server_enabled=os.getenv("AGENTX_SERVER", "true").lower() == "true",
            server_host=os.getenv("AGENTX_HOST", "0.0.0.0"),
            server_port=int(os.getenv("AGENTX_PORT", "8080")),
            server_api_key=os.getenv("AGENTX_API_KEY", ""),
            scheduler_enabled=os.getenv("AGENTX_SCHEDULER", "true").lower() == "true",
            watcher_enabled=os.getenv("AGENTX_WATCHER", "false").lower() == "true",
            mq_enabled=os.getenv("AGENTX_MQ", "false").lower() == "true",
            mq_redis_url=os.getenv("AGENTX_REDIS_URL", ""),
            watchdog_enabled=os.getenv("AGENTX_WATCHDOG", "true").lower() == "true",
            log_level=os.getenv("AGENTX_LOG_LEVEL", "INFO"),
            log_file=os.getenv("AGENTX_LOG_FILE", ""),
            pid_file=os.getenv("AGENTX_PID_FILE", ""),
        )


class AgentXDaemon:
    """
    Autonomous 24/7 daemon for AgentX.

    This is the main entry point for running AgentX as a
    continuously operating, self-healing service.

    Works for ANY project type:
    - Interview bot: schedule question updates, watch resume uploads
    - Trading bot: periodic market scans, webhook for price alerts
    - Content pipeline: watch upload folder, auto-process media
    - Customer support: 24/7 API, scheduled report generation
    """

    def __init__(
        self,
        app: AgentXApp | None = None,
        config: DaemonConfig | None = None,
        app_config: AgentXConfig | None = None,
    ):
        self.config = config or DaemonConfig()
        self._app = app or AgentXApp(app_config)

        # Sub-systems
        self.scheduler = JobScheduler(
            tick_interval=self.config.scheduler_tick_interval
        ) if self.config.scheduler_enabled else None

        self.server = AgentXServer(
            host=self.config.server_host,
            port=self.config.server_port,
            api_key=self.config.server_api_key,
            cors_origins=self.config.cors_origins,
        ) if self.config.server_enabled else None

        self.file_watcher = FileWatcher(
            poll_interval=self.config.watcher_poll_interval
        ) if self.config.watcher_enabled else None

        self.mq_watcher = MessageQueueWatcher(
            redis_url=self.config.mq_redis_url
        ) if self.config.mq_enabled else None

        # State
        self._running = False
        self._start_time = 0.0
        self._restart_count = 0
        self._consecutive_health_failures = 0
        self._shutdown_event: asyncio.Event | None = None
        self._tasks: list[asyncio.Task] = []

        # Lifecycle hooks
        self._on_start_hooks: list[Callable[..., Awaitable[Any]]] = []
        self._on_stop_hooks: list[Callable[..., Awaitable[Any]]] = []
        self._on_error_hooks: list[Callable[[Exception], Awaitable[Any]]] = []
        self._on_health_fail_hooks: list[Callable[..., Awaitable[Any]]] = []

    # ---- Convenience API (fluent interface) ----

    def every(
        self,
        seconds: float = 0,
        minutes: float = 0,
        hours: float = 0,
        name: str = "",
        handler: Callable[..., Awaitable[Any]] | None = None,
        run_immediately: bool = False,
        **kwargs: Any,
    ) -> str | Callable:
        """Schedule a repeating job. Can be used as decorator."""
        if not self.scheduler:
            self.scheduler = JobScheduler()

        if handler:
            return self.scheduler.add_interval(
                name=name or handler.__name__,
                handler=handler,
                seconds=seconds,
                minutes=minutes,
                hours=hours,
                run_immediately=run_immediately,
                **kwargs,
            )

        def decorator(fn: Callable[..., Awaitable[Any]]) -> Callable:
            self.scheduler.add_interval(
                name=name or fn.__name__,
                handler=fn,
                seconds=seconds,
                minutes=minutes,
                hours=hours,
                run_immediately=run_immediately,
                **kwargs,
            )
            return fn
        return decorator

    def cron(
        self,
        hour: int = -1,
        minute: int = 0,
        day_of_week: int = -1,
        day_of_month: int = -1,
        name: str = "",
        handler: Callable[..., Awaitable[Any]] | None = None,
        **kwargs: Any,
    ) -> str | Callable:
        """Schedule a cron-like job. Can be used as decorator."""
        if not self.scheduler:
            self.scheduler = JobScheduler()

        if handler:
            return self.scheduler.add_cron(
                name=name or handler.__name__,
                handler=handler,
                hour=hour,
                minute=minute,
                day_of_week=day_of_week,
                day_of_month=day_of_month,
                **kwargs,
            )

        def decorator(fn: Callable[..., Awaitable[Any]]) -> Callable:
            self.scheduler.add_cron(
                name=name or fn.__name__,
                handler=fn,
                hour=hour,
                minute=minute,
                day_of_week=day_of_week,
                day_of_month=day_of_month,
                **kwargs,
            )
            return fn
        return decorator

    def delayed(
        self,
        delay_seconds: float,
        name: str = "",
        handler: Callable[..., Awaitable[Any]] | None = None,
    ) -> str | Callable:
        """Schedule a one-shot delayed job."""
        if not self.scheduler:
            self.scheduler = JobScheduler()

        if handler:
            return self.scheduler.add_delayed(name=name or handler.__name__, handler=handler, delay_seconds=delay_seconds)

        def decorator(fn: Callable[..., Awaitable[Any]]) -> Callable:
            self.scheduler.add_delayed(name=name or fn.__name__, handler=fn, delay_seconds=delay_seconds)
            return fn
        return decorator

    def on_event(
        self,
        event: str,
        name: str = "",
        handler: Callable[..., Awaitable[Any]] | None = None,
    ) -> str | Callable:
        """Register an event-triggered job."""
        if not self.scheduler:
            self.scheduler = JobScheduler()

        if handler:
            return self.scheduler.add_event_triggered(name=name or handler.__name__, handler=handler, trigger_event=event)

        def decorator(fn: Callable[..., Awaitable[Any]]) -> Callable:
            self.scheduler.add_event_triggered(name=name or fn.__name__, handler=fn, trigger_event=event)
            return fn
        return decorator

    def watch(
        self,
        path: str,
        handler: Callable[[FileEvent], Awaitable[Any]],
        patterns: list[str] | None = None,
        recursive: bool = True,
    ) -> None:
        """Watch a directory for file changes."""
        if not self.file_watcher:
            self.file_watcher = FileWatcher()
        self.file_watcher.watch(path, handler, patterns=patterns, recursive=recursive)

    def on_webhook(
        self,
        source: str,
        handler: Callable[..., Awaitable[Any]],
    ) -> None:
        """Register a webhook handler."""
        if self.server:
            self.server.webhooks.register(source=source, handler=handler)

    def on_message(
        self,
        channel: str,
        handler: Callable[[str, dict[str, Any]], Awaitable[Any]],
    ) -> None:
        """Subscribe to a message queue channel."""
        if not self.mq_watcher:
            self.mq_watcher = MessageQueueWatcher(redis_url=self.config.mq_redis_url)
        self.mq_watcher.subscribe(channel, handler)

    def add_api_route(
        self,
        path: str,
        handler: Callable[..., Awaitable[dict]],
    ) -> None:
        """Add a custom API endpoint."""
        if self.server:
            self.server.add_route(path, handler)

    # ---- Lifecycle Hooks ----

    def on_start(self, fn: Callable[..., Awaitable[Any]]) -> Callable:
        """Register a hook to run after daemon starts."""
        self._on_start_hooks.append(fn)
        return fn

    def on_stop(self, fn: Callable[..., Awaitable[Any]]) -> Callable:
        """Register a hook to run before daemon stops."""
        self._on_stop_hooks.append(fn)
        return fn

    def on_error(self, fn: Callable[[Exception], Awaitable[Any]]) -> Callable:
        """Register an error handler."""
        self._on_error_hooks.append(fn)
        return fn

    # ---- Main Run ----

    async def run_forever(self) -> None:
        """
        Start the daemon and run forever.

        This is the primary entry point. It:
        1. Initializes the AgentX app
        2. Starts the HTTP server
        3. Starts the scheduler
        4. Starts file watchers
        5. Starts message queue listeners
        6. Runs the watchdog
        7. Handles signals for graceful shutdown
        """
        self._shutdown_event = asyncio.Event()
        self._start_time = time.time()
        self._running = True

        # Setup logging
        self._setup_logging()

        # Write PID file
        if self.config.pid_file:
            with open(self.config.pid_file, "w") as f:
                f.write(str(os.getpid()))

        # Register signal handlers
        self._register_signals()

        logger.info("=" * 60)
        logger.info(f"  AgentX Daemon Starting")
        logger.info(f"  PID: {os.getpid()}")
        logger.info(f"  Server: {'enabled' if self.config.server_enabled else 'disabled'}")
        logger.info(f"  Scheduler: {'enabled' if self.config.scheduler_enabled else 'disabled'}")
        logger.info(f"  Watcher: {'enabled' if self.config.watcher_enabled else 'disabled'}")
        logger.info(f"  MQ: {'enabled' if self.config.mq_enabled else 'disabled'}")
        logger.info(f"  Watchdog: {'enabled' if self.config.watchdog_enabled else 'disabled'}")
        logger.info("=" * 60)

        try:
            # 1. Start AgentX app
            await self._app.start()
            logger.info("AgentX app initialized")

            # 2. Start sub-systems
            await self._start_subsystems()

            # 3. Run lifecycle hooks
            for hook in self._on_start_hooks:
                try:
                    await hook()
                except Exception as e:
                    logger.error(f"Start hook error: {e}")

            logger.info("AgentX Daemon is running. Press Ctrl+C to stop.")

            # 4. Start background tasks
            if self.config.watchdog_enabled:
                self._tasks.append(asyncio.create_task(self._watchdog_loop()))
            self._tasks.append(asyncio.create_task(self._health_monitor_loop()))

            # 5. Wait for shutdown signal
            await self._shutdown_event.wait()

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Daemon error: {e}")
            for hook in self._on_error_hooks:
                try:
                    await hook(e)
                except Exception:
                    pass
        finally:
            await self._shutdown()

    async def _start_subsystems(self) -> None:
        """Start all configured sub-systems."""
        # Server
        if self.server:
            self.server.set_app(self._app)
            self.server.set_daemon(self)
            await self.server.start()

        # Scheduler
        if self.scheduler:
            await self.scheduler.start()

        # File watcher
        if self.file_watcher:
            # Add configured watch paths
            for wp in self.config.watch_paths:
                if "path" in wp and "handler" not in wp:
                    logger.warning(f"Watch path '{wp['path']}' has no handler, skipping")
            await self.file_watcher.start()

        # Message queue
        if self.mq_watcher:
            await self.mq_watcher.start()

        # Start the task queue workers
        if self._app.queue:
            await self._app.queue.start()

    async def _shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down AgentX Daemon...")
        self._running = False

        # Run stop hooks
        for hook in self._on_stop_hooks:
            try:
                await hook()
            except Exception as e:
                logger.error(f"Stop hook error: {e}")

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        # Stop sub-systems (reverse order)
        if self.mq_watcher:
            await self.mq_watcher.stop()
        if self.file_watcher:
            await self.file_watcher.stop()
        if self.scheduler:
            await self.scheduler.stop()
        if self.server:
            await self.server.stop()

        # Stop app
        await self._app.stop()

        # Remove PID file
        if self.config.pid_file and os.path.exists(self.config.pid_file):
            os.remove(self.config.pid_file)

        logger.info("AgentX Daemon stopped.")

    # ---- Signal Handling ----

    def _register_signals(self) -> None:
        """Register OS signal handlers for graceful shutdown."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._signal_handler, sig)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                signal.signal(sig, lambda s, f: self._signal_handler(s))

    def _signal_handler(self, sig: Any) -> None:
        """Handle shutdown signal."""
        sig_name = signal.Signals(sig).name if isinstance(sig, int) else str(sig)
        logger.info(f"Received {sig_name}, initiating graceful shutdown...")
        if self._shutdown_event:
            self._shutdown_event.set()

    # ---- Watchdog ----

    async def _watchdog_loop(self) -> None:
        """Monitor system health and auto-restart on failure."""
        while self._running:
            try:
                await asyncio.sleep(self.config.watchdog_interval_seconds)

                # Check sub-systems
                issues = []

                if self.server and not self.server.is_running:
                    issues.append("server")
                if self.scheduler and not self.scheduler._running:
                    issues.append("scheduler")
                if self.file_watcher and not self.file_watcher._running:
                    issues.append("file_watcher")
                if self.mq_watcher and not self.mq_watcher._running:
                    issues.append("mq_watcher")

                if issues and self.config.auto_restart_on_failure:
                    if self._restart_count < self.config.max_restart_attempts:
                        logger.warning(f"Watchdog: restarting failed subsystems: {issues}")
                        await self._restart_subsystems(issues)
                        self._restart_count += 1
                    else:
                        logger.error(
                            f"Watchdog: max restart attempts ({self.config.max_restart_attempts}) "
                            f"reached. Manual intervention required."
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Watchdog error: {e}")

    async def _restart_subsystems(self, subsystems: list[str]) -> None:
        """Restart failed sub-systems."""
        await asyncio.sleep(self.config.restart_cooldown_seconds)

        for name in subsystems:
            try:
                if name == "server" and self.server:
                    await self.server.stop()
                    self.server.set_app(self._app)
                    self.server.set_daemon(self)
                    await self.server.start()
                    logger.info("Watchdog: server restarted")

                elif name == "scheduler" and self.scheduler:
                    await self.scheduler.stop()
                    await self.scheduler.start()
                    logger.info("Watchdog: scheduler restarted")

                elif name == "file_watcher" and self.file_watcher:
                    await self.file_watcher.stop()
                    await self.file_watcher.start()
                    logger.info("Watchdog: file watcher restarted")

                elif name == "mq_watcher" and self.mq_watcher:
                    await self.mq_watcher.stop()
                    await self.mq_watcher.start()
                    logger.info("Watchdog: message queue watcher restarted")

            except Exception as e:
                logger.error(f"Watchdog: failed to restart {name}: {e}")

    # ---- Health Monitoring ----

    async def _health_monitor_loop(self) -> None:
        """Periodic health check loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.health_check_interval_seconds)

                if self._app.health:
                    health = await self._app.health.check_all()
                    if health["healthy"]:
                        self._consecutive_health_failures = 0
                    else:
                        self._consecutive_health_failures += 1
                        logger.warning(
                            f"Health check failed ({self._consecutive_health_failures}/"
                            f"{self.config.unhealthy_threshold}): {health['checks']}"
                        )

                        if self._consecutive_health_failures >= self.config.unhealthy_threshold:
                            logger.error("Health threshold exceeded, triggering recovery")
                            for hook in self._on_health_fail_hooks:
                                try:
                                    await hook()
                                except Exception:
                                    pass

                            # Try to reconnect database
                            if self._app.db and not self._app.db.is_connected:
                                try:
                                    await self._app.db.connect()
                                    logger.info("Database reconnected")
                                except Exception as e:
                                    logger.error(f"Database reconnect failed: {e}")

                            self._consecutive_health_failures = 0

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")

    # ---- Logging ----

    def _setup_logging(self) -> None:
        """Configure daemon logging."""
        level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        console = logging.StreamHandler()
        console.setFormatter(formatter)

        # File handler
        if self.config.log_file:
            os.makedirs(os.path.dirname(self.config.log_file) or ".", exist_ok=True)
            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setFormatter(formatter)
            logging.getLogger("agentx").addHandler(file_handler)

        root_logger = logging.getLogger("agentx")
        root_logger.setLevel(level)
        if not root_logger.handlers:
            root_logger.addHandler(console)

    # ---- Info ----

    @property
    def app(self) -> AgentXApp:
        return self._app

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time if self._start_time else 0

    def stats(self) -> dict[str, Any]:
        """Full daemon stats."""
        result: dict[str, Any] = {
            "running": self._running,
            "uptime_seconds": self.uptime_seconds,
            "restart_count": self._restart_count,
            "pid": os.getpid(),
        }

        if self.scheduler:
            result["scheduler"] = self.scheduler.stats()
        if self.server:
            result["server"] = self.server.stats()
        if self.file_watcher:
            result["file_watcher"] = self.file_watcher.stats()
        if self.mq_watcher:
            result["mq_watcher"] = self.mq_watcher.stats()

        return result


# ═══════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════

def run_daemon(
    app_config: AgentXConfig | None = None,
    daemon_config: DaemonConfig | None = None,
    setup_fn: Callable[[AgentXDaemon], Awaitable[None]] | None = None,
) -> None:
    """
    Run the AgentX daemon from command line.

    Usage from project code:
        from agentx.daemon import run_daemon

        async def setup(daemon):
            daemon.every(minutes=5, name="refresh", handler=my_refresh)

        run_daemon(setup_fn=setup)
    """
    async def _main():
        app = AgentXApp(app_config or AgentXConfig.from_env())
        daemon = AgentXDaemon(
            app=app,
            config=daemon_config or DaemonConfig.from_env(),
        )
        if setup_fn:
            await setup_fn(daemon)
        await daemon.run_forever()

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass

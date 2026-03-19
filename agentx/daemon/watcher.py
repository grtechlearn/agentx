"""
AgentX - File & Event Watchers.

Monitor file system changes, message queues, and external events
to trigger autonomous agent actions.

Usage:
    watcher = FileWatcher()
    watcher.watch("/data/uploads", on_new_file, patterns=["*.pdf", "*.csv"])
    watcher.watch("/config", on_config_change, patterns=["*.yaml", "*.json"])
    await watcher.start()

    # Message queue watcher
    mq = MessageQueueWatcher(redis_url="redis://localhost:6379")
    mq.subscribe("tasks", on_task)
    await mq.start()
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable, Awaitable
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger("agentx.daemon")


class FileEventType(str, Enum):
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"


class FileEvent(BaseModel):
    """A file system change event."""

    path: str = ""
    event_type: FileEventType = FileEventType.MODIFIED
    size: int = 0
    timestamp: float = Field(default_factory=time.time)


class WatchConfig(BaseModel):
    """Configuration for a file watch."""

    path: str = ""
    patterns: list[str] = Field(default_factory=lambda: ["*"])
    recursive: bool = True
    ignore_patterns: list[str] = Field(default_factory=lambda: [
        "*.pyc", "__pycache__", ".git", ".DS_Store", "*.swp", "*.tmp",
    ])
    debounce_seconds: float = 1.0  # ignore rapid duplicate events


class FileWatcher:
    """
    File system watcher for autonomous operations.

    Monitors directories for changes and triggers handlers.
    Uses polling (works everywhere, no native deps needed).
    Can upgrade to watchdog library if installed.
    """

    def __init__(self, poll_interval: float = 2.0):
        self.poll_interval = poll_interval
        self._watches: list[tuple[WatchConfig, Callable[..., Awaitable[Any]]]] = []
        self._file_hashes: dict[str, str] = {}
        self._file_mtimes: dict[str, float] = {}
        self._running = False
        self._task: asyncio.Task | None = None
        self._event_count = 0
        self._last_events: list[FileEvent] = []
        self._max_events = 100

    def watch(
        self,
        path: str,
        handler: Callable[[FileEvent], Awaitable[Any]],
        patterns: list[str] | None = None,
        recursive: bool = True,
        ignore_patterns: list[str] | None = None,
        debounce_seconds: float = 1.0,
    ) -> None:
        """Add a directory to watch."""
        config = WatchConfig(
            path=os.path.abspath(path),
            patterns=patterns or ["*"],
            recursive=recursive,
            ignore_patterns=ignore_patterns or WatchConfig().ignore_patterns,
            debounce_seconds=debounce_seconds,
        )
        self._watches.append((config, handler))
        logger.info(f"Watching: {config.path} (patterns={config.patterns})")

    async def start(self) -> None:
        """Start watching."""
        if self._running:
            return

        # Initial scan to establish baseline
        for config, _ in self._watches:
            self._scan_directory(config)

        self._running = True
        self._task = asyncio.create_task(self._watch_loop())
        logger.info(f"File watcher started, monitoring {len(self._watches)} directories")

    async def stop(self) -> None:
        """Stop watching."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("File watcher stopped")

    def _scan_directory(self, config: WatchConfig) -> dict[str, float]:
        """Scan a directory and return file -> mtime mapping."""
        files: dict[str, float] = {}
        path = Path(config.path)

        if not path.exists():
            return files

        try:
            if config.recursive:
                entries = path.rglob("*")
            else:
                entries = path.glob("*")

            for entry in entries:
                if not entry.is_file():
                    continue

                # Check ignore patterns
                name = entry.name
                rel_path = str(entry.relative_to(path))
                if any(fnmatch(name, p) or fnmatch(rel_path, p) for p in config.ignore_patterns):
                    continue

                # Check include patterns
                if not any(fnmatch(name, p) for p in config.patterns):
                    continue

                try:
                    files[str(entry)] = entry.stat().st_mtime
                except OSError:
                    pass
        except PermissionError:
            logger.warning(f"Permission denied: {config.path}")

        return files

    async def _watch_loop(self) -> None:
        """Main polling loop."""
        while self._running:
            try:
                for config, handler in self._watches:
                    current_files = self._scan_directory(config)
                    previous_files = {
                        k: v for k, v in self._file_mtimes.items()
                        if k.startswith(config.path)
                    }

                    # Detect new files
                    for filepath, mtime in current_files.items():
                        if filepath not in previous_files:
                            event = FileEvent(
                                path=filepath,
                                event_type=FileEventType.CREATED,
                                size=os.path.getsize(filepath) if os.path.exists(filepath) else 0,
                            )
                            await self._emit_event(event, handler)
                        elif mtime > previous_files.get(filepath, 0) + config.debounce_seconds:
                            event = FileEvent(
                                path=filepath,
                                event_type=FileEventType.MODIFIED,
                                size=os.path.getsize(filepath) if os.path.exists(filepath) else 0,
                            )
                            await self._emit_event(event, handler)

                    # Detect deleted files
                    for filepath in previous_files:
                        if filepath not in current_files and filepath.startswith(config.path):
                            event = FileEvent(
                                path=filepath,
                                event_type=FileEventType.DELETED,
                            )
                            await self._emit_event(event, handler)

                    # Update tracking
                    # Remove old entries for this watch path
                    self._file_mtimes = {
                        k: v for k, v in self._file_mtimes.items()
                        if not k.startswith(config.path)
                    }
                    self._file_mtimes.update(current_files)

                await asyncio.sleep(self.poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"File watcher error: {e}")
                await asyncio.sleep(self.poll_interval)

    async def _emit_event(
        self,
        event: FileEvent,
        handler: Callable[[FileEvent], Awaitable[Any]],
    ) -> None:
        """Emit a file event to the handler."""
        self._event_count += 1
        self._last_events.append(event)
        if len(self._last_events) > self._max_events:
            self._last_events = self._last_events[-self._max_events:]

        logger.debug(f"File event: {event.event_type.value} {event.path}")
        try:
            await handler(event)
        except Exception as e:
            logger.error(f"File handler error for {event.path}: {e}")

    def stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "watches": len(self._watches),
            "events_processed": self._event_count,
            "tracked_files": len(self._file_mtimes),
        }


class MessageQueueWatcher:
    """
    Message queue watcher for event-driven agent triggers.

    Supports Redis pub/sub (lightweight) with fallback to in-memory queue.
    Projects can use this to trigger agents from external systems.
    """

    def __init__(self, redis_url: str = ""):
        self.redis_url = redis_url
        self._subscriptions: dict[str, Callable[..., Awaitable[Any]]] = {}
        self._running = False
        self._task: asyncio.Task | None = None
        self._in_memory_queue: asyncio.Queue = asyncio.Queue()
        self._message_count = 0

    def subscribe(
        self,
        channel: str,
        handler: Callable[[str, dict[str, Any]], Awaitable[Any]],
    ) -> None:
        """Subscribe to a channel."""
        self._subscriptions[channel] = handler
        logger.info(f"Subscribed to channel: {channel}")

    async def publish(self, channel: str, data: dict[str, Any]) -> None:
        """Publish a message to a channel (in-memory fallback)."""
        if self.redis_url:
            try:
                import redis.asyncio as aioredis
                r = aioredis.from_url(self.redis_url)
                import json
                await r.publish(channel, json.dumps(data))
                await r.aclose()
                return
            except ImportError:
                pass

        # In-memory fallback
        await self._in_memory_queue.put((channel, data))

    async def start(self) -> None:
        """Start listening."""
        if self._running:
            return
        self._running = True

        if self.redis_url:
            try:
                self._task = asyncio.create_task(self._redis_loop())
                logger.info("Message queue watcher started (Redis)")
                return
            except Exception:
                pass

        self._task = asyncio.create_task(self._memory_loop())
        logger.info("Message queue watcher started (in-memory)")

    async def stop(self) -> None:
        """Stop listening."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _redis_loop(self) -> None:
        """Listen to Redis pub/sub."""
        import json
        try:
            import redis.asyncio as aioredis
            r = aioredis.from_url(self.redis_url)
            pubsub = r.pubsub()

            for channel in self._subscriptions:
                await pubsub.subscribe(channel)

            while self._running:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message and message["type"] == "message":
                    channel = message["channel"]
                    if isinstance(channel, bytes):
                        channel = channel.decode()
                    data = json.loads(message["data"])
                    handler = self._subscriptions.get(channel)
                    if handler:
                        self._message_count += 1
                        try:
                            await handler(channel, data)
                        except Exception as e:
                            logger.error(f"Message handler error on '{channel}': {e}")

            await pubsub.unsubscribe()
            await r.aclose()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Redis watcher error: {e}")

    async def _memory_loop(self) -> None:
        """Process in-memory queue."""
        while self._running:
            try:
                channel, data = await asyncio.wait_for(
                    self._in_memory_queue.get(), timeout=1.0
                )
                handler = self._subscriptions.get(channel)
                if handler:
                    self._message_count += 1
                    await handler(channel, data)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Memory queue error: {e}")

    def stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "subscriptions": list(self._subscriptions.keys()),
            "messages_processed": self._message_count,
            "backend": "redis" if self.redis_url else "in-memory",
        }

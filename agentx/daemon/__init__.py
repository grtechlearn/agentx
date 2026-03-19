"""
AgentX - Autonomous Daemon System.

Run AgentX as a 24/7 self-healing autonomous agent.

Components:
- AgentXDaemon: Main daemon runner with graceful shutdown
- DaemonConfig: Configuration for daemon mode
- JobScheduler: Cron-like job scheduling
- AgentXServer: HTTP/WebSocket API server
- FileWatcher: File system change monitoring
- MessageQueueWatcher: Redis pub/sub integration
"""

from .runner import AgentXDaemon, DaemonConfig, run_daemon
from .scheduler import JobScheduler, JobConfig, JobType, JobStatus, Job, JobRun
from .server import AgentXServer, WebhookHandler, APIResponse
from .watcher import FileWatcher, FileEvent, FileEventType, MessageQueueWatcher, WatchConfig

__all__ = [
    # Runner
    "AgentXDaemon", "DaemonConfig", "run_daemon",
    # Scheduler
    "JobScheduler", "JobConfig", "JobType", "JobStatus", "Job", "JobRun",
    # Server
    "AgentXServer", "WebhookHandler", "APIResponse",
    # Watcher
    "FileWatcher", "FileEvent", "FileEventType", "MessageQueueWatcher", "WatchConfig",
]

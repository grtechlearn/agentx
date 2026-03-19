"""
AgentX Daemon — CLI Entry Point.

Run as:
    python -m agentx.daemon
    python -m agentx.daemon --port 8080
    python -m agentx.daemon --config config.yaml
    python -m agentx.daemon --no-server --scheduler-only

The daemon is a SEPARATE plugin layer — it wraps the normal AgentX app.
If you don't need 24/7, just use AgentXApp directly (normal agents).
If you need 24/7 autonomous operation, wrap with AgentXDaemon.
"""

import argparse
import asyncio
import os
import sys

from .runner import AgentXDaemon, DaemonConfig, run_daemon
from ..app import AgentXApp
from ..config import AgentXConfig


def main():
    parser = argparse.ArgumentParser(
        description="AgentX Autonomous Daemon — Run agents 24/7",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with defaults (API server on :8080 + scheduler)
  python -m agentx.daemon

  # Custom port
  python -m agentx.daemon --port 9000

  # Server only, no scheduler
  python -m agentx.daemon --no-scheduler

  # Full autonomous mode
  python -m agentx.daemon --full

  # From environment variables
  AGENTX_PORT=8080 AGENTX_API_KEY=secret python -m agentx.daemon

  # With config file
  python -m agentx.daemon --config config.yaml
        """,
    )

    parser.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument("--api-key", default="", help="API key for authentication")
    parser.add_argument("--no-server", action="store_true", help="Disable HTTP server")
    parser.add_argument("--no-scheduler", action="store_true", help="Disable job scheduler")
    parser.add_argument("--no-watchdog", action="store_true", help="Disable watchdog")
    parser.add_argument("--full", action="store_true", help="Enable all features")
    parser.add_argument("--log-level", default="INFO", help="Log level (default: INFO)")
    parser.add_argument("--log-file", default="", help="Log file path")
    parser.add_argument("--pid-file", default="", help="PID file path")
    parser.add_argument("--config", default="", help="Config file (YAML/JSON)")
    parser.add_argument("--env", action="store_true", help="Load config from environment")

    args = parser.parse_args()

    # Build daemon config
    if args.env:
        daemon_config = DaemonConfig.from_env()
    elif args.full:
        daemon_config = DaemonConfig.full(port=args.port)
    else:
        daemon_config = DaemonConfig(
            server_enabled=not args.no_server,
            server_host=args.host,
            server_port=args.port,
            server_api_key=args.api_key,
            scheduler_enabled=not args.no_scheduler,
            watchdog_enabled=not args.no_watchdog,
            log_level=args.log_level,
            log_file=args.log_file,
            pid_file=args.pid_file,
        )

    # Build app config
    if args.config:
        # Load from file
        app_config = _load_config_file(args.config)
    else:
        app_config = AgentXConfig.from_env()

    run_daemon(app_config=app_config, daemon_config=daemon_config)


def _load_config_file(path: str) -> AgentXConfig:
    """Load config from YAML or JSON file."""
    if not os.path.exists(path):
        print(f"Config file not found: {path}", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        content = f.read()

    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
            data = yaml.safe_load(content)
        except ImportError:
            print("PyYAML not installed. pip install pyyaml", file=sys.stderr)
            sys.exit(1)
    else:
        import json
        data = json.loads(content)

    return AgentXConfig(**data)


if __name__ == "__main__":
    main()

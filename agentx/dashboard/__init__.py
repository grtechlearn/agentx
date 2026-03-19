"""AgentX Dashboard — lightweight admin UI served by the daemon API server."""

from __future__ import annotations

import os


def get_dashboard_path() -> str:
    """Return the absolute path to the dashboard index.html file."""
    return os.path.join(os.path.dirname(__file__), "index.html")

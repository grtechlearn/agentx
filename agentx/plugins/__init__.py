"""
AgentX - Plugin System.

Extensible plugin architecture for third-party integrations.
Plugins can add agents, tools, middleware, hooks, and custom API endpoints.

Usage:
    # Create a plugin
    class MyPlugin(AgentXPlugin):
        name = "my-plugin"
        version = "1.0.0"

        async def setup(self, app: AgentXApp) -> None:
            app.orchestrator.register(MyCustomAgent(...))

        async def teardown(self, app: AgentXApp) -> None:
            pass

    # Register it
    app = AgentXApp()
    app_plugins = PluginManager(app)
    app_plugins.register(MyPlugin())
    await app_plugins.setup_all()

    # Or auto-discover from installed packages
    app_plugins.discover()   # Finds plugins via entry_points
"""

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("agentx.plugins")


class PluginMeta(BaseModel):
    """Plugin metadata."""

    name: str = ""
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    homepage: str = ""
    requires: list[str] = Field(default_factory=list)  # Required plugins
    tags: list[str] = Field(default_factory=list)


class AgentXPlugin(ABC):
    """
    Base class for AgentX plugins.

    A plugin can:
    - Register agents, tools, and middleware
    - Add API endpoints (daemon mode)
    - Add scheduled jobs (daemon mode)
    - Hook into the agent lifecycle
    - Provide configuration extensions
    """

    name: str = ""
    version: str = "0.1.0"
    description: str = ""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._enabled = True

    @property
    def meta(self) -> PluginMeta:
        return PluginMeta(
            name=self.name,
            version=self.version,
            description=self.description,
        )

    @abstractmethod
    async def setup(self, app: Any) -> None:
        """
        Called when the plugin is loaded.
        Use this to register agents, tools, hooks, etc.

        Args:
            app: The AgentXApp instance
        """

    async def teardown(self, app: Any) -> None:
        """Called when the plugin is unloaded. Override for cleanup."""

    async def on_message(self, message: Any, context: Any) -> Any:
        """
        Middleware hook — called before each message is dispatched.
        Return the message (possibly modified) or None to block.
        """
        return message

    async def on_response(self, message: Any, response: Any, context: Any) -> Any:
        """
        Middleware hook — called after each response is generated.
        Return the response (possibly modified).
        """
        return response

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a config value for this plugin."""
        return self.config.get(key, default)


class PluginManager:
    """
    Manages plugin lifecycle — registration, setup, teardown, and middleware.

    Usage:
        plugins = PluginManager(app)
        plugins.register(MyPlugin())
        plugins.register(AnotherPlugin(config={"key": "value"}))
        await plugins.setup_all()

        # During request processing
        message = await plugins.run_middleware(message, context)
    """

    def __init__(self, app: Any = None):
        self._app = app
        self._plugins: dict[str, AgentXPlugin] = {}
        self._load_order: list[str] = []
        self._initialized = False

    def register(self, plugin: AgentXPlugin) -> PluginManager:
        """Register a plugin."""
        if not plugin.name:
            plugin.name = plugin.__class__.__name__
        if plugin.name in self._plugins:
            logger.warning(f"Plugin '{plugin.name}' already registered, replacing")
        self._plugins[plugin.name] = plugin
        self._load_order.append(plugin.name)
        logger.info(f"Plugin registered: {plugin.name} v{plugin.version}")
        return self

    def unregister(self, name: str) -> bool:
        """Unregister a plugin by name."""
        if name in self._plugins:
            del self._plugins[name]
            self._load_order = [n for n in self._load_order if n != name]
            return True
        return False

    def get(self, name: str) -> AgentXPlugin | None:
        """Get a plugin by name."""
        return self._plugins.get(name)

    async def setup_all(self) -> dict[str, bool]:
        """Initialize all registered plugins. Returns {name: success}."""
        results = {}
        for name in self._load_order:
            plugin = self._plugins.get(name)
            if not plugin or not plugin._enabled:
                continue
            try:
                await plugin.setup(self._app)
                results[name] = True
                logger.info(f"Plugin '{name}' initialized")
            except Exception as e:
                results[name] = False
                logger.error(f"Plugin '{name}' setup failed: {e}")
        self._initialized = True
        return results

    async def teardown_all(self) -> None:
        """Teardown all plugins in reverse order."""
        for name in reversed(self._load_order):
            plugin = self._plugins.get(name)
            if not plugin:
                continue
            try:
                await plugin.teardown(self._app)
                logger.info(f"Plugin '{name}' torn down")
            except Exception as e:
                logger.error(f"Plugin '{name}' teardown failed: {e}")

    async def run_message_middleware(self, message: Any, context: Any) -> Any:
        """Run all plugin message middleware in order."""
        for name in self._load_order:
            plugin = self._plugins.get(name)
            if not plugin or not plugin._enabled:
                continue
            try:
                result = await plugin.on_message(message, context)
                if result is None:
                    logger.debug(f"Plugin '{name}' blocked message")
                    return None
                message = result
            except Exception as e:
                logger.error(f"Plugin '{name}' middleware error: {e}")
        return message

    async def run_response_middleware(self, message: Any, response: Any, context: Any) -> Any:
        """Run all plugin response middleware in order."""
        for name in self._load_order:
            plugin = self._plugins.get(name)
            if not plugin or not plugin._enabled:
                continue
            try:
                response = await plugin.on_response(message, response, context)
            except Exception as e:
                logger.error(f"Plugin '{name}' response middleware error: {e}")
        return response

    def discover(self, group: str = "agentx.plugins") -> list[str]:
        """
        Auto-discover plugins from installed packages via entry_points.

        Packages register plugins in their pyproject.toml:
            [project.entry-points."agentx.plugins"]
            my_plugin = "my_package:MyPlugin"
        """
        discovered = []
        try:
            if hasattr(importlib.metadata, 'entry_points'):
                eps = importlib.metadata.entry_points()
                # Python 3.12+ returns a SelectableGroups
                if hasattr(eps, 'select'):
                    plugin_eps = eps.select(group=group)
                elif isinstance(eps, dict):
                    plugin_eps = eps.get(group, [])
                else:
                    plugin_eps = [ep for ep in eps if ep.group == group]

                for ep in plugin_eps:
                    try:
                        plugin_cls = ep.load()
                        plugin = plugin_cls()
                        self.register(plugin)
                        discovered.append(plugin.name or ep.name)
                    except Exception as e:
                        logger.error(f"Failed to load plugin '{ep.name}': {e}")
        except Exception as e:
            logger.debug(f"Plugin discovery failed: {e}")

        if discovered:
            logger.info(f"Discovered {len(discovered)} plugins: {discovered}")
        return discovered

    def enable(self, name: str) -> bool:
        """Enable a plugin."""
        plugin = self._plugins.get(name)
        if plugin:
            plugin._enabled = True
            return True
        return False

    def disable(self, name: str) -> bool:
        """Disable a plugin (keeps registered but skips middleware)."""
        plugin = self._plugins.get(name)
        if plugin:
            plugin._enabled = False
            return True
        return False

    def list_plugins(self) -> list[dict[str, Any]]:
        """List all registered plugins."""
        return [
            {
                "name": p.name,
                "version": p.version,
                "description": p.description,
                "enabled": p._enabled,
            }
            for p in self._plugins.values()
        ]

    @property
    def plugin_count(self) -> int:
        return len(self._plugins)

    @property
    def is_initialized(self) -> bool:
        return self._initialized


__all__ = [
    "AgentXPlugin",
    "PluginManager",
    "PluginMeta",
]

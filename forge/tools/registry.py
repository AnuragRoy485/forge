"""Assemble the toolset from config: built-ins plus any drop-in plugins."""

from __future__ import annotations

from typing import Dict

from ..config import Config
from ..plugins import load_plugins
from .base import Tool
from .files import ListDirTool, ReadFileTool, WriteFileTool
from .notes import NotesTool
from .shell import ShellTool


def build_tools(config: Config) -> Dict[str, Tool]:
    root = config.resolved_workspace()
    root.mkdir(parents=True, exist_ok=True)
    tools = [
        ShellTool(timeout=config.shell_timeout, confirm=config.confirm_shell),
        ReadFileTool(root),
        WriteFileTool(root),
        ListDirTool(root),
        NotesTool(config.resolved_notes_file()),
    ]
    if config.enable_web:
        from .web import WebFetchTool
        tools.append(WebFetchTool())
    by_name = {t.name: t for t in tools}
    plugins, _errors = load_plugins(config.resolved_plugin_dir())
    for name, tool in plugins.items():
        by_name.setdefault(name, tool)  # built-ins keep their names
    return by_name

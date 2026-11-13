"""Assemble the toolset from config."""

from __future__ import annotations

from typing import Dict

from ..config import Config
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
    return {t.name: t for t in tools}

"""Plugin system: drop a .py file into the plugin directory and FORGE picks it up.

A plugin module either exposes TOOLS = [MyTool(), ...] or simply defines
subclasses of forge.tools.base.Tool - both are discovered at startup.
See examples/plugins/ in the repo for a ready-to-copy template.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from typing import Dict, List, Tuple

from .tools.base import Tool


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"forge_plugin_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build a module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_plugins(plugin_dir: Path) -> Tuple[Dict[str, Tool], List[str]]:
    """Discover plugin tools in plugin_dir. Returns (tools, errors).

    A broken plugin never stops FORGE - it is reported in errors and skipped.
    """
    tools: Dict[str, Tool] = {}
    errors: List[str] = []
    if not plugin_dir.is_dir():
        return tools, errors
    for path in sorted(plugin_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        try:
            module = _load_module(path)
        except Exception as exc:  # noqa: BLE001 - plugins must never crash the agent
            errors.append(f"{path.name}: failed to load: {exc}")
            continue
        found: List[Tool] = list(getattr(module, "TOOLS", []))
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if (issubclass(cls, Tool) and cls is not Tool
                    and cls.__module__ == module.__name__):
                if not any(isinstance(t, cls) for t in found):
                    found.append(cls())
        for tool in found:
            if not getattr(tool, "name", ""):
                errors.append(f"{path.name}: a {type(tool).__name__} has no name, skipped")
                continue
            tools[tool.name] = tool
    return tools, errors

"""Configuration loading for FORGE.

Order of precedence (later wins):
  built-in defaults -> forge.config.yaml -> environment variables
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

CONFIG_FILENAMES = ["forge.config.yaml", "forge.config.yml"]


@dataclass
class Config:
    """Runtime configuration for the agent."""

    # Model
    model: str = "qwen2.5:3b"
    ollama_host: str = "http://localhost:11434"
    temperature: float = 0.2
    num_ctx: int = 8192

    # Agent behaviour
    max_steps: int = 15
    verbose: bool = True

    # Tools
    workspace: str = "."            # file tools are sandboxed to this directory
    shell_timeout: int = 60         # seconds per shell command
    confirm_shell: bool = True      # ask before running each shell command
    enable_web: bool = False        # web tool stays OFF unless explicitly enabled
    notes_file: str = "~/.forge/notes.json"

    # Prompt
    system_prompt_extra: str = ""

    def resolved_workspace(self) -> Path:
        return Path(self.workspace).expanduser().resolve()

    def resolved_notes_file(self) -> Path:
        return Path(self.notes_file).expanduser()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _find_config_file(explicit: Optional[str] = None) -> Optional[Path]:
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() else None
    candidates: List[Path] = [Path.cwd() / name for name in CONFIG_FILENAMES]
    candidates += [Path.home() / ".forge" / name for name in CONFIG_FILENAMES]
    for c in candidates:
        if c.is_file():
            return c
    return None


def load_config(explicit_path: Optional[str] = None) -> Config:
    cfg = Config()

    path = _find_config_file(explicit_path)
    if path is not None:
        if yaml is None:
            raise RuntimeError(
                f"Found {path} but PyYAML is not installed. Run: pip install -r requirements.txt"
            )
        data = yaml.safe_load(path.read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a YAML mapping at the top level.")
        for key, value in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                raise ValueError(f"Unknown config key in {path}: {key!r}")

    env_map = {
        "FORGE_MODEL": ("model", str),
        "FORGE_OLLAMA_HOST": ("ollama_host", str),
        "FORGE_MAX_STEPS": ("max_steps", int),
        "FORGE_WORKSPACE": ("workspace", str),
    }
    for env_name, (attr, cast) in env_map.items():
        if env_name in os.environ:
            setattr(cfg, attr, cast(os.environ[env_name]))

    return cfg

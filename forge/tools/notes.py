"""Persistent memory: simple key/value notes the agent keeps between runs."""

from __future__ import annotations

import json
from pathlib import Path

from .base import Tool


class NotesTool(Tool):
    name = "notes"
    description = (
        "Remember and recall facts across sessions. "
        "Actions: remember (needs key, value), recall (optional key), list."
    )
    args_doc = (
        "action (string): remember | recall | list; "
        "key (string): note name; value (string): what to store (remember only)"
    )

    def __init__(self, notes_file: Path):
        self.file = notes_file

    def _load(self) -> dict:
        if self.file.is_file():
            try:
                return json.loads(self.file.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def _save(self, data: dict) -> None:
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.file.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def run(self, action: str = "list", key: str = "", value: str = "", **_) -> str:
        data = self._load()
        action = (action or "list").strip().lower()
        if action == "remember":
            if not key:
                return "Error: remember needs a key."
            data[key] = value
            self._save(data)
            return f"Remembered {key!r}."
        if action == "recall":
            if key:
                return data.get(key, f"No note named {key!r}.")
            return json.dumps(data, indent=2, ensure_ascii=False) if data else "(no notes yet)"
        if action == "list":
            return "\n".join(sorted(data)) if data else "(no notes yet)"
        return f"Error: unknown notes action {action!r}."

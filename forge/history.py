"""Conversation history persisted to disk, so FORGE remembers past sessions."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List


class ChatHistory:
    """A bounded list of (task, answer) turns stored as JSON on disk."""

    def __init__(self, path: Path, max_turns: int = 20):
        self.path = path
        self.max_turns = max_turns
        self.turns: List[Dict] = []
        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                data = []
            if isinstance(data, list):
                self.turns = [t for t in data if isinstance(t, dict) and "task" in t]

    def __len__(self) -> int:
        return len(self.turns)

    def add(self, task: str, answer: str) -> None:
        self.turns.append({"task": task, "answer": answer, "ts": int(time.time())})
        self.turns = self.turns[-(self.max_turns * 2):]  # keep the file bounded

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.turns, indent=2, ensure_ascii=False))

    def clear(self) -> None:
        self.turns = []
        if self.path.exists():
            self.path.unlink()

    def messages(self) -> List[Dict[str, str]]:
        """Replay the most recent turns as chat messages for the model."""
        msgs: List[Dict[str, str]] = []
        for turn in self.turns[-self.max_turns:]:
            msgs.append({"role": "user", "content": str(turn.get("task", ""))})
            msgs.append({"role": "assistant", "content": str(turn.get("answer", ""))})
        return msgs

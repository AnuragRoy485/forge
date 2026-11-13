"""Tool interface."""

from __future__ import annotations

from typing import Any


class Tool:
    """Base class for everything the agent can do."""

    name: str = ""
    description: str = ""
    # Human/LLM readable argument docs, e.g. 'command (string): the shell command to run'
    args_doc: str = ""

    def run(self, **kwargs: Any) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def spec(self) -> str:
        return f"- {self.name}: {self.description}\n  Arguments: {self.args_doc}"

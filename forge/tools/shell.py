"""Shell tool with guardrails."""

from __future__ import annotations

import re
import subprocess
from typing import Callable, Optional

from .base import Tool

# Things an agent should never run unattended.
BLOCKED_PATTERNS = [
    r"\brm\s+-[a-z]*r[a-z]*f[a-z]*\s+/",   # rm -rf rooted at /
    r"\bmkfs\b",
    r"\bdd\b.*\bof=/dev/",
    r":\(\)\s*{\s*:\|:&\s*}\s*;",           # fork bomb
    r"\bshutdown\b", r"\breboot\b", r"\bhalt\b",
    r"\bformat\b\s+[a-zA-Z]:",
]
MAX_OUTPUT_CHARS = 6000


class ShellTool(Tool):
    name = "shell"
    description = "Run a shell command on this machine and return its output."
    args_doc = "command (string): the command to execute"

    def __init__(self, timeout: int = 60, confirm: bool = True,
                 confirm_fn: Optional[Callable[[str], bool]] = None):
        self.timeout = timeout
        self.confirm = confirm
        self.confirm_fn = confirm_fn or self._default_confirm

    @staticmethod
    def _default_confirm(command: str) -> bool:
        answer = input(f"\n[forge] Run shell command?\n  $ {command}\n[y/N] ").strip().lower()
        return answer in ("y", "yes")

    def run(self, command: str = "", **_) -> str:
        command = (command or "").strip()
        if not command:
            return "Error: empty command."
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, command):
                return f"Refused: command matches a blocked pattern ({pattern!r})."
        if self.confirm and not self.confirm_fn(command):
            return "Skipped: the user declined to run this command."
        try:
            proc = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {self.timeout}s."
        out = ""
        if proc.stdout:
            out += proc.stdout
        if proc.stderr:
            out += ("\n[stderr]\n" if out else "[stderr]\n") + proc.stderr
        if not out:
            out = "(no output)"
        out = f"[exit code {proc.returncode}]\n" + out
        if len(out) > MAX_OUTPUT_CHARS:
            out = out[:MAX_OUTPUT_CHARS] + f"\n... (truncated, {len(out)} chars total)"
        return out

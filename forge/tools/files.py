"""File tools, sandboxed to the configured workspace directory."""

from __future__ import annotations

from pathlib import Path

from .base import Tool

MAX_READ_CHARS = 8000


def _resolve(root: Path, path: str) -> Path:
    target = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if root not in target.parents and target != root:
        raise PermissionError(f"Path {path!r} is outside the workspace {root}.")
    return target


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read the contents of a text file in the workspace."
    args_doc = "path (string): file path relative to the workspace"

    def __init__(self, root: Path):
        self.root = root

    def run(self, path: str = "", **_) -> str:
        try:
            target = _resolve(self.root, path)
            if not target.is_file():
                return f"Error: {path!r} is not a file."
            text = target.read_text(errors="replace")
        except (PermissionError, OSError) as exc:
            return f"Error: {exc}"
        if len(text) > MAX_READ_CHARS:
            text = text[:MAX_READ_CHARS] + f"\n... (truncated, {len(text)} chars total)"
        return text


class WriteFileTool(Tool):
    name = "write_file"
    description = "Write text to a file in the workspace, creating folders as needed."
    args_doc = "path (string): file path relative to the workspace; content (string): full file content"

    def __init__(self, root: Path):
        self.root = root

    def run(self, path: str = "", content: str = "", **_) -> str:
        try:
            target = _resolve(self.root, path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        except (PermissionError, OSError) as exc:
            return f"Error: {exc}"
        return f"Wrote {len(content)} chars to {path}."


class ListDirTool(Tool):
    name = "list_dir"
    description = "List files and folders in a workspace directory."
    args_doc = "path (string, optional): directory relative to the workspace, default '.'"

    def __init__(self, root: Path):
        self.root = root

    def run(self, path: str = ".", **_) -> str:
        try:
            target = _resolve(self.root, path or ".")
            if not target.is_dir():
                return f"Error: {path!r} is not a directory."
            entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except (PermissionError, OSError) as exc:
            return f"Error: {exc}"
        lines = [("d" if e.is_dir() else "f") + "  " + e.name for e in entries]
        return "\n".join(lines) if lines else "(empty directory)"

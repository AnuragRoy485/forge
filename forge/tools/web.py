"""Optional web fetch. OFF unless enable_web: true - FORGE is offline by default."""

from __future__ import annotations

from .base import Tool

MAX_CHARS = 6000


class WebFetchTool(Tool):
    name = "web_fetch"
    description = "Fetch a URL and return its text. Only available when enable_web is true."
    args_doc = "url (string): the page to fetch"

    def run(self, url: str = "", **_) -> str:
        if not url:
            return "Error: empty url."
        try:
            import requests
            resp = requests.get(url, timeout=20, headers={"User-Agent": "forge/1.0"})
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - surface any fetch failure to the agent
            return f"Error fetching {url}: {exc}"
        text = resp.text
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS] + f"\n... (truncated, {len(resp.text)} chars total)"
        return text

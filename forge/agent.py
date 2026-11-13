"""The FORGE agent loop: think -> act -> observe, until the task is done."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .config import Config
from .llm import OllamaClient
from .tools.base import Tool
from .tools.registry import build_tools

SYSTEM_TEMPLATE = """You are FORGE (Fully Offline Reasoning & Generative Engine), an AI agent running \
entirely on the user's own machine. You complete tasks by thinking step by step and using tools.

AVAILABLE TOOLS:
{tool_specs}

RESPONSE FORMAT - reply with exactly ONE JSON object and nothing else:
{{"thought": "your reasoning about the next step", "action": "<tool name>", "input": {{...}}}}

When the task is complete (or you are answering a question that needs no tools):
{{"thought": "why you are done", "action": "final", "input": {{"answer": "your full answer to the user"}}}}

RULES:
- One action per reply. Never invent tool results; wait for the observation.
- Prefer reading files before changing them.
- Keep thoughts short. Put real content in the input.
- File paths are relative to the workspace: {workspace}
- You have NO internet access unless a web tool is listed above.
- If a tool fails, adapt; do not repeat the identical failing call.
{extra}\
"""

JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class ActionParseError(ValueError):
    pass


def parse_action(text: str) -> dict:
    """Extract the single JSON action object from a model reply."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    match = JSON_BLOCK.search(cleaned)
    if not match:
        raise ActionParseError("no JSON object found in reply")
    try:
        action = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ActionParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(action, dict) or "action" not in action:
        raise ActionParseError("JSON object must contain an \"action\" key")
    action.setdefault("input", {})
    if not isinstance(action["input"], dict):
        action["input"] = {"value": action["input"]}
    return action


@dataclass
class Step:
    number: int
    thought: str
    action: str
    observation: str


class Agent:
    def __init__(self, config: Config, llm: Optional[OllamaClient] = None,
                 tools: Optional[Dict[str, Tool]] = None,
                 on_step: Optional[Callable[[Step], None]] = None):
        self.config = config
        self.llm = llm or OllamaClient(
            host=config.ollama_host, model=config.model,
            temperature=config.temperature, num_ctx=config.num_ctx,
        )
        self.tools = tools if tools is not None else build_tools(config)
        self.on_step = on_step
        self.steps: List[Step] = []

    def system_prompt(self) -> str:
        specs = "\n".join(t.spec() for t in self.tools.values())
        return SYSTEM_TEMPLATE.format(
            tool_specs=specs,
            workspace=self.config.resolved_workspace(),
            extra=(self.config.system_prompt_extra + "\n") if self.config.system_prompt_extra else "",
        )

    def run(self, task: str) -> str:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": task},
        ]
        for n in range(1, self.config.max_steps + 1):
            reply = self.llm.chat(messages)
            messages.append({"role": "assistant", "content": reply})
            try:
                action = parse_action(reply)
            except ActionParseError as exc:
                messages.append({
                    "role": "user",
                    "content": (
                        f"Format error: {exc}. Reply with ONE JSON object only, e.g. "
                        '{"thought": "...", "action": "final", "input": {"answer": "..."}}'
                    ),
                })
                continue

            name = str(action["action"])
            thought = str(action.get("thought", ""))

            if name == "final":
                answer = str(action["input"].get("answer", "")).strip()
                return answer or "(no answer)"

            tool = self.tools.get(name)
            if tool is None:
                observation = (
                    f"Error: unknown tool {name!r}. "
                    f"Available: {', '.join(self.tools)} and final."
                )
            else:
                try:
                    observation = tool.run(**action["input"])
                except TypeError as exc:
                    observation = f"Error: bad arguments for {name}: {exc}"
                except Exception as exc:  # noqa: BLE001 - tools must never crash the loop
                    observation = f"Error: {name} failed: {exc}"

            step = Step(number=n, thought=thought, action=name, observation=observation)
            self.steps.append(step)
            if self.on_step:
                self.on_step(step)
            messages.append({
                "role": "user",
                "content": f"OBSERVATION ({name}):\n{observation}",
            })

        return (
            f"Reached the step limit ({self.config.max_steps}) without finishing. "
            "Last observation:\n" + (self.steps[-1].observation if self.steps else "(none)")
        )

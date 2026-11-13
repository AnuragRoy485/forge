"""The FORGE agent loop: think -> act -> observe, until the task is done.

Optionally plans first (subtask decomposition), remembers past conversations
(history), and streams model output token by token.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .config import Config
from .history import ChatHistory
from .llm import OllamaClient
from .planner import PLAN_TEMPLATE, SYNTH_TEMPLATE, parse_plan
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
                 history: Optional[ChatHistory] = None,
                 on_step: Optional[Callable[[Step], None]] = None,
                 on_plan: Optional[Callable[[List[str]], None]] = None,
                 on_token: Optional[Callable[[str], None]] = None):
        self.config = config
        self.llm = llm or OllamaClient(
            host=config.ollama_host, model=config.model,
            temperature=config.temperature, num_ctx=config.num_ctx,
        )
        self.tools = tools if tools is not None else build_tools(config)
        self.history = history
        self.on_step = on_step
        self.on_plan = on_plan
        self.on_token = on_token
        self.steps: List[Step] = []

    def system_prompt(self) -> str:
        specs = "\n".join(t.spec() for t in self.tools.values())
        return SYSTEM_TEMPLATE.format(
            tool_specs=specs,
            workspace=self.config.resolved_workspace(),
            extra=(self.config.system_prompt_extra + "\n") if self.config.system_prompt_extra else "",
        )

    def _chat(self, messages: List[Dict[str, str]]) -> str:
        """One model call - streamed when a token callback is set."""
        if self.on_token and hasattr(self.llm, "chat_stream"):
            reply = self.llm.chat_stream(messages, on_token=self.on_token)
            self.on_token("\n")  # end the streamed line cleanly
            return reply
        return self.llm.chat(messages)

    def _record(self, task: str, answer: str) -> None:
        if self.history is not None:
            self.history.add(task, answer)
            self.history.save()

    # -- planning -------------------------------------------------------

    def _make_plan(self, task: str) -> List[str]:
        specs = "\n".join(t.spec() for t in self.tools.values())
        prompt = PLAN_TEMPLATE.format(
            tool_specs=specs, max_subtasks=self.config.max_subtasks, task=task,
        )
        reply = self._chat([{"role": "user", "content": prompt}])
        return parse_plan(reply, self.config.max_subtasks)

    def _run_plan(self, task: str, subtasks: List[str]) -> str:
        if self.on_plan:
            self.on_plan(subtasks)
        results: List[str] = []
        for i, sub in enumerate(subtasks, 1):
            label = (
                f"This is subtask {i} of {len(subtasks)} of a larger task.\n"
                f"Overall task: {task}\n"
                f"Your subtask: {sub}"
            )
            answer = self._run_loop(label, record=False)  # subtasks stay out of history
            results.append(f"{i}. {sub}\n   Result: {answer}")
        reply = self._chat([{
            "role": "user",
            "content": SYNTH_TEMPLATE.format(task=task, results="\n".join(results)),
        }])
        try:
            action = parse_action(reply)
            answer = str(action["input"].get("answer", "")).strip()
            if action["action"] == "final" and answer:
                self._record(task, answer)
                return answer
        except ActionParseError:
            pass
        answer = reply or "(no answer)"
        self._record(task, answer)
        return answer

    # -- main entry -----------------------------------------------------

    def run(self, task: str, plan: Optional[bool] = None) -> str:
        use_plan = self.config.auto_plan if plan is None else plan
        if use_plan:
            subtasks = self._make_plan(task)
            if len(subtasks) > 1:
                return self._run_plan(task, subtasks)
        return self._run_loop(task)

    def _run_loop(self, task: str, record: bool = True) -> str:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt()},
        ]
        if self.history is not None:
            messages.extend(self.history.messages())
        messages.append({"role": "user", "content": task})
        for n in range(1, self.config.max_steps + 1):
            reply = self._chat(messages)
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
                answer = answer or "(no answer)"
                if record:
                    self._record(task, answer)
                return answer

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

        result = (
            f"Reached the step limit ({self.config.max_steps}) without finishing. "
            "Last observation:\n" + (self.steps[-1].observation if self.steps else "(none)")
        )
        if record:
            self._record(task, result)
        return result

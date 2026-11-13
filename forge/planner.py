"""Multi-step planning: decompose a task into subtasks, then work through them."""

from __future__ import annotations

import json
import re
from typing import List

PLAN_TEMPLATE = """You are the planning brain of FORGE, an offline AI agent. Break the user's task into \
a short ordered list of subtasks that an agent with these tools can complete one by one.

AVAILABLE TOOLS:
{tool_specs}

RULES:
- 1 to {max_subtasks} subtasks, ordered, each self-contained and concrete.
- A simple question or a single-step task is ONE subtask.
- Reply with exactly ONE JSON object and nothing else:
{{"subtasks": ["first step", "second step"]}}

TASK: {task}"""

SYNTH_TEMPLATE = """You are FORGE, an offline AI agent. The user's task was broken into subtasks \
and each one was completed by your tool-using self.

TASK: {task}

SUBTASK RESULTS:
{results}

Reply with exactly ONE JSON object and nothing else:
{{"thought": "why the task is complete", "action": "final", "input": {{"answer": "the full final answer to the user, summarizing what was done"}}}}"""

JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_plan(text: str, max_subtasks: int) -> List[str]:
    """Extract the subtask list from a planning reply. Empty list = no usable plan."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    match = JSON_BLOCK.search(cleaned)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    subtasks = data.get("subtasks")
    if not isinstance(subtasks, list):
        return []
    return [str(s).strip() for s in subtasks if str(s).strip()][:max_subtasks]

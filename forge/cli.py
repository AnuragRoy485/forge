"""Command line interface for FORGE."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__, __name_full__
from .agent import Agent, Step
from .config import load_config
from .llm import OllamaError

BANNER = r"""
  _____ ___  ____   ____ _____
 |  ___/ _ \|  _ \ / ___| ____|
 | |_ | | | | |_) | |  _|  _|
 |  _|| |_| |  _ <| |_| | |___
 |_|   \___/|_| \_\\____|_____|
"""


def _print_step(step: Step) -> None:
    thought = step.thought.strip()
    if thought:
        print(f"\n  [{step.number}] thought : {thought}")
    obs = step.observation.strip()
    if len(obs) > 500:
        obs = obs[:500] + "\n  ... (truncated)"
    print(f"  [{step.number}] {step.action:<8}: {obs}")


def _make_agent(cfg, quiet: bool) -> Agent:
    return Agent(cfg, on_step=None if quiet else _print_step)


def cmd_run(args) -> int:
    cfg = load_config(args.config)
    cfg.confirm_shell = not args.yes
    agent = _make_agent(cfg, args.quiet)
    try:
        agent.llm.check_ready()
    except OllamaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    answer = agent.run(args.task)
    print(f"\n== FORGE ==\n{answer}")
    return 0


def cmd_chat(args) -> int:
    cfg = load_config(args.config)
    cfg.confirm_shell = not args.yes
    agent = _make_agent(cfg, args.quiet)
    try:
        agent.llm.check_ready()
    except OllamaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(BANNER)
    print(f"FORGE {__version__} - {__name_full__}")
    print(f"model: {cfg.model} | workspace: {cfg.resolved_workspace()}")
    print("Type a task, or 'exit' to quit.\n")
    while True:
        try:
            task = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not task:
            continue
        if task.lower() in ("exit", "quit"):
            break
        agent.steps.clear()
        answer = agent.run(task)
        print(f"\nforge> {answer}\n")
    return 0


def cmd_tools(args) -> int:
    cfg = load_config(args.config)
    agent = _make_agent(cfg, quiet=True)
    print("Available tools:\n")
    for tool in agent.tools.values():
        print(tool.spec())
    print("- final: finish the task and return the answer")
    return 0


def cmd_config(args) -> int:
    cfg = load_config(args.config)
    print(json.dumps(cfg.to_dict(), indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="forge",
        description=f"FORGE - {__name_full__}. A fully offline AI agent powered by a local LLM.",
    )
    parser.add_argument("--version", action="version", version=f"forge {__version__}")
    parser.add_argument("--config", help="path to a forge.config.yaml", default=None)
    sub = parser.add_subparsers(dest="command")

    def add_common(p):
        p.add_argument("--yes", "-y", action="store_true",
                       help="run shell commands without asking each time")
        p.add_argument("--quiet", "-q", action="store_true",
                       help="hide intermediate steps, print only the final answer")

    p_run = sub.add_parser("run", help="run a single task and exit")
    p_run.add_argument("task", help="what you want FORGE to do")
    add_common(p_run)
    p_run.set_defaults(func=cmd_run)

    p_chat = sub.add_parser("chat", help="interactive chat session")
    add_common(p_chat)
    p_chat.set_defaults(func=cmd_chat)

    p_tools = sub.add_parser("tools", help="list the tools FORGE can use")
    p_tools.set_defaults(func=cmd_tools)

    p_cfg = sub.add_parser("config", help="print the effective configuration")
    p_cfg.set_defaults(func=cmd_config)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

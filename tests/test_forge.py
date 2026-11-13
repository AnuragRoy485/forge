"""Offline unit tests - no Ollama needed, a scripted fake LLM stands in."""

import tempfile
import unittest
from pathlib import Path

from forge.agent import Agent, parse_action, ActionParseError
from forge.config import Config
from forge.tools.files import ReadFileTool, WriteFileTool
from forge.tools.notes import NotesTool
from forge.tools.shell import ShellTool


class FakeLLM:
    """Plays back scripted replies, then answers final."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.model = "fake"

    def check_ready(self):
        pass

    def chat(self, messages):
        if self.replies:
            return self.replies.pop(0)
        return '{"thought": "done", "action": "final", "input": {"answer": "All done."}}'


class TestParseAction(unittest.TestCase):
    def test_plain_json(self):
        a = parse_action('{"thought": "t", "action": "shell", "input": {"command": "ls"}}')
        self.assertEqual(a["action"], "shell")
        self.assertEqual(a["input"]["command"], "ls")

    def test_fenced_json(self):
        a = parse_action('```json\n{"action": "final", "input": {"answer": "hi"}}\n```')
        self.assertEqual(a["action"], "final")

    def test_surrounding_text(self):
        a = parse_action('Sure! Here you go:\n{"action": "list_dir", "input": {"path": "."}}\nHope that helps')
        self.assertEqual(a["action"], "list_dir")

    def test_bad_input_raises(self):
        with self.assertRaises(ActionParseError):
            parse_action("no json here at all")


class TestTools(unittest.TestCase):
    def test_write_then_read(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            w, r = WriteFileTool(root), ReadFileTool(root)
            self.assertIn("Wrote", w.run(path="a/b.txt", content="hello"))
            self.assertEqual(r.run(path="a/b.txt"), "hello")

    def test_sandbox_escape_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            r = ReadFileTool(Path(d))
            self.assertIn("outside the workspace", r.run(path="../../etc/passwd"))

    def test_shell_blocked_pattern(self):
        sh = ShellTool(confirm=False)
        self.assertIn("Refused", sh.run(command="rm -rf /"))

    def test_shell_runs(self):
        sh = ShellTool(confirm=False)
        out = sh.run(command="echo forge-test")
        self.assertIn("forge-test", out)
        self.assertIn("exit code 0", out)

    def test_notes_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            n = NotesTool(Path(d) / "notes.json")
            n.run(action="remember", key="name", value="forge")
            self.assertEqual(n.run(action="recall", key="name"), "forge")
            self.assertIn("name", n.run(action="list"))


class TestAgentLoop(unittest.TestCase):
    def test_full_loop_with_tools(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Config(workspace=d, confirm_shell=False, max_steps=5)
            llm = FakeLLM([
                '{"thought": "save it", "action": "write_file", "input": {"path": "x.txt", "content": "42"}}',
                '{"thought": "check", "action": "read_file", "input": {"path": "x.txt"}}',
            ])
            agent = Agent(cfg, llm=llm)
            answer = agent.run("save 42 in x.txt and read it back")
            self.assertEqual(answer, "All done.")
            self.assertEqual(len(agent.steps), 2)
            self.assertEqual((Path(d) / "x.txt").read_text(), "42")

    def test_format_error_recovers(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Config(workspace=d, confirm_shell=False, max_steps=4)
            llm = FakeLLM(["I don't know what JSON is"])
            agent = Agent(cfg, llm=llm)
            self.assertEqual(agent.run("say hi"), "All done.")

    def test_step_limit(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Config(workspace=d, confirm_shell=False, max_steps=2)
            llm = FakeLLM([
                '{"thought": "loop", "action": "list_dir", "input": {}}',
                '{"thought": "loop", "action": "list_dir", "input": {}}',
                '{"thought": "loop", "action": "list_dir", "input": {}}',
            ])
            agent = Agent(cfg, llm=llm)
            self.assertIn("step limit", agent.run("loop forever"))


if __name__ == "__main__":
    unittest.main()

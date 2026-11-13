"""Offline unit tests - no Ollama needed, a scripted fake LLM stands in."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from forge.agent import Agent, parse_action, ActionParseError
from forge.config import Config
from forge.history import ChatHistory
from forge.llm import OllamaClient
from forge.planner import parse_plan
from forge.plugins import load_plugins
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

    def chat_stream(self, messages, on_token=None):
        reply = self.chat(messages)
        if on_token:
            for word in reply.split(" "):
                on_token(word + " ")
        return reply


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


class TestPlanning(unittest.TestCase):
    def test_parse_plan(self):
        self.assertEqual(parse_plan('{"subtasks": ["a", "b"]}', 6), ["a", "b"])
        self.assertEqual(parse_plan("garbage in, no plan", 6), [])
        self.assertEqual(parse_plan('{"subtasks": ["a", "b", "c"]}', 2), ["a", "b"])

    def test_planned_run(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Config(workspace=d, confirm_shell=False, max_steps=3)
            llm = FakeLLM([
                '{"subtasks": ["save the number", "read it back"]}',
                # subtask 1 loop
                '{"thought": "save", "action": "write_file", "input": {"path": "n.txt", "content": "7"}}',
                '{"thought": "done", "action": "final", "input": {"answer": "saved 7"}}',
                # subtask 2 loop
                '{"thought": "read", "action": "read_file", "input": {"path": "n.txt"}}',
                '{"thought": "done", "action": "final", "input": {"answer": "it says 7"}}',
                # synthesis
                '{"thought": "wrap up", "action": "final", "input": {"answer": "Saved 7 and read it back."}}',
            ])
            plans = []
            agent = Agent(cfg, llm=llm, on_plan=plans.append)
            answer = agent.run("save 7 in n.txt and read it back", plan=True)
            self.assertEqual(answer, "Saved 7 and read it back.")
            self.assertEqual((Path(d) / "n.txt").read_text(), "7")
            self.assertEqual(plans[0], ["save the number", "read it back"])

    def test_single_subtask_falls_back_to_plain_loop(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Config(workspace=d, confirm_shell=False)
            llm = FakeLLM(['{"subtasks": ["just answer"]}'])
            agent = Agent(cfg, llm=llm)
            self.assertEqual(agent.run("what is 2+2", plan=True), "All done.")


class TestHistory(unittest.TestCase):
    def test_history_persists_to_disk(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "h.json"
            h = ChatHistory(path, max_turns=10)
            h.add("hello", "hi there")
            h.save()
            h2 = ChatHistory(path, max_turns=10)
            self.assertEqual(len(h2), 1)
            msgs = h2.messages()
            self.assertEqual(msgs[0], {"role": "user", "content": "hello"})
            self.assertEqual(msgs[1], {"role": "assistant", "content": "hi there"})
            h2.clear()
            self.assertFalse(path.exists())

    def test_agent_records_and_replays_history(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Config(workspace=d, confirm_shell=False)
            llm = FakeLLM([])
            hist = ChatHistory(Path(d) / "h.json", max_turns=10)
            agent = Agent(cfg, llm=llm, history=hist)
            agent.run("first question")
            agent.run("second question")
            reloaded = ChatHistory(Path(d) / "h.json", max_turns=10)
            self.assertEqual(len(reloaded), 2)
            self.assertEqual(reloaded.turns[0]["task"], "first question")
            self.assertEqual(reloaded.turns[1]["answer"], "All done.")


PLUGIN_SRC = '''from forge.tools.base import Tool

class ShoutTool(Tool):
    name = "shout"
    description = "Uppercase a message."
    args_doc = "text (string): what to shout"

    def run(self, text: str = "", **_):
        return text.upper()
'''


class TestPlugins(unittest.TestCase):
    def test_plugin_discovery(self):
        with tempfile.TemporaryDirectory() as d:
            pdir = Path(d) / "plugins"
            pdir.mkdir()
            (pdir / "shout.py").write_text(PLUGIN_SRC)
            tools, errors = load_plugins(pdir)
            self.assertFalse(errors)
            self.assertIn("shout", tools)
            self.assertEqual(tools["shout"].run(text="hi"), "HI")

    def test_plugin_usable_by_agent(self):
        with tempfile.TemporaryDirectory() as d:
            pdir = Path(d) / "plugins"
            pdir.mkdir()
            (pdir / "shout.py").write_text(PLUGIN_SRC)
            cfg = Config(workspace=d, confirm_shell=False, plugin_dir=str(pdir))
            llm = FakeLLM([
                '{"thought": "shout", "action": "shout", "input": {"text": "forge"}}',
            ])
            agent = Agent(cfg, llm=llm)
            self.assertIn("shout", agent.tools)
            agent.run("shout forge")
            self.assertEqual(agent.steps[0].observation, "FORGE")

    def test_broken_plugin_never_stops_forge(self):
        with tempfile.TemporaryDirectory() as d:
            pdir = Path(d) / "plugins"
            pdir.mkdir()
            (pdir / "broken.py").write_text("raise RuntimeError('boom')")
            tools, errors = load_plugins(pdir)
            self.assertEqual(tools, {})
            self.assertEqual(len(errors), 1)


class TestStreaming(unittest.TestCase):
    def test_agent_streams_tokens(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Config(workspace=d, confirm_shell=False)
            llm = FakeLLM([])
            seen = []
            agent = Agent(cfg, llm=llm, on_token=seen.append)
            self.assertEqual(agent.run("hi"), "All done.")
            self.assertTrue(seen)
            self.assertIn("All done", "".join(seen))

    def test_ollama_chat_stream_parses_lines(self):
        lines = [
            '{"message": {"content": "Hel"}, "done": false}',
            '{"message": {"content": "lo"}, "done": false}',
            '{"message": {"content": ""}, "done": true}',
        ]

        class FakeResp:
            def raise_for_status(self):
                pass

            def iter_lines(self, decode_unicode=False):
                return iter(lines)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        client = OllamaClient()
        with mock.patch("requests.post", return_value=FakeResp()):
            got = []
            text = client.chat_stream([{"role": "user", "content": "hi"}], on_token=got.append)
        self.assertEqual(text, "Hello")
        self.assertEqual("".join(got), "Hello")


if __name__ == "__main__":
    unittest.main()

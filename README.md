```
  _____ ___  ____   ____ _____
 |  ___/ _ \|  _ \ / ___| ____|
 | |_ | | | | |_) | |  _|  _|
 |  _|| |_| |  _ <| |_| | |___
 |_|   \___/|_| \_\\____|_____|
```

# FORGE

**Fully Offline Reasoning & Generative Engine** — a self-reliant AI agent that runs 100% on your machine. No cloud APIs, no API keys, no subscriptions. Your data never leaves your computer.

FORGE pairs a local LLM (served by [Ollama](https://ollama.com)) with an agent loop that can **plan, use tools, and finish real tasks**: run shell commands, read and write files, keep persistent notes — all offline.

## Features

- **Fully offline** — every inference runs locally via Ollama. Works on a plane.
- **Agentic loop** — think → act → observe, repeat, until the task is done.
- **Multi-step planning** — `--plan` breaks a big task into subtasks and works through them in order, then synthesizes one answer.
- **Conversation history** — chats are saved to disk and restored across sessions, so FORGE remembers what you talked about yesterday.
- **Plugin system** — drop a `.py` file into your plugin directory and your custom tool is live. No restarts, no boilerplate.
- **Streaming output** — watch tokens appear as the model generates them.
- **Real tools** — shell, file read/write/list, persistent memory (`notes`), optional web fetch (off by default).
- **Safety first** — file access sandboxed to your workspace, dangerous commands blocked, shell confirmation before anything runs.
- **Any Ollama model** — `qwen2.5`, `llama3.1`, `mistral`, `phi3`... one line in the config.

## How it works

```
        you give FORGE a task
                |
        (optional) planner: task -> subtask 1..n
                |
                v
   +---------------------------+
   |   local LLM (Ollama)      |<----- runs on localhost only, streams tokens
   +---------------------------+
                |  {"thought": ..., "action": ..., "input": ...}
                v
   +---------------------------+
   |   tool runner             |  shell | files | notes | plugins | (web)
   +---------------------------+
                |  observation
                +------> loops back until "final"
```

## Install

Requires Python 3.9+.

```bash
# 1. Install Ollama (https://ollama.com), then pull a model
ollama pull qwen2.5:3b

# 2. Get FORGE
git clone https://github.com/AnuragRoy485/forge.git
cd forge
pip install -r requirements.txt

# 3. Run it
python -m forge run "create a file hello.txt that says hello from FORGE"
```

Optional: install as a package to get the `forge` command:

```bash
pip install -e .
forge run "list the files in this folder"
```

## Usage

```bash
# One-shot task
python -m forge run "summarize what files are in this directory"

# Big task? Let the planner break it into subtasks first
python -m forge run "build a small flask app with tests" --plan

# Interactive chat (remembers previous sessions)
python -m forge chat

# Start a chat with a clean slate
python -m forge chat --new

# Review or wipe saved conversation history
python -m forge history
python -m forge history --clear

# Skip the per-command shell confirmation (use with care)
python -m forge run "find all .py files and count their lines" --yes

# See the tools (built-ins + plugins) and the effective config
python -m forge tools
python -m forge config
```

## Multi-step planning

For tasks too big for one think/act chain, FORGE can plan first:

```bash
python -m forge run "research this folder and write a summary report" --plan
```

The planner asks the model to decompose the task into up to `max_subtasks` ordered subtasks, runs each through the full agent loop, then synthesizes the results into one final answer. Set `auto_plan: true` in the config to plan every task, and `--no-plan` to skip it for one run.

## Conversation history

Every task and answer is saved to `~/.forge/history.json`. The last `history_turns` turns are replayed into each prompt, so follow-ups work across sessions:

```
# session 1
you> my favourite editor is neovim
forge> Got it.

# session 2 (next day, new terminal)
you> write a config note for my editor
forge> ...writes a neovim config note...
```

Disable with `history_enabled: false`; inspect or wipe with `forge history` / `forge history --clear`. This is separate from the `notes` tool, which is the agent's own deliberate long-term memory.

## Plugins: write your own tools

Drop any `.py` file into your plugin directory (`~/.forge/plugins` by default) and FORGE discovers it at startup. A plugin subclasses `forge.tools.base.Tool`:

```python
# ~/.forge/plugins/word_count.py
from forge.tools.base import Tool

class WordCountTool(Tool):
    name = "word_count"
    description = "Count the words in a piece of text."
    args_doc = "text (string): the text to count words in"

    def run(self, text: str = "", **_):
        return f"{len(text.split())} word(s)"
```

That's it — `forge tools` lists it and the agent can use it immediately. A plugin module can also expose `TOOLS = [MyTool(), ...]`; both styles are discovered. A broken plugin never stops FORGE: it is reported on stderr and skipped. See [examples/plugins/word_count.py](examples/plugins/word_count.py).

## Streaming output

With `stream: true` (default), model tokens print as they generate — you watch FORGE think instead of waiting for a finished reply. Set `stream: false` or pass `--quiet` for silent operation.

## Configuration

Edit `forge.config.yaml` (or keep your own at `~/.forge/config.yaml`):

| Key | Default | What it does |
|---|---|---|
| `model` | `qwen2.5:3b` | Any model served by Ollama |
| `ollama_host` | `http://localhost:11434` | Where Ollama listens |
| `temperature` | `0.2` | Low values give more reliable tool use |
| `max_steps` | `15` | Safety cap on think/act cycles per task |
| `stream` | `true` | Stream model tokens to the terminal |
| `auto_plan` | `false` | Decompose every task into subtasks (or use `--plan`) |
| `max_subtasks` | `6` | Cap on planner subtasks per task |
| `history_enabled` | `true` | Save conversations and restore them across sessions |
| `history_file` | `~/.forge/history.json` | Where conversation history lives |
| `history_turns` | `20` | Past turns replayed into each prompt |
| `workspace` | `.` | File tools are sandboxed inside this directory |
| `confirm_shell` | `true` | Ask before running each shell command |
| `enable_web` | `false` | Keep off for fully offline use |
| `notes_file` | `~/.forge/notes.json` | The agent's long-term memory |
| `plugin_dir` | `~/.forge/plugins` | Drop-in `.py` tool plugins are loaded from here |

Environment overrides: `FORGE_MODEL`, `FORGE_OLLAMA_HOST`, `FORGE_MAX_STEPS`, `FORGE_WORKSPACE`.

## Example session

```
you> remember that my favourite editor is neovim, then create a note about it

  [1] thought : store the fact first
  [1] notes   : Remembered 'favourite_editor'.
  [2] thought : now write it to a file
  [2] write_file: Wrote 47 chars to editor_note.txt.

forge> Done — I remembered that your favourite editor is neovim and saved it to editor_note.txt.
```

## Safety

- File tools cannot leave the configured `workspace` directory.
- A denylist blocks destructive commands (`rm -rf /`, fork bombs, disk formatting...).
- With `confirm_shell: true`, nothing runs without your `y`.
- `enable_web` defaults to `false` — FORGE cannot touch the network unless you say so.

## Development

```bash
python -m unittest discover -s tests -v   # 22 tests, no Ollama needed
```

## Roadmap

- [x] Multi-step planning with subtask decomposition
- [x] Conversation history persisted across sessions
- [x] Plugin system for custom tools
- [x] Streaming output

## License

MIT — see [LICENSE](LICENSE).

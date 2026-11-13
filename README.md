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
- **Real tools** — shell, file read/write/list, persistent memory (`notes`), optional web fetch (off by default).
- **Safety first** — file access sandboxed to your workspace, dangerous commands blocked, shell confirmation before anything runs.
- **Any Ollama model** — `qwen2.5`, `llama3.1`, `mistral`, `phi3`... one line in the config.
- **Simple CLI** — one-shot tasks or an interactive chat session.

## How it works

```
        you give FORGE a task
                |
                v
   +---------------------------+
   |   local LLM (Ollama)      |<----- runs on localhost only
   +---------------------------+
                |  {"thought": ..., "action": ..., "input": ...}
                v
   +---------------------------+
   |   tool runner             |  shell | files | notes | (web)
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

# Interactive chat
python -m forge chat

# Skip the per-command shell confirmation (use with care)
python -m forge run "find all .py files and count their lines" --yes

# See the tools and the effective config
python -m forge tools
python -m forge config
```

## Configuration

Edit `forge.config.yaml` (or keep your own at `~/.forge/config.yaml`):

| Key | Default | What it does |
|---|---|---|
| `model` | `qwen2.5:3b` | Any model served by Ollama |
| `ollama_host` | `http://localhost:11434` | Where Ollama listens |
| `temperature` | `0.2` | Low values give more reliable tool use |
| `max_steps` | `15` | Safety cap on think/act cycles per task |
| `workspace` | `.` | File tools are sandboxed inside this directory |
| `confirm_shell` | `true` | Ask before running each shell command |
| `enable_web` | `false` | Keep off for fully offline use |
| `notes_file` | `~/.forge/notes.json` | The agent's long-term memory |

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
python -m unittest discover -s tests -v   # 12 tests, no Ollama needed
```

## Roadmap

- [ ] Multi-step planning with subtask decomposition
- [ ] Conversation history persisted across sessions
- [ ] Plugin system for custom tools
- [ ] Streaming output

## License

MIT — see [LICENSE](LICENSE).

# Hearth

**A private AI assistant for your terminal that runs entirely on your own computer.**

Hearth chats with you using an open-source model running locally through [Ollama](https://ollama.com).
It can search the web, read web pages, look at your files, write files and run shell commands, and it
always asks before changing anything. No account, no API key, and your conversations never leave
your machine (web searches go to DuckDuckGo only when the assistant decides it needs one).

```text
$ hearth what changed in the latest Python release?
> web_search(query=latest Python release what's new)
  1. What's New In Python 3.14 — Python documentation
> fetch_url(url=https://docs.python.org/3/whatsnew/3.14.html)
  Title: What's New In Python 3.14
Python 3.14 brings ... (with sources)
```

## Install

You need two things: **Ollama** (runs the model) and **Hearth** (the assistant).

**1. Install Ollama.** Download it from [ollama.com/download](https://ollama.com/download) (macOS,
Windows, Linux) and open it once so it is running in the background.

**2. Install Hearth.** The easiest way is with [uv](https://docs.astral.sh/uv/):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install git+https://github.com/OpenSourceJesus/hearth
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv tool install git+https://github.com/OpenSourceJesus/hearth
```

Already use pipx? `pipx install git+https://github.com/OpenSourceJesus/hearth` works too.
Hearth needs Python 3.10 or newer; uv will fetch one for you if needed.

**3. Run it.**

```bash
hearth
```

The first time, Hearth offers to download its default model (`qwen3:4b`, about 2.5 GB).

To update later: `uv tool upgrade hearth-agent`. To remove: `uv tool uninstall hearth-agent`.

## Usage

```bash
hearth                                   # start a conversation
hearth explain what a mutex is           # ask one question and exit
cat error.log | hearth what went wrong?  # send a file or command output along with your question
hearth -m llama3.2                       # use a different model
hearth --no-tools                        # plain chat, no tools
```

Inside a conversation:

| Command         | What it does                                  |
| --------------- | --------------------------------------------- |
| `/help`         | Show commands                                 |
| `/model [name]` | Show or switch the model (downloads if needed)|
| `/models`       | List models you have installed                |
| `/tools`        | List the assistant's tools                    |
| `/reset`        | Start a fresh conversation                    |
| `/exit`         | Quit (or press Ctrl+D)                        |

Press **Ctrl+C** to stop a reply that is in progress.

## What the assistant can do

| Tool             | Asks you first? | Description                                   |
| ---------------- | --------------- | --------------------------------------------- |
| `web_search`     | no              | Searches DuckDuckGo                           |
| `fetch_url`      | no              | Reads the text of a web page                  |
| `list_directory` | no              | Lists files in a folder                       |
| `read_file`      | no              | Reads a text file                             |
| `write_file`     | **yes**         | Creates or overwrites a file                  |
| `run_shell`      | **yes**         | Runs a command in your current folder         |

When the assistant wants to write a file or run a command, Hearth shows you exactly what it will do
and waits for `y` (yes), `n` (no) or `a` (always allow this tool for the rest of the session).
Passing `--yes` skips these prompts; only use it when you trust what you're asking for.

## Choosing a model

Any Ollama model that supports tool calling works; browse them at
[ollama.com/search?c=tools](https://ollama.com/search?c=tools). Some good starting points:

| Model         | Download | Good for                                     |
| ------------- | -------- | -------------------------------------------- |
| `qwen3:4b`    | 2.5 GB   | Default. Fast on most laptops, solid tool use|
| `qwen3:8b`    | 5.2 GB   | Smarter, needs ~8 GB of free RAM/VRAM        |
| `llama3.2:3b` | 2.0 GB   | Lightweight alternative                      |

Set your favorite permanently with an environment variable, e.g. `export HEARTH_MODEL=qwen3:8b`.

Reasoning models like Qwen3 "think" before answering. Hearth hides that reasoning behind a spinner;
pass `--show-thinking` to watch it.

## Options

| Flag / variable          | Meaning                                                   |
| ------------------------ | --------------------------------------------------------- |
| `-m`, `--model`          | Model name (default `$HEARTH_MODEL`, else `qwen3:4b`)     |
| `--host`, `$OLLAMA_HOST` | Ollama server URL, if it's not on `localhost:11434`       |
| `-y`, `--yes`            | Don't ask before writing files or running commands        |
| `--no-tools`             | Disable all tools                                         |
| `--think` / `--no-think` | Ask the model to turn reasoning on or off (not all obey)  |
| `--show-thinking`        | Print the model's reasoning                               |
| `--max-steps N`          | Max tool calls per reply (default 15)                     |

Answers are written to stdout and everything else (tool activity, prompts) to stderr, so
`hearth summarize this < notes.txt > summary.md` does what you'd expect.

## Troubleshooting

- **"Could not reach Ollama"**: open the Ollama app, or run `ollama serve` in another terminal.
- **"does not support tools"**: the model can't call tools. Choose one from the list above or use
  `--no-tools`.
- **Slow replies**: try a smaller model such as `qwen3:1.7b`. The first reply after starting is
  slower because Ollama loads the model into memory.

## Development

```bash
git clone https://github.com/OpenSourceJesus/hearth && cd hearth
uv sync
uv run pytest
uv run hearth
```

## License

MIT

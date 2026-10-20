"""Command-line interface for Hearth."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
import ollama
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TransferSpeedColumn
from rich.status import Status
from rich.syntax import Syntax

from hearth_agent import __version__
from hearth_agent.agent import Agent
from hearth_agent.tools import BUILTIN_TOOLS, Tool

DEFAULT_MODEL = "qwen3:4b"
HISTORY_FILE = Path.home() / ".hearth_history"

HELP_TEXT = """\
[bold]Commands[/]
  /help            Show this help
  /model [name]    Show the current model, or switch to another one
  /models          List models installed in Ollama
  /tools           List the tools the assistant can use
  /reset           Start a new conversation
  /exit            Quit (or press Ctrl+D)

Press Ctrl+C to stop a reply that is in progress."""


class RichUI:
    """Streams the answer to stdout; status, tool activity and prompts go to stderr."""

    def __init__(self, show_thinking: bool = False, interactive: bool = True) -> None:
        self.err = Console(stderr=True, highlight=False)
        self.show_thinking = show_thinking
        self.interactive = interactive
        self.always_allowed: set[str] = set()
        self._status: Status | None = None
        self._wrote_text = False
        self._last_char = "\n"
        self._thinking_open = False

    def _stop_status(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None

    def on_wait(self) -> None:
        self._stop_status()
        self._status = self.err.status("[dim]Thinking...[/]", spinner="dots")
        self._status.start()

    def on_thinking(self, text: str) -> None:
        if not self.show_thinking:
            return
        self._stop_status()
        if not self._thinking_open:
            self.err.print("[dim italic]thinking:[/]")
            self._thinking_open = True
        self.err.print(escape(text), style="dim italic", end="")

    def on_text(self, text: str) -> None:
        self._stop_status()
        if self._thinking_open:
            self.err.print()
            self._thinking_open = False
        if not self._wrote_text:
            text = text.lstrip("\n")
            if not text:
                return
        sys.stdout.write(text)
        sys.stdout.flush()
        self._wrote_text = True
        self._last_char = text[-1]

    def on_message_end(self) -> None:
        self._stop_status()
        if self._thinking_open:
            self.err.print()
            self._thinking_open = False
        if self._wrote_text and self._last_char != "\n":
            sys.stdout.write("\n")
            sys.stdout.flush()
        self._wrote_text = False
        self._last_char = "\n"

    def on_tool_call(self, name: str, args: dict[str, Any]) -> None:
        summary = ", ".join(f"{k}={_short(v)}" for k, v in args.items())
        self.err.print(f"[cyan]> {escape(name)}[/]([dim]{escape(summary)}[/])")

    def on_tool_result(self, name: str, result: str) -> None:
        first = result.strip().splitlines()[0] if result.strip() else "(no output)"
        style = "red" if result.startswith("Error") or "declined" in first else "dim"
        self.err.print(f"  [{style}]{escape(_short(first, 100))}[/]")

    def approve(self, tool: Tool, args: dict[str, Any]) -> bool:
        self._stop_status()
        if tool.name in self.always_allowed:
            return True
        if tool.name == "run_shell":
            body: Any = Syntax(str(args.get("command", "")), "bash", word_wrap=True)
            title = "Run this shell command?"
        elif tool.name == "write_file":
            content = str(args.get("content", ""))
            preview = content if len(content) < 2000 else content[:2000] + "\n..."
            body = f"[bold]{escape(str(args.get('path', '')))}[/]\n\n{escape(preview)}"
            title = f"Write this file? ({len(content)} characters)"
        else:
            body = escape(json.dumps(args, indent=2))
            title = f"Allow {tool.name}?"
        self.err.print(Panel(body, title=title, border_style="yellow"))
        if not self.interactive or not sys.stdin.isatty():
            self.err.print("[yellow]Declined automatically (no terminal). Use --yes to allow.[/]")
            return False
        try:
            answer = (
                self.err.input(
                    "[yellow]Allow?[/] [bold]y[/]es / [bold]n[/]o / [bold]a[/]lways this session: "
                )
                .strip()
                .lower()
            )
        except EOFError:
            return False
        if answer in {"a", "always"}:
            self.always_allowed.add(tool.name)
            return True
        return answer in {"y", "yes"}


def _short(value: Any, limit: int = 80) -> str:
    text = value if isinstance(value, str) else json.dumps(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _full_name(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


def connect(host: str | None, err: Console) -> ollama.Client | None:
    client = ollama.Client(host=host)
    try:
        client.list()
    except (ConnectionError, httpx.HTTPError):
        where = host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        err.print(
            f"[red]Could not reach Ollama at {escape(where)}.[/]\n"
            "Hearth runs models locally with Ollama. To get started:\n"
            "  1. Install Ollama from [link=https://ollama.com/download]https://ollama.com/download[/link]\n"
            "  2. Make sure it is running (open the app, or run [bold]ollama serve[/])\n"
            "  3. Run [bold]hearth[/] again"
        )
        return None
    return client


def installed_models(client: ollama.Client) -> list[str]:
    return sorted(m.model for m in client.list().models if m.model)


def ensure_model(client: ollama.Client, model: str, err: Console, assume_yes: bool) -> bool:
    if _full_name(model) in {_full_name(m) for m in installed_models(client)}:
        return True
    err.print(f"Model [bold]{escape(model)}[/] is not downloaded yet.")
    if not assume_yes:
        if not sys.stdin.isatty():
            err.print(f"Run [bold]ollama pull {escape(model)}[/] first, or pass --yes.")
            return False
        answer = err.input("Download it now? [bold][Y/n][/] ").strip().lower()
        if answer not in {"", "y", "yes"}:
            return False
    try:
        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            console=err,
            transient=True,
        ) as progress:
            task = progress.add_task(f"Pulling {model}", total=None)
            for update in client.pull(model, stream=True):
                progress.update(
                    task,
                    description=f"{model}: {update.status}",
                    total=update.total or None,
                    completed=update.completed or 0,
                )
    except ollama.ResponseError as exc:
        err.print(f"[red]Could not download {escape(model)}: {escape(exc.error)}[/]")
        err.print("Browse available models at https://ollama.com/search?c=tools")
        return False
    err.print(f"[green]Downloaded {escape(model)}.[/]")
    return True


def explain_response_error(exc: ollama.ResponseError, model: str, err: Console) -> None:
    message = exc.error or str(exc)
    err.print(f"[red]Ollama error: {escape(message)}[/]")
    if "does not support tools" in message:
        err.print(
            f"[yellow]{escape(model)} can't use tools. Pick a tool-capable model "
            "(see https://ollama.com/search?c=tools) or start Hearth with --no-tools.[/]"
        )
    elif "does not support thinking" in message:
        err.print("[yellow]This model has no thinking mode; drop the --think flag.[/]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hearth",
        description="A private, local AI assistant for your terminal, powered by Ollama.",
        epilog="Examples:\n"
        "  hearth                                 start a chat\n"
        "  hearth what is the latest Python release\n"
        "  cat notes.txt | hearth summarize this\n"
        "  hearth -m llama3.2 --no-tools tell me a joke",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("prompt", nargs="*", help="ask a single question and exit")
    parser.add_argument(
        "-m",
        "--model",
        default=os.environ.get("HEARTH_MODEL", DEFAULT_MODEL),
        help=f"Ollama model to use (default: $HEARTH_MODEL or {DEFAULT_MODEL})",
    )
    parser.add_argument("--host", help="Ollama server URL (default: $OLLAMA_HOST or localhost)")
    parser.add_argument(
        "-y", "--yes", action="store_true", help="allow file writes and shell commands without asking"
    )
    parser.add_argument("--no-tools", action="store_true", help="plain chat, no tools")
    think = parser.add_mutually_exclusive_group()
    think.add_argument(
        "--think",
        dest="think",
        action="store_const",
        const=True,
        help="turn on the model's reasoning mode (if supported)",
    )
    think.add_argument(
        "--no-think",
        dest="think",
        action="store_const",
        const=False,
        help="turn off the model's reasoning mode (faster)",
    )
    parser.add_argument("--show-thinking", action="store_true", help="print the model's reasoning")
    parser.add_argument("--max-steps", type=int, default=15, help="max tool calls per reply")
    parser.add_argument("-V", "--version", action="version", version=f"hearth {__version__}")
    return parser


def _setup_readline() -> None:
    try:
        import readline
    except ImportError:
        return
    try:
        readline.read_history_file(HISTORY_FILE)
    except (OSError, ValueError):
        pass
    readline.set_history_length(1000)

    import atexit

    def save() -> None:
        try:
            readline.write_history_file(HISTORY_FILE)
        except OSError:
            pass

    atexit.register(save)


def run_once(agent: Agent, prompt: str, err: Console) -> int:
    try:
        agent.run(prompt)
    except KeyboardInterrupt:
        agent.ui.on_message_end()
        return 130
    except ollama.ResponseError as exc:
        agent.ui.on_message_end()
        explain_response_error(exc, agent.model, err)
        return 1
    return 0


def repl(agent: Agent, client: ollama.Client, err: Console, assume_yes: bool) -> int:
    _setup_readline()
    tools = ", ".join(agent.tools) or "none"
    err.print(
        f"[bold]Hearth[/] {__version__}  [dim]model:[/] {escape(agent.model)}  "
        f"[dim]tools:[/] {tools}\n[dim]Type /help for commands, Ctrl+D to quit.[/]"
    )
    while True:
        try:
            line = err.input("\n[bold green]you >[/] ").strip()
        except EOFError:
            err.print()
            return 0
        except KeyboardInterrupt:
            err.print()
            continue
        if not line:
            continue
        if line.startswith("/"):
            cmd, _, arg = line.partition(" ")
            arg = arg.strip()
            if cmd in {"/exit", "/quit", "/q"}:
                return 0
            if cmd == "/help":
                err.print(HELP_TEXT)
            elif cmd in {"/reset", "/clear", "/new"}:
                agent.reset()
                err.print("[dim]Started a new conversation.[/]")
            elif cmd == "/tools":
                for tool in agent.tools.values():
                    flag = " [yellow](asks first)[/]" if tool.needs_approval else ""
                    err.print(f"  [cyan]{tool.name}[/]{flag}  {escape(tool.description)}")
                if not agent.tools:
                    err.print("[dim]Tools are turned off.[/]")
            elif cmd == "/models":
                for name in installed_models(client):
                    marker = "*" if _full_name(name) == _full_name(agent.model) else " "
                    err.print(f" {marker} {escape(name)}")
            elif cmd == "/model":
                if not arg:
                    err.print(f"Current model: [bold]{escape(agent.model)}[/]")
                elif ensure_model(client, arg, err, assume_yes):
                    agent.model = arg
                    err.print(f"[dim]Switched to {escape(arg)}.[/]")
            else:
                err.print(f"[red]Unknown command {escape(cmd)}.[/] Type /help.")
            continue
        err.print("[bold blue]hearth >[/]")
        status = run_once(agent, line, err)
        if status == 130:
            err.print("[dim](stopped)[/]")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prompt = " ".join(args.prompt).strip()
    piped = "" if sys.stdin.isatty() else sys.stdin.read().strip()
    if piped:
        prompt = f"{prompt}\n\n{piped}" if prompt else piped
    interactive = not prompt

    ui = RichUI(show_thinking=args.show_thinking, interactive=sys.stdin.isatty())
    err = ui.err
    client = connect(args.host, err)
    if client is None:
        return 1
    if not ensure_model(client, args.model, err, args.yes):
        return 1

    agent = Agent(
        client=client,
        model=args.model,
        ui=ui,
        tools=[] if args.no_tools else BUILTIN_TOOLS,
        max_steps=args.max_steps,
        think=args.think,
        auto_approve=args.yes,
    )
    if interactive:
        return repl(agent, client, err, args.yes)
    return run_once(agent, prompt, err)

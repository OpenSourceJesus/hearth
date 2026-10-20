"""Tools the model can call. Every tool takes JSON-friendly arguments and returns a string."""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import httpx

MAX_OUTPUT_CHARS = 12_000
SHELL_TIMEOUT_SECONDS = 120
USER_AGENT = "Mozilla/5.0 (compatible; HearthAgent/0.1; +https://github.com/OpenSourceJesus/hearth)"


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., str]
    needs_approval: bool = False

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... [truncated {len(text) - limit} more characters]"


def _resolve(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _coerce_args(params: Mapping[str, Any], args: Mapping[str, Any]) -> dict[str, Any]:
    """Small models often send numbers as strings; coerce them to the declared JSON type."""
    props = params.get("properties", {})
    coerced: dict[str, Any] = {}
    for key, value in args.items():
        kind = props.get(key, {}).get("type")
        if kind == "integer" and isinstance(value, str) and value.strip().lstrip("-").isdigit():
            value = int(value)
        elif kind == "boolean" and isinstance(value, str):
            value = value.strip().lower() in {"true", "1", "yes"}
        coerced[key] = value
    return coerced


def execute_tool(tools: Mapping[str, Tool], name: str, args: Mapping[str, Any] | None) -> str:
    tool = tools.get(name)
    if tool is None:
        return f"Error: unknown tool '{name}'. Available tools: {', '.join(tools) or 'none'}."
    try:
        return tool.func(**_coerce_args(tool.parameters, args or {}))
    except TypeError as exc:
        return f"Error: invalid arguments for {name}: {exc}"
    except Exception as exc:  # noqa: BLE001 - tool failures are reported back to the model
        return f"Error: {type(exc).__name__}: {exc}"


def web_search(query: str, max_results: int = 5) -> str:
    from ddgs import DDGS

    max_results = max(1, min(int(max_results), 10))
    results = DDGS().text(query, max_results=max_results)
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results, 1):
        title, href, body = r.get("title", "").strip(), r.get("href", ""), r.get("body", "").strip()
        lines.append(f"{i}. {title}\n   {href}\n   {body}")
    return "\n".join(lines)


class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "svg", "head", "template"}
    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
        raw = re.sub(r"\n\s*\n+", "\n\n", raw)
        return raw.strip()


def html_to_text(html: str) -> tuple[str, str]:
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return parser.title.strip(), parser.text()


def fetch_url(url: str) -> str:
    if not re.match(r"^https?://", url):
        url = "https://" + url
    resp = httpx.get(url, follow_redirects=True, timeout=20, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "")
    if "html" in ctype:
        title, body = html_to_text(resp.text)
        return truncate(f"Title: {title}\nURL: {resp.url}\n\n{body}")
    if ctype.startswith("text/") or "json" in ctype or "xml" in ctype:
        return truncate(resp.text)
    return f"Fetched {resp.url} but it is not a text page (content-type: {ctype or 'unknown'})."


def read_file(path: str, start_line: int = 1, max_lines: int = 400) -> str:
    p = _resolve(path)
    if not p.is_file():
        return f"Error: {p} is not a file."
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(1, int(start_line))
    chunk = lines[start - 1 : start - 1 + max(1, int(max_lines))]
    end = start + len(chunk) - 1
    header = f"{p} (lines {start}-{end} of {len(lines)})\n"
    body = "\n".join(f"{n:>5} | {line}" for n, line in enumerate(chunk, start))
    return truncate(header + body)


def list_directory(path: str = ".") -> str:
    p = _resolve(path)
    if not p.is_dir():
        return f"Error: {p} is not a directory."
    entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
    lines = [f"{p}:"]
    for entry in entries[:300]:
        if entry.is_dir():
            lines.append(f"  {entry.name}/")
        else:
            try:
                size = entry.stat().st_size
            except OSError:
                size = 0
            lines.append(f"  {entry.name}  ({size} bytes)")
    if len(entries) > 300:
        lines.append(f"  ... and {len(entries) - 300} more entries")
    if len(entries) == 0:
        lines.append("  (empty)")
    return "\n".join(lines)


def write_file(path: str, content: str) -> str:
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content.encode('utf-8'))} bytes to {p}."


def run_shell(command: str) -> str:
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=SHELL_TIMEOUT_SECONDS,
            cwd=os.getcwd(),
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {SHELL_TIMEOUT_SECONDS} seconds."
    out = [f"exit code: {proc.returncode}"]
    if proc.stdout:
        out.append(f"stdout:\n{proc.stdout}")
    if proc.stderr:
        out.append(f"stderr:\n{proc.stderr}")
    return truncate("\n".join(out))


BUILTIN_TOOLS: list[Tool] = [
    Tool(
        name="web_search",
        description="Search the web (DuckDuckGo). Returns titles, URLs and snippets. Use for current "
        "events, facts you are unsure about, or anything after your training data.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "max_results": {"type": "integer", "description": "1-10, default 5."},
            },
            "required": ["query"],
        },
        func=web_search,
    ),
    Tool(
        name="fetch_url",
        description="Download a web page and return its readable text. Use after web_search to read "
        "a result in full.",
        parameters={
            "type": "object",
            "properties": {"url": {"type": "string", "description": "The http(s) URL to fetch."}},
            "required": ["url"],
        },
        func=fetch_url,
    ),
    Tool(
        name="list_directory",
        description="List files and folders in a directory on the user's computer.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Directory path. Defaults to '.'."}},
        },
        func=list_directory,
    ),
    Tool(
        name="read_file",
        description="Read a text file on the user's computer. Output has line numbers.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path."},
                "start_line": {"type": "integer", "description": "First line to read, default 1."},
                "max_lines": {"type": "integer", "description": "Lines to read, default 400."},
            },
            "required": ["path"],
        },
        func=read_file,
    ),
    Tool(
        name="write_file",
        description="Create or overwrite a text file on the user's computer. The user must approve.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path."},
                "content": {"type": "string", "description": "Full new contents of the file."},
            },
            "required": ["path", "content"],
        },
        func=write_file,
        needs_approval=True,
    ),
    Tool(
        name="run_shell",
        description="Run a shell command in the current directory and return its output. "
        "The user must approve. Do not run interactive commands.",
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The command to run."}},
            "required": ["command"],
        },
        func=run_shell,
        needs_approval=True,
    ),
]

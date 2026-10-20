"""The agent loop: stream a reply, run any requested tools, feed results back, repeat."""

from __future__ import annotations

import os
import platform
from collections.abc import Iterable, Iterator
from datetime import datetime
from typing import Any, Protocol

from ollama import ChatResponse, Message

from hearth_agent.tools import Tool, execute_tool


class ChatClient(Protocol):
    def chat(self, **kwargs: Any) -> Iterator[ChatResponse]: ...


class AgentUI(Protocol):
    def on_wait(self) -> None: ...
    def on_thinking(self, text: str) -> None: ...
    def on_text(self, text: str) -> None: ...
    def on_message_end(self) -> None: ...
    def on_tool_call(self, name: str, args: dict[str, Any]) -> None: ...
    def on_tool_result(self, name: str, result: str) -> None: ...
    def approve(self, tool: Tool, args: dict[str, Any]) -> bool: ...


class SilentUI:
    """UI that shows nothing and approves nothing. Useful for tests and scripting."""

    def on_wait(self) -> None: ...
    def on_thinking(self, text: str) -> None: ...
    def on_text(self, text: str) -> None: ...
    def on_message_end(self) -> None: ...
    def on_tool_call(self, name: str, args: dict[str, Any]) -> None: ...
    def on_tool_result(self, name: str, result: str) -> None: ...

    def approve(self, tool: Tool, args: dict[str, Any]) -> bool:
        return False


def default_system_prompt(tools: Iterable[Tool]) -> str:
    names = [t.name for t in tools]
    lines = [
        "You are Hearth, a helpful AI assistant running privately on the user's own computer.",
        f"Current date and time: {datetime.now().astimezone():%A, %B %d, %Y %H:%M %Z}.",
        f"Operating system: {platform.system()} {platform.release()}.",
        f"Current working directory: {os.getcwd()}.",
        "Be accurate and concise. Use Markdown when it helps readability.",
    ]
    if names:
        lines += [
            f"You can call these tools: {', '.join(names)}.",
            "Use a tool whenever it would make your answer more accurate or current; "
            "don't guess about files, commands or recent events.",
            "When you use information from the web, cite the source URLs.",
            "If the user declines a tool call, respect that and continue without it.",
        ]
    return "\n".join(lines)


def split_leaked_reasoning(content: str) -> tuple[str, str]:
    """Some models ignore think=False and write their reasoning into the reply, ending with </think>."""
    head, sep, tail = content.rpartition("</think>")
    if not sep:
        return content, ""
    return tail.strip(), head.replace("<think>", "").strip()


class Agent:
    def __init__(
        self,
        client: ChatClient,
        model: str,
        ui: AgentUI,
        tools: Iterable[Tool] = (),
        system_prompt: str | None = None,
        max_steps: int = 15,
        think: bool | None = None,
        auto_approve: bool = False,
    ) -> None:
        self.client = client
        self.model = model
        self.ui = ui
        self.tools = {t.name: t for t in tools}
        self.system_prompt = system_prompt or default_system_prompt(self.tools.values())
        self.max_steps = max_steps
        self.think = think
        self.auto_approve = auto_approve
        self.messages: list[Message | dict[str, Any]] = []
        self.reset()

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def run(self, user_input: str) -> str:
        """Process one user turn and return the final assistant text.

        If interrupted (e.g. Ctrl+C), history is rolled back so the conversation stays valid.
        """
        checkpoint = len(self.messages)
        self.messages.append({"role": "user", "content": user_input})
        try:
            return self._loop()
        except BaseException:
            del self.messages[checkpoint:]
            raise

    def _loop(self) -> str:
        schemas = [t.schema() for t in self.tools.values()] or None
        for _ in range(self.max_steps):
            content, thinking, tool_calls = self._stream_step(schemas)
            self.messages.append(
                Message(
                    role="assistant",
                    content=content,
                    thinking=thinking or None,
                    tool_calls=tool_calls or None,
                )
            )
            if not tool_calls:
                return content
            for call in tool_calls:
                name = call.function.name
                args = dict(call.function.arguments or {})
                self.ui.on_tool_call(name, args)
                result = self._run_tool(name, args)
                self.ui.on_tool_result(name, result)
                self.messages.append({"role": "tool", "tool_name": name, "content": result})
        note = f"(Stopped after {self.max_steps} tool-use steps without a final answer.)"
        self.ui.on_text(note)
        self.ui.on_message_end()
        return note

    def _stream_step(self, schemas: list[dict[str, Any]] | None) -> tuple[str, str, list]:
        self.ui.on_wait()
        kwargs: dict[str, Any] = {"model": self.model, "messages": self.messages, "stream": True}
        if schemas:
            kwargs["tools"] = schemas
        if self.think is not None:
            kwargs["think"] = self.think
        content: list[str] = []
        thinking: list[str] = []
        tool_calls: list[Message.ToolCall] = []
        for chunk in self.client.chat(**kwargs):
            msg = chunk.message
            if msg.thinking:
                thinking.append(msg.thinking)
                self.ui.on_thinking(msg.thinking)
            if msg.content:
                content.append(msg.content)
                self.ui.on_text(msg.content)
            if msg.tool_calls:
                tool_calls.extend(msg.tool_calls)
        self.ui.on_message_end()
        text, leaked = split_leaked_reasoning("".join(content))
        return text, "".join(thinking) + leaked, tool_calls

    def _run_tool(self, name: str, args: dict[str, Any]) -> str:
        tool = self.tools.get(name)
        if tool is not None and tool.needs_approval and not self._is_approved(tool, args):
            return "The user declined to run this tool call."
        return execute_tool(self.tools, name, args)

    def _is_approved(self, tool: Tool, args: dict[str, Any]) -> bool:
        if self.auto_approve:
            return True
        return self.ui.approve(tool, args)

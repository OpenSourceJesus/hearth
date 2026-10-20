from __future__ import annotations

from typing import Any

import pytest
from ollama import ChatResponse, Message

from hearth_agent.agent import Agent, SilentUI
from hearth_agent.tools import Tool


def chunk(content: str = "", tool_calls: list[tuple[str, dict]] | None = None) -> ChatResponse:
    calls = None
    if tool_calls:
        calls = [
            Message.ToolCall(function=Message.ToolCall.Function(name=n, arguments=a)) for n, a in tool_calls
        ]
    return ChatResponse(message=Message(role="assistant", content=content, tool_calls=calls))


class FakeClient:
    """Replays a scripted list of streamed responses, one list per chat() call."""

    def __init__(self, script: list[list[ChatResponse]]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return iter(self.script.pop(0))


class RecordingUI(SilentUI):
    def __init__(self, approve: bool = False) -> None:
        self.text: list[str] = []
        self.tool_calls: list[str] = []
        self._approve = approve
        self.approvals: list[str] = []

    def on_text(self, text: str) -> None:
        self.text.append(text)

    def on_tool_call(self, name: str, args: dict[str, Any]) -> None:
        self.tool_calls.append(name)

    def approve(self, tool: Tool, args: dict[str, Any]) -> bool:
        self.approvals.append(tool.name)
        return self._approve


def echo_tool(needs_approval: bool = False) -> Tool:
    return Tool(
        name="echo",
        description="Echo text back.",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        func=lambda text: f"echo: {text}",
        needs_approval=needs_approval,
    )


def test_plain_reply_streams_and_is_recorded():
    client = FakeClient([[chunk("Hel"), chunk("lo!")]])
    ui = RecordingUI()
    agent = Agent(client, "m", ui)

    assert agent.run("hi") == "Hello!"
    assert "".join(ui.text) == "Hello!"
    assert [m["role"] if isinstance(m, dict) else m.role for m in agent.messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert "tools" not in client.calls[0]


def test_tool_call_result_is_sent_back_to_model():
    client = FakeClient(
        [
            [chunk(tool_calls=[("echo", {"text": "ping"})])],
            [chunk("The tool said ping.")],
        ]
    )
    ui = RecordingUI()
    agent = Agent(client, "m", ui, tools=[echo_tool()])

    assert agent.run("use echo") == "The tool said ping."
    assert ui.tool_calls == ["echo"]
    second_call_messages = client.calls[1]["messages"]
    assert second_call_messages[-1] == {"role": "tool", "tool_name": "echo", "content": "echo: ping"}
    assert client.calls[0]["tools"][0]["function"]["name"] == "echo"


def test_declined_tool_is_not_executed():
    ran = []
    tool = Tool(
        "danger",
        "",
        {"type": "object", "properties": {}},
        lambda: ran.append(1) or "ran",
        needs_approval=True,
    )
    client = FakeClient([[chunk(tool_calls=[("danger", {})])], [chunk("ok")]])
    ui = RecordingUI(approve=False)
    agent = Agent(client, "m", ui, tools=[tool])

    agent.run("go")
    assert ran == []
    assert ui.approvals == ["danger"]
    assert "declined" in client.calls[1]["messages"][-1]["content"]


def test_auto_approve_skips_prompt():
    client = FakeClient([[chunk(tool_calls=[("echo", {"text": "x"})])], [chunk("done")]])
    ui = RecordingUI(approve=False)
    agent = Agent(client, "m", ui, tools=[echo_tool(needs_approval=True)], auto_approve=True)

    agent.run("go")
    assert ui.approvals == []
    assert client.calls[1]["messages"][-1]["content"] == "echo: x"


def test_max_steps_stops_runaway_tool_loops():
    loop = [[chunk(tool_calls=[("echo", {"text": "again"})])] for _ in range(3)]
    agent = Agent(FakeClient(loop), "m", RecordingUI(), tools=[echo_tool()], max_steps=3)

    assert "Stopped after 3" in agent.run("loop forever")


def test_interrupt_rolls_back_history():
    class ExplodingClient:
        def chat(self, **kwargs):
            yield chunk("partial")
            raise KeyboardInterrupt

    agent = Agent(ExplodingClient(), "m", RecordingUI())
    with pytest.raises(KeyboardInterrupt):
        agent.run("hi")
    assert len(agent.messages) == 1


def test_reasoning_leaked_into_content_is_moved_out_of_history():
    client = FakeClient([[chunk("let me reason...\n</think>\n\n"), chunk("Four.")]])
    agent = Agent(client, "m", RecordingUI())

    assert agent.run("2+2?") == "Four."
    assert agent.messages[-1].content == "Four."
    assert agent.messages[-1].thinking == "let me reason..."


def test_think_flag_is_forwarded_only_when_set():
    client = FakeClient([[chunk("a")], [chunk("b")]])
    Agent(client, "m", RecordingUI()).run("x")
    Agent(client, "m", RecordingUI(), think=False).run("y")
    assert "think" not in client.calls[0]
    assert client.calls[1]["think"] is False

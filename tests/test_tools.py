from __future__ import annotations

from hearth_agent.tools import (
    BUILTIN_TOOLS,
    execute_tool,
    html_to_text,
    list_directory,
    read_file,
    run_shell,
    truncate,
    write_file,
)

TOOLS = {t.name: t for t in BUILTIN_TOOLS}


def test_schemas_are_valid_function_definitions():
    for tool in BUILTIN_TOOLS:
        schema = tool.schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == tool.name
        assert schema["function"]["parameters"]["type"] == "object"
        for required in schema["function"]["parameters"].get("required", []):
            assert required in schema["function"]["parameters"]["properties"]


def test_risky_tools_need_approval():
    assert TOOLS["run_shell"].needs_approval
    assert TOOLS["write_file"].needs_approval
    assert not TOOLS["read_file"].needs_approval


def test_write_then_read_file(tmp_path):
    target = tmp_path / "sub" / "notes.txt"
    assert "Wrote" in write_file(str(target), "one\ntwo\nthree\n")
    out = read_file(str(target), start_line=2, max_lines=1)
    assert "lines 2-2 of 3" in out
    assert "two" in out and "one" not in out


def test_list_directory(tmp_path):
    (tmp_path / "folder").mkdir()
    (tmp_path / "file.txt").write_text("hi")
    out = list_directory(str(tmp_path))
    assert "folder/" in out
    assert "file.txt  (2 bytes)" in out
    assert out.index("folder/") < out.index("file.txt")


def test_run_shell_captures_output_and_exit_code():
    out = run_shell("echo hello && echo oops 1>&2 && exit 3")
    assert "exit code: 3" in out
    assert "hello" in out
    assert "oops" in out


def test_html_to_text_strips_scripts_and_keeps_title():
    html = (
        "<html><head><title>My Page</title><style>p{}</style></head>"
        "<body><script>alert(1)</script><h1>Header</h1><p>Body   text</p></body></html>"
    )
    title, text = html_to_text(html)
    assert title == "My Page"
    assert "alert" not in text and "p{}" not in text
    assert "Header" in text and "Body text" in text


def test_execute_tool_reports_errors_instead_of_raising(tmp_path):
    assert "unknown tool" in execute_tool(TOOLS, "nope", {})
    assert "invalid arguments" in execute_tool(TOOLS, "read_file", {"wrong": 1})
    assert "not a file" in execute_tool(TOOLS, "read_file", {"path": str(tmp_path / "missing")})


def test_execute_tool_coerces_string_numbers(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("1\n2\n3\n")
    out = execute_tool(TOOLS, "read_file", {"path": str(f), "start_line": "3", "max_lines": "1"})
    assert "lines 3-3 of 3" in out


def test_truncate():
    assert truncate("abc", 5) == "abc"
    assert truncate("abcdefgh", 3).startswith("abc\n... [truncated 5")

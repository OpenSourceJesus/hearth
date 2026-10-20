from __future__ import annotations

import io

import pytest

from hearth_agent import cli


def test_unreachable_ollama_exits_with_help(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert cli.main(["--host", "http://127.0.0.1:9", "hello"]) == 1
    assert "Could not reach Ollama" in capsys.readouterr().err


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert "hearth" in capsys.readouterr().out


def test_model_names_default_to_latest_tag():
    assert cli._full_name("llama3.2") == "llama3.2:latest"
    assert cli._full_name("qwen3:4b") == "qwen3:4b"

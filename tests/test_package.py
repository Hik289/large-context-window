from __future__ import annotations

import sys

import pytest

import agent_memory
from agent_memory.cli import main


def test_version_is_exposed_without_loading_optional_backends() -> None:
    assert agent_memory.__version__ == "0.1.0"


def test_cli_version(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["agent-memory", "--version"])

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 0
    assert "0.1.0" in capsys.readouterr().out

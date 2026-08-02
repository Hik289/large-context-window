from __future__ import annotations

import subprocess
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


def test_module_entrypoint_exposes_version() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "agent_memory", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert agent_memory.__version__ in completed.stdout


def test_config_command_reports_active_file() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "agent_memory", "config"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Model config:" in completed.stdout
    assert "chat_low:" in completed.stdout
    assert "PLACEHOLDER" in completed.stdout

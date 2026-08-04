from __future__ import annotations

import json
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


def test_doctor_json_is_safe_and_machine_readable(capsys) -> None:
    exit_code = main(["doctor", "--json"])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["package"] == "UltraMem"
    assert report["core_ready"] is True
    assert report["models_ready"] is False
    assert set(report["optional_groups"]) == {
        "dev",
        "documents",
        "evaluation",
        "figures",
        "llm",
        "local-models",
        "retrieval",
    }
    assert "api_key" not in json.dumps(report).lower()


def test_package_root_exposes_lightweight_method_api() -> None:
    from agent_memory import DualNode, TokenLedger, validate_batch

    assert DualNode.__name__ == "DualNode"
    assert TokenLedger.__name__ == "TokenLedger"
    assert callable(validate_batch)


def test_core_install_does_not_import_optional_backends() -> None:
    script = """
import sys
from agent_memory import *
from agent_memory import MemoryClient
from agent_memory.methods import *
import agent_memory.utils.embedding

client = MemoryClient(api_key="test-token", server_url="https://memory.example")
assert client.is_remote
for module in ("chromadb", "openai", "torch", "transformers"):
    assert module not in sys.modules, module
assert "agent_memory.core.local_client" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_optional_interfaces_remain_discoverable() -> None:
    assert "DualIndex" in dir(agent_memory)
    assert "MemoryClient" in dir(agent_memory)

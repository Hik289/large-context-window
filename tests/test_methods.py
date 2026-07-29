from __future__ import annotations

import json

import pytest

from agent_memory.methods import DualNode, TokenLedger, validate_batch, validate_one
from agent_memory.methods.configs.isolation import V4ConfigError, validate_config


def _valid_node(node_id: str = "node-1") -> DualNode:
    return DualNode(
        node_id=node_id,
        level="L0",
        tenant_id="demo",
        distilled_text="Compact routing memory.",
        detailed_text="Detailed source evidence used for grounded answering.",
        distilled_tokens=4,
        detailed_tokens=9,
        source_evidence_ids=[f"evidence:{node_id}"],
    )


def test_dual_node_round_trip_and_validation() -> None:
    node = _valid_node()
    restored = DualNode.from_dict(node.to_dict())

    assert restored == node
    assert validate_one(restored) == []


def test_batch_acceptance_tracks_representation_and_provenance() -> None:
    report = validate_batch([_valid_node("a"), _valid_node("b")])

    assert report["overall_pass"] is True
    assert report["invalid_node_count"] == 0
    assert report["counts"]["n_have_provenance"] == 2


def test_token_ledger_exports_without_deadlock(tmp_path) -> None:
    ledger = TokenLedger(run_id="test", method="dual-view")
    ledger.record("retrieval", "local", 20, 0, 0.01)
    ledger.record("final_answer", "reader", 100, 12, 0.02)

    output = tmp_path / "ledger.json"
    ledger.export(str(output), include_raw=True)
    data = json.loads(output.read_text())

    assert data["totals"]["calls"] == 2
    assert data["totals"]["total_tokens"] == 132
    assert len(data["records"]) == 2


def test_isolation_rejects_forbidden_subsampling() -> None:
    config = {
        "run_id": "r1",
        "dataset": "own_full",
        "method": "V4",
        "seed": 7,
        "models": {
            "hierarchy_low": "chat_low",
            "hierarchy_high": "chat_high",
            "answer": "chat_high",
            "judge": "judge",
        },
        "paths": {
            "corpus": "corpus.parquet",
            "queries": "questions.jsonl",
            "chroma": "index",
            "output": "outputs",
            "manifest": "manifest.json",
        },
        "tokenizer": "cl100k_base",
        "max_queries": 10,
    }

    with pytest.raises(V4ConfigError, match="FORBIDDEN config keys"):
        validate_config(config)

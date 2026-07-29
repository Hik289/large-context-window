# Reproducibility Guide

This guide separates checks that are fully self-contained from experiments that require
licensed data or configured model endpoints.

## 1. Credential-Free Verification

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

pytest
python -m agent_memory.methods.dual_node
python -m agent_memory.methods.token_ledger
python -m agent_memory.methods.configs.isolation
```

These commands validate serialization, dual-representation invariants, provenance,
configuration isolation, token accounting, and the package CLI.

## 2. Figure Reproduction

```bash
pip install -e ".[figures]"
python figures/make_all_figs.py
```

The figure scripts embed the final aggregate tables used by the manuscript. They write
PDF and PNG outputs beside the scripts; generated figures are ignored by Git.

## 3. Model Configuration

```bash
cp .env.example .env
cp configs/models.example.yaml configs/models.yaml
```

Each model alias must resolve to an explicit chat-completions-compatible endpoint and
model identifier. Record the alias configuration, embedding model, seed, corpus manifest,
and Git commit with every run.

## 4. Data Contract

The benchmark expects:

- a document table containing stable `doc_id`, `title`, `source_type`, and `content`
  fields;
- a query JSONL stream with stable question identifiers and gold answers;
- expected evidence identifiers for retrieval evaluation;
- a deterministic tier manifest for each corpus scale;
- a local directory for vector indexes and generated outputs.

Corpora, question files, manifests, and generated indices are intentionally not shipped.
Use only data for which you have the appropriate access and redistribution rights.

## 5. Full Replay Boundary

The repository contains the installable memory components, configuration guards,
external-benchmark adapters, and aggregate plotting scripts. A full EnterpriseRAG replay
also requires the official benchmark data and evaluation package. The adapter in
`scripts/eval/metrics_based_eval.py` imports that package's `src.*` modules and is not a
standalone evaluator.

For an auditable replay, preserve:

1. repository commit;
2. corpus and query manifest checksums;
3. model aliases and provider endpoints without secrets;
4. embedding model and dimension;
5. random seed and candidate depths;
6. per-stage token ledger;
7. predictions, citations, and metric outputs.

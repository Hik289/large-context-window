# Reproducibility Guide

This guide separates credential-free method checks from private full-system runs that
require authorized corpora and configured model endpoints.

## Credential-Free Verification

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

pytest
python -m agent_memory.methods.dual_node
python -m agent_memory.methods.token_ledger
python -m agent_memory.methods.configs.isolation
```

These commands validate serialization, dual-representation invariants, provenance
fields, configuration isolation, token accounting, and package importability without
calling a model or loading a private corpus.

## Method Figure Assets

```bash
pip install -e ".[figures]"
python figures/make_all_figs.py
```

Generated figures are ignored by Git. Some plotting utilities are retained for authors
who have the private aggregate records used in manuscript preparation; they are not
required for the public method checks.

## Model Configuration

```bash
cp .env.example .env
cp configs/models.example.yaml configs/models.yaml
```

Each model alias must resolve to an explicit chat-completions-compatible endpoint and
model identifier. The resolver fails loudly when an alias is missing or unresolved. Raw
API keys should stay in environment variables, never in YAML.

## Data Contract

Private runs should provide:

- a document table with stable source identifiers, titles, source types, and content;
- a query stream with stable question identifiers and answer targets;
- expected evidence identifiers when retrieval evaluation is required;
- a corpus manifest and local index directory;
- an output directory for predictions, citations, ledgers, and run metadata.

Corpora, query files, manifests, generated indexes, and model outputs are intentionally
not shipped. Use only data for which you have the appropriate access and redistribution
rights.

## Full System Boundary

The repository contains the installable memory components, configuration guards,
optional integration adapters, and plotting utilities. A full private replay may also
require upstream data and evaluation packages that are not redistributed here. The
adapter under `scripts/eval/` is an integration boundary, not a standalone evaluator.

## Replay Record

For an auditable private run, preserve:

- repository commit;
- corpus and query manifests;
- model aliases and provider endpoints without secrets;
- embedding model and dimension;
- retrieval and promotion configuration;
- per-stage token ledger;
- predictions, citations, and metric outputs.

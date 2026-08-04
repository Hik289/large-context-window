# UltraMem: Ultra-Large Memory Context Windows

**Source-resolved retrieval for persistent memory beyond the prompt**

<p align="center">
  <img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-blue">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
</p>

UltraMem is an installable research artifact for agents that operate over document
collections too large to place in a single prompt. It separates low-cost candidate
discovery from source-faithful evidence loading: compact memory records retrieve broadly,
immutable identifiers resolve every hit to its source text, and only a bounded evidence
set reaches the answer model.

<p align="center">
  <img src="assets/pipeline.png" width="100%" alt="UltraMem dual-view retrieval and bounded evidence pipeline">
</p>

<p align="center"><em>Address the full memory; read only the evidence needed for the current answer.</em></p>

## Highlights

- **Dual-view memory.** Each `DualNode` pairs a compact retrieval representation with
  detailed source evidence and stable provenance identifiers.
- **Source-resolved generation.** Distilled hits are mapped back to authoritative text
  before reranking, packing, and answer generation.
- **Bounded active context.** Candidate depth and answer-context size are independent,
  allowing persistent memory to grow without expanding every prompt.
- **Auditable execution.** Model aliases, representation checks, source identifiers,
  and per-stage token ledgers make runs inspectable.
- **Provider-neutral configuration.** Model calls use a general
  chat-completions-compatible endpoint; credentials remain in environment variables.

## Architecture

| Stage | Input | Output | Contract |
| --- | --- | --- | --- |
| Build | Authorized source spans | Distilled and detailed views | Preserve source identifiers |
| Recall | Original and rewritten queries | Multi-route candidates | Search both views |
| Resolve | Distilled candidates | Authoritative source text | Never answer from a summary alone |
| Select | Source-text candidates | Ranked evidence set | Fuse, deduplicate, and rerank |
| Pack | Ranked evidence | Bounded active context | Enforce an explicit token budget |
| Answer | Loaded evidence | Answer and used sources | Attribute without rewriting the answer |

The key distinction is between **addressable memory** and **active context**. The former
can span 10M to 250M tokens; the latter remains a small, inspectable evidence budget for
one query.

## Installation

UltraMem requires Python 3.11 or newer.

```bash
git clone https://github.com/Hik289/large-context-window.git
cd large-context-window

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

The core installation provides the dual-view data contracts, token ledger, CLI, and
remote client without loading a vector database or model runtime. Add only the capability
groups needed for a deployment:

```bash
pip install -e ".[retrieval,llm,documents]"  # complete local memory pipeline
pip install -e ".[local-models]"             # optional local model execution
pip install -e ".[evaluation,figures]"       # evaluation and publication figures
pip install -e ".[dev]"                      # tests and package build tools
```

## Quick Start

The zero-credential example validates the representation and provenance contracts:

```bash
python examples/minimal_contract.py
```

The same primitives are available from the package root:

```python
from agent_memory import DualNode, TokenLedger, validate_batch

node = DualNode(
    node_id="policy:001",
    level="L0",
    distilled_text="Travel reimbursement policy and approval rules.",
    detailed_text="Employees must submit receipts within 30 days of travel.",
    distilled_tokens=7,
    detailed_tokens=10,
    source_evidence_ids=["policy:001#section-4"],
)

report = validate_batch([node])
assert report["overall_pass"]

ledger = TokenLedger(run_id="demo", method="dual-view")
ledger.record("retrieval", "local", input_tokens=7, output_tokens=0, wall_seconds=0.01)
print(ledger.grand_total())
```

## Client API

`MemoryClient` provides one entry point for deployed and in-process operation. A core
installation can connect to a compatible UltraMem service without importing local model,
document, or vector-store dependencies:

```python
import os

from agent_memory import MemoryClient

memory = MemoryClient(
    api_key=os.environ["ULTRAMEM_API_KEY"],
    server_url=os.environ["ULTRAMEM_SERVER_URL"],
)
memory.add(
    "Quarterly access reviews are due by the final business day.",
    metadata={"source_id": "policy:access-review"},
)
matches = memory.query("When is the access review due?", top_k=5)
```

Local operation accepts a `DictConfig` plus a user-scoped identifier and exposes file,
chat, email, planner, and advanced retrieval workflows. See the [API guide](docs/API.md)
for supported operations, optional dependencies, and error behavior.

## General Model API

Copy the public templates and provide the endpoints used by your run:

```bash
cp .env.example .env
cp configs/models.example.yaml configs/models.yaml
```

```dotenv
LLM_API_BASE=https://your-endpoint.example/v1
LLM_API_KEY=your-secret
LLM_CHAT_MODEL=your-chat-model
LLM_JUDGE_MODEL=your-judge-model
```

Model names live in `configs/models.yaml`; API keys stay in `.env`. The resolver rejects
missing aliases, unresolved model names, and non-general providers before a request is
sent. Set `ULTRAMEM_MODELS_CONFIG` when the alias file is outside the repository root.

Inspect the active, non-secret configuration and optional dependency groups:

```bash
ultramem doctor
ultramem doctor --json
ultramem config
```

Use `ultramem doctor --strict` in deployment checks when model aliases must be fully
configured. The command never prints API keys.

## Reproducibility

Credential-free checks cover package imports, CLI behavior, dual-view invariants,
provenance, configuration isolation, and token accounting:

```bash
make check
```

Equivalent commands are listed in the
[reproducibility guide](docs/REPRODUCIBILITY.md). Full benchmark replays additionally
require authorized corpora and configured model endpoints; private data, generated
indexes, raw predictions, and credentials are intentionally excluded.

| Reproduction level | Included | Entry point |
| --- | --- | --- |
| Package and CLI checks | Yes | `pytest` |
| Representation and provenance checks | Yes | `python -m agent_memory.methods.dual_node` |
| Token-accounting checks | Yes | `python -m agent_memory.methods.token_ledger` |
| Publication plotting code | Yes | `python figures/make_all_figs.py` |
| Full corpus replay | Requires authorized data and endpoints | `docs/REPRODUCIBILITY.md` |

## Repository Layout

```text
large-context-window/
├── src/agent_memory/
│   ├── methods/          # dual nodes, indexing, hierarchy construction, token ledger
│   ├── document_eval/    # ingestion, retrieval, answering, and metrics
│   ├── retriever/        # semantic, hybrid, planning, and reformulation strategies
│   ├── builder/          # document, chat, and email memory builders
│   ├── processors/       # text, PDF, Word, PowerPoint, Excel, and Markdown readers
│   ├── core/             # memory entries, stores, filters, planners, and source cues
│   └── db_clients/       # vector-store and cache adapters
├── configs/              # public model-routing templates
├── examples/             # runnable, credential-free examples
├── scripts/              # evaluation and optional external integrations
├── figures/              # manuscript plotting utilities
├── tests/                # package and method checks
└── docs/                 # API, artifact, and reproducibility contracts
```

## Artifact Scope

This release contains the method implementation and public contracts. It does not ship
private or licensed corpora, generated vector indexes, raw model outputs, real
credentials, or provider account configuration. External datasets and optional baseline
implementations retain their own licenses and access terms.

- [Artifact contract](docs/ARTIFACT.md)
- [API guide](docs/API.md)
- [Reproducibility guide](docs/REPRODUCIBILITY.md)
- [Evaluation boundary](docs/RESULTS.md)
- [Contribution guide](CONTRIBUTING.md)
- [Security policy](SECURITY.md)

## License

UltraMem is released under the [MIT License](LICENSE).

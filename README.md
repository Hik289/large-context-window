# Large Context Window Memory

**A provenance-preserving dual-view memory solution for agents operating beyond the prompt**

<p align="center">
  <img alt="Python package" src="https://img.shields.io/badge/Python-package-blue">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
</p>

Large Context Window Memory is a method artifact for agents that must reason over
persistent document collections without flattening those collections into a single
prompt. The repository is organized as a solution package: it releases the memory
representation, retrieval path, promotion controls, context
packing, provenance checks, configuration templates, and credential-free tests needed
to instantiate the method on an authorized corpus.

The central design is a dual-view memory. Each source span is represented as a compact
distilled node for navigation and a source-linked detailed view for grounding. At query
time, the system searches the distilled and detailed surfaces, promotes detailed
payloads only when they are needed, packs a bounded working context, and preserves the
evidence trail used by the final answer.

<p align="center">
  <img src="assets/pipeline.png" alt="Dual-view memory construction, retrieval, reranking, bounded context packing, and citation pipeline">
</p>

## Method

The method treats long-context reasoning as a memory-system problem rather than a
larger-prompt problem. It separates what the agent needs for navigation from what it
needs for faithful answer generation, then reconnects those views through explicit
identifiers and provenance records.

**Dual-view representation.** `DualNode` is the canonical memory unit. It carries a
distilled text field for low-cost routing, a detailed text or detail reference for
grounded reading, source evidence identifiers, lifecycle state, and model attribution.
The validation helpers enforce that every node is auditable before it enters a run.

**Hierarchical construction.** The builder converts document-level records into
source-linked memory nodes. Distillation is routed through named model aliases, while
detailed evidence remains anchored to the original span. This gives the system a compact
semantic surface without discarding the text needed for citation and inspection.

**Dual indexing.** `DualIndex` maintains separate distilled and detailed collections.
The distilled side returns navigational candidates. The detailed side can be searched
without loading full text into the answer context, so evidence discovery and evidence
consumption remain distinct operations.

**On-demand promotion.** Detailed payloads are loaded only after the retrieval policy
chooses to promote them. This keeps the context window focused on useful evidence
instead of treating every detailed match as prompt material.

**Budgeted packing.** The context packer combines distilled memories and promoted
detailed evidence under an explicit budget. The goal is not to maximize prompt volume;
it is to provide the answer model with a compact, inspectable working set.

**Provenance and accounting.** Citation extraction, source evidence identifiers, model
alias resolution, and token-ledger records make the method auditable. The code fails
loudly when required model aliases are missing or unresolved.

## Contributions

- A dual-view memory schema that keeps distilled navigation and detailed evidence in
  one auditable node contract.
- A retrieval interface that separates candidate discovery from detailed-context
  loading.
- A promotion path for bringing detailed memory into the prompt only when the query
  requires it.
- A package-level configuration boundary that supports general chat-completions-style
  endpoints without committing provider-specific credentials.
- Credential-free tests for method invariants, package importability, configuration
  isolation, and token accounting.

## Installation

```bash
git clone git@github.com:Hik289/large-context-window.git
cd large-context-window

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Install only the capabilities needed for a run:

```bash
pip install -e ".[retrieval,llm]"
pip install -e ".[documents]"
pip install -e ".[evaluation,figures]"
pip install -r requirements.txt
```

## Local Verification

The following checks require no private corpus and no model credential:

```bash
agent-memory --version
pytest
python -m agent_memory.methods.dual_node
python -m agent_memory.methods.token_ledger
python -m agent_memory.methods.configs.isolation
```

## Configuration

Copy the public templates and fill in only the endpoints you use:

```bash
cp .env.example .env
cp configs/models.example.yaml configs/models.yaml
```

Model aliases map to general chat-completions-style endpoints through environment
variables. Raw keys stay outside YAML. Real `.env` files, corpora, model outputs,
generated indexes, manifests, and run artifacts are ignored by Git.

## Artifact Boundary

This repository contains the method implementation and public contracts. It does not
ship private corpora, licensed evaluation data, generated vector indexes, raw model
outputs, provider account configuration, or real credentials. Optional adapters under
`scripts/` are included only as integration boundaries for users who already have
authorized access to the corresponding upstream resources.

For review and reuse details, see:

- [Artifact contract](docs/ARTIFACT.md)
- [Reproducibility guide](docs/REPRODUCIBILITY.md)
- [Evaluation boundary](docs/RESULTS.md)

## Repository Map

```text
large-context-window/
├── src/agent_memory/
│   ├── methods/            # dual nodes, dual index, hierarchy builder, token ledger
│   ├── document_eval/      # document ingestion, retrieval, answering, and metrics utilities
│   ├── retriever/          # semantic, hybrid, planning, and reformulation retrieval
│   ├── builder/            # document, chat, and email memory builders
│   ├── processors/         # text, PDF, Word, PowerPoint, Excel, and Markdown processors
│   ├── core/               # memory entries, stores, filters, planners, and source cues
│   └── db_clients/         # local vector-store and cache adapters
├── scripts/                # optional integration adapters
├── figures/                # method schematic and manuscript plotting utilities
├── configs/                # public model-routing templates
├── tests/                  # credential-free method and package checks
└── docs/                   # artifact, reproducibility, and evaluation contracts
```

## Scope

Large-context memory is useful only when sources are available, permitted, and
traceable. This artifact focuses on the method layer: representation, retrieval,
promotion, packing, and provenance. Deployment choices such as access control, secret
management, provider routing, and corpus governance remain the responsibility of the
deploying environment.

## License

The code is released under the [MIT License](LICENSE). External datasets, upstream
evaluation suites, and optional baseline implementations retain their own licenses and
terms.

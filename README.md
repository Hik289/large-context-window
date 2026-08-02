# UltraMem

**Source-resolved retrieval for persistent memory beyond the prompt**

<p align="center">
  <img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-blue">
  <img alt="Package version 0.1.0" src="https://img.shields.io/badge/version-0.1.0-6f42c1">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
</p>

UltraMem is a method artifact for agents that must reason over persistent document
collections without flattening those collections into a single prompt. It releases the
memory representation, retrieval path, source-resolution controls, context packing,
provenance checks, configuration templates, and credential-free tests needed to
instantiate the method on an authorized corpus.

The central design separates the representation used to find evidence from the text
allowed to support an answer. Each source span has a compact typed record for semantic
access and a source-text view for exact evidence, connected by immutable document
identifiers. Original and rewritten queries search both views; fused candidates are
resolved to source text, reranked, and packed into a bounded evidence context. Only
after generation does attribution identify the sources used by the answer.

<p align="center">
  <img src="assets/pipeline.png" alt="UltraMem maps ultra-large persistent memory through dual-view retrieval into a bounded source-faithful evidence context">
</p>

<p align="center"><em>UltraMem addresses the full memory while reading only the evidence needed for one answer.</em></p>

## Why UltraMem

A retrieval record is useful for finding evidence, but it is not necessarily safe to
quote as evidence. UltraMem therefore gives retrieval and generation different views
of the same source and reconnects them with immutable identifiers.

| Stage | Representation | Enforced contract |
| --- | --- | --- |
| Discover | Distilled records and source text | Search both views with original and rewritten queries |
| Resolve | Source identifiers | Map every distilled hit back to an authoritative source |
| Select | Ranked source-text candidates | Fuse, deduplicate, and rerank before loading evidence |
| Answer | Bounded detailed-evidence context | Generate only from loaded source evidence |
| Attribute | Fixed answer and loaded sources | Report support after generation without changing the answer |

## How It Scales

The figure separates four capacities that are often collapsed into a single context
window:

1. **Persistent-memory capacity** is the complete addressable collection—illustrated
   with policies, email, chat, tables, and code at scales from 10M to 250M tokens.
2. **Candidate capacity** is the set reached by multi-route recall. Original and
   rewritten queries search both the source-text and typed-memory views; identifier
   resolution reconnects every distilled hit to its source.
3. **Evidence-context capacity** is the small, fixed budget consumed by the answer
   model. Rank fusion and cross-encoder reranking decide which source-text chunks enter
   this budget.
4. **Reported-source capacity** is the still smaller set retained by post-answer
   attribution. It records which loaded documents support the completed answer rather
   than treating every retrieved candidate as a citation.

This separation is the key scaling principle: persistent memory may grow without
forcing the answer model to reread the full collection, while source resolution keeps
compressed retrieval records from becoming unsupported answer evidence.

## Method

The method treats long-context reasoning as a memory-system problem rather than a
larger-prompt problem. It assigns semantic access, source-faithful generation, and
attribution to separate stages, then reconnects them through explicit identifiers and
provenance records.

**Dual-view representation.** `DualNode` is the canonical memory unit. It carries a
distilled text field for low-cost routing, a detailed text or detail reference for
grounded reading, source evidence identifiers, lifecycle state, and model attribution.
The validation helpers enforce that every node is auditable before it enters a run.

**Hierarchical construction.** The builder converts document-level records into
source-linked memory nodes. Distillation is routed through named model aliases, while
detailed evidence remains anchored to the original span. This gives the system a compact
semantic surface without discarding the text needed for citation and inspection.

**Dual indexing.** `DualIndex` maintains separate distilled and detailed collections.
Original and rewritten queries search both collections. The distilled side broadens
semantic access; the source-text side preserves direct matches. Rank fusion combines
the routes without assuming comparable score scales.

**Source resolution and reranking.** Every distilled hit is mapped back to source text
before deduplication and cross-encoder reranking. The answer model therefore receives
verbatim evidence rather than compressed records that may have omitted qualifiers,
dates, exceptions, or conflicts.

**Budgeted packing.** The context packer admits the highest-ranked source-text chunks
under an explicit budget. The goal is not to maximize prompt volume; it is to provide
the answer model with a compact, inspectable working set.

**Provenance and accounting.** Citation extraction, source evidence identifiers, model
alias resolution, and token-ledger records make the method auditable. The code fails
loudly when required model aliases are missing or unresolved.

## Design Guarantees

- A dual-view memory schema that keeps distilled navigation and detailed evidence in
  one auditable node contract.
- A retrieval design that separates candidate discovery from source-text
  loading.
- An identifier-resolution path that maps distilled hits back to admissible answer
  evidence.
- A package-level configuration boundary that supports general chat-completions-style
  endpoints without committing provider-specific credentials.
- Credential-free tests for method invariants, package importability, configuration
  isolation, and token accounting.

## Installation

```bash
git clone https://github.com/Hik289/large-context-window.git
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

The editable core install is intentionally small. Add optional dependency groups only
for the capabilities you plan to use.

## Quick Start

The minimal example constructs a dual-view node and checks its representation and
provenance invariants. It requires no corpus, model endpoint, or API key.

```bash
python examples/minimal_contract.py
```

The same contract is available directly from Python:

```python
from agent_memory.methods import DualNode, validate_batch

node = DualNode(
    node_id="policy:001",
    level="L0",
    distilled_text="Travel reimbursement policy and approval rules.",
    detailed_text="Employees must submit receipts within 30 days...",
    distilled_tokens=7,
    detailed_tokens=42,
    source_evidence_ids=["policy:001#section-4"],
)

report = validate_batch([node])
assert report["overall_pass"]
```

## Local Verification

The following checks require no private corpus and no model credential:

```bash
agent-memory --version
python -m agent_memory --version
pytest
python -m agent_memory.methods.dual_node
python -m agent_memory.methods.token_ledger
python -m agent_memory.methods.configs.isolation
```

Run the complete credential-free test and package-build gate with `make check`.

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
- [Contribution guide](CONTRIBUTING.md)
- [Security policy](SECURITY.md)

## Repository Map

```text
large-context-window/
├── examples/                # zero-credential contract demonstration
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
source resolution, packing, and provenance. Deployment choices such as access control,
secret management, provider routing, and corpus governance remain the responsibility
of the deploying environment.

## License

The code is released under the [MIT License](LICENSE). External datasets, upstream
evaluation suites, and optional baseline implementations retain their own licenses and
terms.

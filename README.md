# Ultra-Large Memory Context Window Benchmark

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-3776AB">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
</p>

**Ultra-Large Memory Context Window Benchmark (ULMCW-Bench)** is a research
artifact for studying memory-augmented retrieval under enterprise-scale context
budgets. The repository provides the build pipeline, retrieval variants,
evaluation scripts, and figure-generation utilities used to compare raw
long-context retrieval against compressed, hierarchical memory systems.

<p align="center">
  <img src="assets/pipeline.png" alt="ULMCW-Bench pipeline" width="92%">
</p>

The benchmark treats long context as a resource that must be built, routed, and
audited. Documents are converted into raw evidence and distilled-memory streams;
queries retrieve from both views; the final reader is evaluated by answer
quality, token cost, build wall time, and citation traceability.

## Highlights

- **Ultra-large context regimes.** Corpus tiers cover tens to hundreds of
  millions of tokens, making the benchmark suitable for stress-testing memory
  construction and retrieval at realistic enterprise scale.
- **Dual-view memory.** The pipeline keeps a high-recall raw evidence stream and
  a compact distilled-memory stream, allowing token-efficiency studies without
  discarding source-grounded fallback evidence.
- **Hierarchical routing.** Section- and document-level summaries expose coarse
  memory states for routing, promotion, and retrieval analysis.
- **Reproducible evaluation.** Scripts report answer quality, retrieval quality,
  token usage, build wall time, and citation coverage from the same run outputs.
- **General API surface.** LLM calls are configured through generic
  chat-completions-style variables. The public configuration does not assume a
  particular hosted provider.

## Method

ULMCW-Bench instantiates a five-stage memory pipeline:

| Stage | Role |
|---|---|
| Corpus scaling | Build tiered corpora and question manifests for controlled context-window growth. |
| Memory distillation | Convert chunks into compact answer-bearing memory entries while retaining provenance. |
| Dual indexing | Maintain parallel raw-evidence and distilled-memory collections for recall and efficiency. |
| Adaptive retrieval | Expand, route, merge, and rerank candidates under a fixed token budget. |
| Citation-grounded evaluation | Score answers with matched evidence, token accounting, and traceability checks. |

The default build emits five collections:

| Collection | Description |
|---|---|
| `raw_chunks` | Chunk-level source evidence used as the high-recall stream. |
| `distilled_memory` | Compressed memory entries used as the low-token stream. |
| `cognitive` | Typed factual, procedural, definitional, requirement, decision, and constraint relations. |
| `section_summaries` | Section-level hierarchical memory nodes. |
| `doc_summaries` | Document-level hierarchical memory nodes. |

## Repository Layout

```text
enterprise-rag-bench/
├── assets/                  # README assets, including the generated pipeline figure
├── configs/                 # General model-routing schema and examples
├── figures/                 # Paper figure scripts and summary-plot generators
├── scripts/
│   ├── build/               # Corpus manifests, index construction, and run drivers
│   ├── eval/                # Official metrics and aggregation utilities
│   ├── external/            # External benchmark adapters and optional baselines
│   └── methods/             # Dual-index, hierarchy, and token-ledger implementations
├── src/agent_memory/
│   ├── document_eval/       # Build pipeline, retrieval, answering, and metrics
│   ├── retriever/           # Semantic, hybrid, plan-based, and reformulation retrievers
│   ├── builder/             # Document, chat, and email memory builders
│   ├── processors/          # PDF, Word, PowerPoint, Excel, Markdown, and text processors
│   ├── core/                # Memory stores, entries, filters, cues, and planners
│   ├── db_clients/          # ChromaDB and Redis backends
│   └── utils/               # Embeddings, logging, latency, and token accounting
├── .env.example
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Python 3.11 or newer is recommended. The default embedding path uses a local
sentence-transformers model, so a separate embedding service is not required for
smoke tests.

## Configuration

Copy the environment template and fill in a general chat API endpoint:

```bash
cp .env.example .env
```

Required variables:

| Variable | Meaning |
|---|---|
| `LLM_API_BASE` | Base URL for a chat-completions-compatible endpoint. |
| `LLM_API_KEY` | Secret key or bearer token for that endpoint. |
| `LLM_CHAT_MODEL` | Model identifier used for memory construction and answer generation. |
| `LLM_JUDGE_MODEL` | Model identifier used for judge-based evaluation. |
| `DATA_DIR` | Local root containing corpora, questions, manifests, and generated indices. |

Model aliases live in `configs/models.example.yaml`. Copy it to
`configs/models.yaml` when running new experiments:

```bash
cp configs/models.example.yaml configs/models.yaml
```

The resolver fails loudly when a required alias is missing. This keeps
large-scale runs auditable and prevents silent model substitution.

## Quick Start

Build or verify a tier manifest:

```bash
python scripts/build/build_erag_tier_verify.py --data-dir "$DATA_DIR"
```

Build the memory collections:

```bash
python scripts/build/run_phase21.py \
  --phase build \
  --docs-parquet "$DATA_DIR/docs.parquet" \
  --tier-manifest "$DATA_DIR/tier_manifest.parquet" \
  --questions-jsonl "$DATA_DIR/questions.jsonl" \
  --split-json "$DATA_DIR/splits.json" \
  --chroma-path "$DATA_DIR/chroma" \
  --output-root "$DATA_DIR/outputs"
```

Run evaluation:

```bash
python scripts/build/run_phase21.py \
  --phase eval \
  --docs-parquet "$DATA_DIR/docs.parquet" \
  --tier-manifest "$DATA_DIR/tier_manifest.parquet" \
  --questions-jsonl "$DATA_DIR/questions.jsonl" \
  --split-json "$DATA_DIR/splits.json" \
  --chroma-path "$DATA_DIR/chroma" \
  --output-root "$DATA_DIR/outputs" \
  --methods combined ddi hdm cdm
```

Compute official metrics from saved predictions:

```bash
python scripts/eval/metrics_based_eval.py \
  --predictions "$DATA_DIR/outputs/preds.json" \
  --gold "$DATA_DIR/questions.jsonl"
```

Regenerate paper figures:

```bash
cd figures
python make_all_figs.py --list
python make_all_figs.py
```

## Evaluation Outputs

Each run writes structured JSON summaries under the selected output directory.
The expected artifacts are:

| Output | Contents |
|---|---|
| `build_stats.json` | Documents processed, collection sizes, wall time, and extraction status. |
| `predictions.json` | Model answers, retrieved evidence, and citation metadata. |
| `metrics.json` | Answer quality, citation coverage, retrieval statistics, and token accounting. |
| `token_ledger.jsonl` | Per-stage input/output token usage for efficiency analysis. |

## Data

The repository does not ship corpora or generated indices. Prepare a local
`DATA_DIR` with:

- a document parquet file containing `doc_id`, `title`, `source_type`, and
  `content` columns;
- a question JSONL file with gold answers and expected evidence identifiers;
- optional tier manifests for fixed-scale comparisons;
- optional split metadata for development and test partitions.

Generated data, local indices, logs, and prediction files are ignored by git.

## Citation

```bibtex
@misc{ulmcwbench2026,
  title = {Ultra-Large Memory Context Window Benchmark},
  author = {Anonymous Authors},
  year = {2026},
  note = {Code and benchmark artifact}
}
```

## License

This project is released under the MIT License. See `LICENSE` for details.

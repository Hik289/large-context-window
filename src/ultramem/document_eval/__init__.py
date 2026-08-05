"""
ultramem.document_eval — Stage 1 Document algorithms (DDI / HDM / CDM / Combined).

Designed as a self-contained additive package that:
- Reuses ultramem.utils.{embedding, llm} for LLM + embedding clients
- Reuses ultramem.db_clients.chromadb_client (via direct chromadb usage)
- Does NOT modify ChatMemoryBuilder, LoCoMo config, or LoCoMo+ Cognitive Post-filter
- Stores all new ChromaDB collections under names containing "_doc_" to physically
  isolate from LoCoMo collections (regression protection)

Public entrypoints:
- DocumentBuildPipeline: ingest documents into all needed collections
- DocumentRetriever: Option-A architecture (DDI + HDM + CDM) with toggle flags
"""
from ultramem.document_eval.types import (
    RawChunkEntry,
    DistilledMemoryEntry,
    CognitiveEntry,
    SectionNode,
    DocumentNode,
    DocumentRetrievalConfig,
)
from ultramem.document_eval.pipeline import DocumentBuildPipeline
from ultramem.document_eval.retriever import DocumentRetriever

__all__ = [
    "RawChunkEntry",
    "DistilledMemoryEntry",
    "CognitiveEntry",
    "SectionNode",
    "DocumentNode",
    "DocumentRetrievalConfig",
    "DocumentBuildPipeline",
    "DocumentRetriever",
]

"""High-level local and remote memory client."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

from omegaconf import DictConfig

from agent_memory.builder.chat_memory_builder import NormalizedChatMessage
from agent_memory.builder.email_memory_builder import NormalizedEmail
from agent_memory.builder.memory_builder import MemoryBuilder
from agent_memory.core.local_client import LocalMemoryClient
from agent_memory.core.memory_entry import MemoryEntry
from agent_memory.core.remote_client import RemoteMemoryClient
from agent_memory.retriever.hybrid_retriever import HybridRetriever
from agent_memory.retriever.plan_based_retriever import PlanBasedRetriever
from agent_memory.retriever.prompted_policy_retriever import PromptedPolicyRetriever
from agent_memory.retriever.query_reformulation_retriever import QueryReformulationRetriever
from agent_memory.retriever.semantic_retriever import SemanticRetriever


_RETRIEVER_REGISTRY: Dict[str, type] = {
    "semantic": SemanticRetriever,
    "prompt": PromptedPolicyRetriever,
    "plan": PlanBasedRetriever,
    "reformulate": QueryReformulationRetriever,
    "hybrid": HybridRetriever,
}


class MemoryClient:
    """Facade over an in-process store or a configured remote memory service.

    Pass ``cfg`` and ``user_id`` for local operation. Pass ``api_key`` for
    remote add, query, and planner-query operations. Document processors and
    named retrieval strategies run locally because they require the local
    configuration and store.
    """

    def __init__(
        self,
        cfg: Optional[DictConfig] = None,
        user_id: Optional[str] = None,
        api_key: Optional[str] = None,
        server_url: Optional[str] = None,
    ) -> None:
        if api_key and (cfg is not None or user_id is not None):
            raise ValueError("Choose either local configuration or a remote API key.")
        if api_key:
            self._client: LocalMemoryClient | RemoteMemoryClient = RemoteMemoryClient(
                api_key=api_key,
                server_url=server_url,
            )
            return
        if cfg is None or not user_id:
            raise ValueError("Local MemoryClient requires both cfg and user_id.")
        self._client = LocalMemoryClient(cfg, user_id)

    @property
    def is_remote(self) -> bool:
        """Return whether this instance uses the remote service adapter."""
        return isinstance(self._client, RemoteMemoryClient)

    def _local(self, feature: str) -> LocalMemoryClient:
        if not isinstance(self._client, LocalMemoryClient):
            raise NotImplementedError(
                f"{feature} requires local mode; initialize MemoryClient with cfg and user_id."
            )
        return self._client

    def add(
        self,
        text: Union[str, List[str], List[Dict[str, str]]],
        metadata: Optional[Dict[str, Any]] = None,
        builder: Optional[Union[str, Type[MemoryBuilder], MemoryBuilder]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Any:
        """Persist text or structured messages."""
        if self.is_remote:
            if builder is not None or progress_callback is not None:
                raise ValueError("Custom builders and progress callbacks require local mode.")
            return self._client.add(text, metadata=metadata)
        return self._client.add(
            text,
            metadata=metadata,
            progress_callback=progress_callback,
            builder=builder,
        )

    def add_emails(self, emails: List[NormalizedEmail]) -> List[MemoryEntry]:
        """Extract and persist memories from an email thread."""
        return self._local("add_emails").add_emails(emails)

    def add_chats(self, messages: List[NormalizedChatMessage]) -> List[MemoryEntry]:
        """Extract and persist memories from a chat thread."""
        return self._local("add_chats").add_chats(messages)

    def add_file(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        builder: Optional[Union[str, Type[MemoryBuilder], MemoryBuilder]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[MemoryEntry]:
        """Process a supported file and persist its memories."""
        return self._local("add_file").add_file(
            file_path,
            metadata=metadata,
            builder=builder,
            progress_callback=progress_callback,
        )

    def query(
        self,
        context: Union[str, List[str], List[Dict[str, str]]],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
        enable_hybrid_search: bool = False,
        enable_llm_filter: bool = False,
        query_mode: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Return memories matching a query or structured context."""
        return self._client.query(
            context,
            top_k=top_k,
            where=where,
            include=include,
            enable_hybrid_search=enable_hybrid_search,
            enable_llm_filter=enable_llm_filter,
            query_mode=query_mode,
            **kwargs,
        )

    def planner_query(
        self,
        context: Union[str, List[str], List[Dict[str, str]]],
        top_k: int = 5,
        latency_tracker: Any = None,
    ) -> Any:
        """Run source-aware planner retrieval."""
        return self._client.planner_query(
            context,
            top_k=top_k,
            latency_tracker=latency_tracker,
        )

    def advanced_query(
        self,
        context: Union[str, List[str], List[Dict[str, str]]],
        top_k: int = 5,
        query_type: str = "prompt",
        latency_tracker: Any = None,
    ) -> List[MemoryEntry]:
        """Run one of the registered local retrieval strategies."""
        local_client = self._local("advanced_query")
        try:
            retriever_cls = _RETRIEVER_REGISTRY[query_type]
        except KeyError as exc:
            supported = ", ".join(sorted(_RETRIEVER_REGISTRY))
            raise ValueError(
                f"Unsupported query_type {query_type!r}. Choose one of: {supported}."
            ) from exc
        retriever = retriever_cls(local_client.cfg, memory_client=local_client)
        return retriever.retrieve(
            query=context,
            top_k=top_k,
            latency_tracker=latency_tracker,
        )

    def advance_query(
        self,
        context: Union[str, List[str], List[Dict[str, str]]],
        top_k: int = 5,
        query_type: str = "prompt",
        latency_tracker: Any = None,
    ) -> List[MemoryEntry]:
        """Compatibility alias for :meth:`advanced_query`."""
        return self.advanced_query(context, top_k, query_type, latency_tracker)

    def list_memories(self, limit: int = 20) -> List[MemoryEntry]:
        """List local memory records, capped by ``limit``."""
        return self._local("list_memories").list_memories(limit=limit)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Fetch a local record by its natural-language key."""
        return self._local("get").get(key)

    def delete(self, key: str) -> None:
        """Delete a local record by key."""
        self._local("delete").delete(key)

    def count(self) -> int:
        """Return the number of records in a local store."""
        return self._local("count").count()

    def clear(self) -> None:
        """Remove every record from a local store."""
        self._local("clear").clear()

    def delete_all(self, **kwargs: Any) -> None:
        """Delete local records matching the supplied filters."""
        self._local("delete_all").delete_all(**kwargs)

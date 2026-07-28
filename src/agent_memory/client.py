"""
MemoryClient — slim public facade. Either supply an API key (remote) or a
local config + user_id (in-process Chroma store).
"""

from pathlib import Path
from re import M
from typing import Any, Callable, Dict, List, Optional, Type, Union
from omegaconf import DictConfig

from agent_memory.builder.chat_memory_builder import NormalizedChatMessage
from agent_memory.builder.email_memory_builder import NormalizedEmail
from agent_memory.builder.memory_builder import MemoryBuilder
from agent_memory.core.memory_entry import MemoryEntry
from agent_memory.core.local_client import LocalMemoryClient
from agent_memory.core.remote_client import RemoteMemoryClient
from agent_memory.retriever.hybrid_retriever import HybridRetriever
from agent_memory.retriever.plan_based_retriever import PlanBasedRetriever
from agent_memory.retriever.prompted_policy_retriever import PromptedPolicyRetriever
from agent_memory.retriever.query_reformulation_retriever import QueryReformulationRetriever
from agent_memory.retriever.semantic_retriever import SemanticRetriever


# Lookup table for advance_query strategies; populated lazily inside the
# method so importing the module stays cheap.
_RETRIEVER_REGISTRY: Dict[str, Optional[type]] = {
    "semantic": SemanticRetriever,
    "prompt": PromptedPolicyRetriever,
    "plan": PlanBasedRetriever,
    "reformulate": QueryReformulationRetriever,
    "hybrid": HybridRetriever,
    "grpo": None,  # sentinel: not implemented yet
}


class MemoryClient:
    """Thin facade over local/remote memory backends.

    Notes for maintainers:
    - The default deployment uses the in-process Chroma-backed client.
    - When an API key is provided we authenticate against the remote service
      and the user_id is derived implicitly (never exposed to the caller).
    - Filters and metadata get the user_id stamped automatically.
    """

    def __init__(
        self,
        cfg: Optional[DictConfig] = None,
        user_id: str = None,
        api_key: Optional[str] = None,
    ):
        """
        Build the facade.

        Args:
            cfg: Configuration object
            api_key: API key for authentication (will derive user_id)
        """

        if api_key:
            self._client = RemoteMemoryClient(api_key=api_key)
        else:
            if not cfg or not user_id:
                raise ValueError(
                    "For local MemoryClient, both cfg and user_id must be provided."
                )
            self._client = LocalMemoryClient(cfg, user_id)

    def add(
        self,
        text: Union[str, List[str], List[Dict[str, str]]] = None,
        metadata: Optional[Dict] = None,        
        builder: Optional[Union[str, Type[MemoryBuilder], MemoryBuilder]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None, # for progress bar
    ) -> List[MemoryEntry]:
        """Persist memory content. user_id is stamped automatically.

        The duplicate per-user variant has been removed; user_id is always
        derived from the API key / local context. Stores or updates the
        record keyed by 'key'.

        Args:
            text: Text to add. Can be:
                - str: Natural language text
                - List[str]: Multiple text entries
                - List[Dict[str, str]]: Structured context with key-value pairs
            metadata: Additional metadata to store with the memory record
            progress_callback: Optional callback for progress updates
            builder: Optional custom builder. Can be:
                    - str: Builder type name (e.g., 'chat', 'markdown', 'doc')
                    - Type[MemoryBuilder]: MemoryBuilder class to instantiate
                    - MemoryBuilder: Custom MemoryBuilder instance
                    - None: Use default builder

        Returns:
            Record ID (derived from key)
        """
        return self._client.add(
            text, metadata=metadata, progress_callback=progress_callback, builder=builder
        )

    def add_emails(
        self,
        emails: List[NormalizedEmail],
    ) -> List[MemoryEntry]:
        """Extract memories from a single email or thread.

        Applies a thread-level relevance filter first, then runs extraction
        per message.

        Args:
            emails: List of emails in the thread (chronological).

        Returns:
            List of MemoryEntry objects created.
        """
        return self._client.add_emails(emails)

    def add_chats(
        self,
        messages: List[NormalizedChatMessage],
    ) -> List[MemoryEntry]:
        """Extract memories from a chat / Teams thread.

        Concatenates the messages, runs extraction, and emits source cues
        with metadata attached.

        Args:
            messages: List of chat messages in a single thread (chronological).

        Returns:
            List of MemoryEntry objects created.
        """
        return self._client.add_chats(messages)

    def add_file(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict] = None,
        builder: Optional[Union[str, Type[MemoryBuilder], MemoryBuilder]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[MemoryEntry]:
        """Read a file and persist its memories. user_id is stamped automatically.

        Mirrors :meth:`add` but pulls content from disk. The duplicate
        per-user variant has been removed.

        Args:
            file_path: Path to the file to add.
            metadata: Additional metadata to store with the memory record
            builder: Optional custom builder. Can be:
                    - str: Builder type name (e.g., 'chat', 'markdown', 'doc')
                    - Type[MemoryBuilder]: MemoryBuilder class to instantiate
                    - MemoryBuilder: Custom MemoryBuilder instance
                    - None: Auto-detect based on file type (default)
            progress_callback: Optional callback for progress updates (current, total, message)

        Returns:
            Record ID (derived from key)
        """
        return self._client.add_file(file_path, metadata=metadata, builder=builder, progress_callback=progress_callback)

    def query(
        self,
        context: Union[str, List[str], List[Dict[str, str]]],
        top_k: int = 5,
        where: Optional[Dict] = None,
        include: Optional[List[str]] = None,
        enable_hybrid_search: bool = False,
        enable_llm_filter: bool = False,
        query_mode = None,
        # filtering: bool = True,
        # enhance_query: bool = True,
        **kwargs,
    ) -> List[MemoryEntry]:
        """
        Vector similarity search by context.

        Args:
            context: Context to search for. Can be:
                - str: Natural language query text
                - List[str]: Multiple query strings
                - List[Dict[str, str]]: Structured context with key-value pairs
            k: Number of results to return
            where: Filter conditions for metadata-based filtering
            include: Fields to include in results (e.g., ["metadatas", "distances"])
            enable_hybrid_search: Whether to enable hybrid search combining semantic + keyword search
            enable_llm_filter: Whether to use LLM to filter irrelevant memories
            query_mode: Query mode (ORIGINAL, PRIMARY_ONLY, CUE_ONLY, or BOTH)
            filtering: Whether to apply additional filtering on the retrieved memories

        Returns:
            Backend-specific result object containing matching memories
        """
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
        latency_tracker=None,
    ) -> List[MemoryEntry]:
        """
        Source-aware retrieval driven by an LLM planner.

        Builds a small program from FILTER, RESOLVE and SEMANTIC_SEARCH
        primitives — useful when the question involves sender/recipient
        constraints, document scoping, or hierarchical drill-down.

        For plain semantic search prefer :meth:`query`.

        Args:
            context: Search context (str, List[str], or structured context)
            top_k: Maximum results to return
            latency_tracker: Optional latency tracker

        Returns:
            List of MemoryEntry objects
        """
        return self._client.planner_query(
            context,
            top_k=top_k,
            latency_tracker=latency_tracker,
        )

    def advance_query(
        self,
        context: Union[str, List[str], List[Dict[str, str]]],
        top_k: int = 5,
        query_type = "prompt",
        latency_tracker = None,
    ) -> List[MemoryEntry]:
        """
        Run a retrieval strategy by name.

        Args:
            context: Context to search for. Can be:
                - str: Natural language query text
                - List[str]: Multiple query strings
                - List[Dict[str, str]]: Structured context with key-value pairs
            top_k: Number of results to return
            query_type: Type of query strategy (e.g., "prompted_policy", "grpo")
            latency_tracker: Optional LatencyTracker for performance measurement

        Returns:
            Backend-specific result object containing matching memories
        """

        if query_type not in _RETRIEVER_REGISTRY:
            raise ValueError(f"Unsupported query_type: {query_type}")

        retriever_cls = _RETRIEVER_REGISTRY[query_type]
        if retriever_cls is None:
            # Currently a placeholder slot.
            raise NotImplementedError("GRPO query type is not implemented yet.")

        retriever = retriever_cls(self._client.cfg, memory_client=self._client)
        return retriever.retrieve(
            query=context,
            top_k=top_k,
            latency_tracker=latency_tracker,
        )
        

    def list_memories(self, limit: int = 20) -> List[MemoryEntry]:
        """
        List records in the current collection (capped by ``limit``).

        Args:
            limit: Maximum number of memories to return

        Returns:
            Dict with total count and list of memory records
        """
        return self._client.list_memories(limit=limit)

    def get(
        self,
        key: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch a single record by its natural-language key.

        Args:
            key: Natural language key to retrieve

        Returns:
            Dict with id, metadata, document fields or None if not found
        """
        return self._client.get(key)

    def delete(self, key: str) -> None:
        """
        Remove a record identified by its natural-language key.

        Args:
            key: Natural language key to delete
        """
        self._client.delete(key)

    def count(self) -> int:
        """
        Return the number of stored records.

        Returns:
            Total count of memory records
        """
        return self._client.count()

    def clear(self) -> None:
        """
        Remove every record in the current collection.
        """
        self._client.clear()

    def delete_all(self, **kwargs) -> None:
        """
        Bulk-delete records matching the given filter kwargs.
        """
        self._client.delete_all(**kwargs)

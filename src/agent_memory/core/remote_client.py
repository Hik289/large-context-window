import os
import requests
import json
from typing import Any, Dict, List, Optional, Union


class RemoteMemoryClient:
    """API-key authenticated client facade for memory operations.

    Internal rationale:
    - For internal deployment we always sit on top of the local Chroma store.
    - Callers supply only an API key; configuration is built behind the scenes.
    - The user_id derived from the API key is silently injected into metadata and filters.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
    ):
        """
        Construct the memory facade.

        Args:
            cfg: Configuration object
            api_key: API key for authentication (will derive user_id)
        """

        # Stash the API key.
        self.api_key = api_key

        # Resolve the remote service URL (env override, default to localhost).
        self.server_url = os.getenv("AGENT_MEMORY_REMOTE_SERVER_URL", "http://localhost:8000")

    def add(
        self,
        context: Union[str, List[str], List[Dict[str, str]]],
        metadata: Optional[Dict] = None,
    ) -> str:
        """Persist memory content; user_id is injected automatically.

        Removed duplicate variant: user_id is always derived from API key.
        Insert or update a memory record identified by 'key'.

        Args:
            context: Context to add. Can be:
                - str: Natural language text
                - List[str]: Multiple text entries
                - List[Dict[str, str]]: Structured context with key-value pairs
            metadata: Additional metadata to store with the memory record

        Returns:
            Record ID (derived from key)
        """

        if metadata is None:
            metadata = {}

        # Build the JSON body.
        payload = {
            "context": context,
            "metadata": metadata
        }

        # Headers, including bearer token authentication.
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else None
        }

        try:
            # POST the payload.
            response = requests.post(
                f"{self.server_url}/api/v1/memory/add",
                json=payload,
                headers=headers,
                timeout=30
            )

            # Surface non-2xx responses.
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to add memory: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid response format: {str(e)}")

    def query(
        self,
        context: Union[str, List[str], List[Dict[str, str]]],
        top_k: int = 5,
        where: Optional[Dict] = None,
        include: Optional[List[str]] = None,
        enable_hybrid_search: bool = False,
        **kwargs,
    ):
        """
        Vector search over remote memories.

        Args:
            context: Context to search for. Can be:
                - str: Natural language query text
                - List[str]: Multiple query strings
                - List[Dict[str, str]]: Structured context with key-value pairs
            k: Number of results to return
            where: Filter conditions for metadata-based filtering
            include: Fields to include in results (e.g., ["metadatas", "distances"])
            filtering: Whether to apply additional filtering on the retrieved memories

        Returns:
            Backend-specific result object containing matching memories
        """

        # Build the JSON body.
        payload = {
            "context": context,
            "top_k": top_k,
            "where": where,
            "include": include,
            "enable_hybrid_search": enable_hybrid_search,
            **kwargs
        }

        # Build headers (bearer auth).
        # TODO: replace the api_key in Authentication with proper bearer access token
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else None
        }

        try:
            # POST the request.
            response = requests.post(
                f"{self.server_url}/api/v1/memory/query",
                json=payload,
                headers=headers,
                timeout=30
            )

            # Surface non-2xx responses.
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to query memories: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid response format: {str(e)}")

    def planner_query(
        self,
        context: Union[str, List[str], List[Dict[str, str]]],
        top_k: int = 5,
        latency_tracker=None,
    ):
        """
        Run a planner-driven retrieval against the remote service.

        Args:
            context: Search context (str, list of strings, or structured)
            top_k: Maximum results to return
            latency_tracker: Ignored for remote client

        Returns:
            Response dict from the server
        """
        query_str = context if isinstance(context, str) else str(context)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else None,
        }
        try:
            response = requests.get(
                f"{self.server_url}/api/memory/planner_query",
                params={"q": query_str, "k": top_k},
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to execute planner query: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid response format: {str(e)}")

    # count, clear, delete_all unchanged (optional: restrict clear/delete_all to admin roles later)

    def get(
        self,
        key: str,
        user_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch a single record using its natural-language key.

        Args:
            key: Natural-language key to look up

        Returns:
            Dict containing id, metadata, document — or None when not found
        """
        raise NotImplementedError

    def delete(self, key: str) -> None:
        """
        Remove a record matching the given natural-language key.

        Args:
            key: Natural-language key to delete
        """
        raise NotImplementedError

    def count(self) -> int:
        """
        Number of memory records currently stored.

        Returns:
            Total count of memory records
        """
        raise NotImplementedError

    def clear(self) -> None:
        """
        Remove every record from the collection.
        """
        raise NotImplementedError

    def delete_all(self, **kwargs) -> None:
        """
        Remove every record matching the given parameters.
        """

        raise NotImplementedError

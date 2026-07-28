#!/usr/bin/env python3
"""
Chroma database inspector.

A small CLI utility for poking at the Chroma collection used by the
agent_memory memory system: list, search, and lookup memories from disk.
"""

import os
import json
import argparse
from typing import Optional, Dict, List, Any
import chromadb
from chromadb.config import Settings


class ChromaViewer:
    """Read-only helper for inspecting a Chroma persistent collection."""

    def __init__(self, persist_path: str, collection_name: Optional[str] = None):
        """Open the Chroma client at ``persist_path`` and (auto-)select a collection."""
        self.persist_path = persist_path

        # Open the persistent Chroma client.
        self.client = chromadb.PersistentClient(
            path=persist_path,
            settings=Settings(anonymized_telemetry=False)
        )

        # Pick a collection automatically if none was supplied.
        if collection_name is None:
            available = self.client.list_collections()
            if not available:
                raise ValueError("No collections found in database")

            names = [c.name for c in available]
            # Prefer the canonical name, otherwise just take the first one.
            collection_name = "agent_memory" if "agent_memory" in names else names[0]
            print(f"Auto-detected collection: {collection_name}")

        self.collection_name = collection_name

        try:
            self.collection = self.client.get_collection(collection_name)
            print(f"Successfully connected to collection: {collection_name}")
            print(f"Database path: {persist_path}")
        except Exception as exc:
            print(f"Error connecting to collection '{collection_name}': {exc}")
            available = self.client.list_collections()
            print(f"Available collections: {[c.name for c in available]}")
            raise

    def get_stats(self) -> Dict:
        """Return a small dict summarising the collection."""
        return {
            "total_memories": self.collection.count(),
            "collection_name": self.collection_name,
            "database_path": self.persist_path,
        }

    def list_all_memories(self, limit: Optional[int] = None, offset: int = 0) -> Dict:
        """Return all (or a paginated slice of) memories in the collection."""
        try:
            if limit:
                outcome = self.collection.get(limit=limit, offset=offset, include=["metadatas", "documents"])
            else:
                outcome = self.collection.get(include=["metadatas", "documents"])

            memories = []
            for i, (doc, meta) in enumerate(zip(outcome["documents"], outcome["metadatas"])):
                memories.append({
                    "index": offset + i,  # Account for pagination offset.
                    "id": outcome["ids"][i],
                    "memory": doc,
                    "metadata": meta,
                })

            return {
                "total_found": len(memories),
                "total_in_db": self.collection.count(),
                "offset": offset,
                "memories": memories,
            }
        except Exception as exc:
            print(f"Error listing memories: {exc}")
            return {"total_found": 0, "memories": []}

    def get_memories_by_indices(self, indices: List[int]) -> Dict:
        """Look up memories using positional indices into the full collection listing."""
        try:
            # Fetch every id once so we can map indices -> ids.
            all_ids = self.collection.get(include=[])["ids"]

            valid = [i for i in indices if 0 <= i < len(all_ids)]
            invalid = [i for i in indices if i not in valid]

            if not valid:
                return {
                    "requested_indices": indices,
                    "invalid_indices": invalid,
                    "total_found": 0,
                    "memories": [],
                }

            # Resolve indices -> id strings.
            target_ids = [all_ids[i] for i in valid]

            outcome = self.collection.get(ids=target_ids, include=["metadatas", "documents"])

            memories = []
            for i, mem_id in enumerate(target_ids):
                doc_pos = outcome["ids"].index(mem_id)
                memories.append({
                    "requested_index": valid[i],
                    "id": mem_id,
                    "memory": outcome["documents"][doc_pos],
                    "metadata": outcome["metadatas"][doc_pos],
                })

            return {
                "requested_indices": indices,
                "invalid_indices": invalid,
                "total_found": len(memories),
                "memories": memories,
            }
        except Exception as exc:
            print(f"Error getting memories by indices: {exc}")
            return {
                "requested_indices": indices,
                "total_found": 0,
                "memories": [],
                "error": str(exc),
            }

    def search_memories(self, query: str, top_k: int = 10, user_id: Optional[str] = None, use_semantic_search: bool = False) -> Dict:
        """Search memories by text. Defaults to substring-based search; optionally semantic."""
        where_clause = {"user_id": user_id} if user_id else None

        # Plain text search is the default path.
        if not use_semantic_search:
            return self._simple_text_search(query, top_k, where_clause)

        # Semantic search must be requested explicitly.
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_clause,
                include=["metadatas", "documents", "distances"],
            )

            memories = []
            if results["documents"] and results["documents"][0]:
                for i, (doc, meta, dist) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )):
                    memories.append({
                        "rank": i + 1,
                        "id": results["ids"][0][i],
                        "memory": doc,
                        "metadata": meta,
                        "similarity_score": round(1 - dist, 3),
                        "distance": round(dist, 3),
                    })

            return {
                "query": query,
                "search_type": "semantic",
                "total_found": len(memories),
                "memories": memories,
            }
        except Exception as exc:
            err_msg = str(exc)
            print(f"Error in semantic search: {err_msg}")
            print("Falling back to simple text search...")

            return self._simple_text_search(query, top_k, where_clause)

    def _simple_text_search(self, query: str, top_k: int, where_clause: Optional[Dict] = None) -> Dict:
        """Substring-based search across the document text and the ``value`` metadata field."""
        try:
            # Pull every document so we can scan locally.
            outcome = self.collection.get(
                where=where_clause,
                include=["metadatas", "documents"],
            )

            q_lower = query.lower()
            hits = []

            for i, (doc, meta) in enumerate(zip(outcome["documents"], outcome["metadatas"])):
                # Try matching the document text and the metadata "value" field.
                doc_lower = doc.lower()
                value_lower = ""
                where_matched = []

                if q_lower in doc_lower:
                    where_matched.append("memory")

                if meta and "value" in meta and meta["value"]:
                    value_lower = meta["value"].lower()
                    if q_lower in value_lower:
                        where_matched.append("value")

                if where_matched:
                    # Lightweight relevance heuristic: weighted token-frequency.
                    doc_score = doc_lower.count(q_lower) / len(doc_lower.split()) if doc_lower else 0
                    val_score = value_lower.count(q_lower) / len(value_lower.split()) if value_lower else 0

                    # Slight bias toward matches found in the memory body.
                    blended = (doc_score * 0.6) + (val_score * 0.4)

                    hits.append({
                        "rank": len(hits) + 1,
                        "id": outcome["ids"][i],
                        "memory": doc,
                        "metadata": meta,
                        "text_score": round(blended, 4),
                        "match_locations": where_matched,
                    })

            # Sort by score and trim to top_k, then renumber ranks.
            hits.sort(key=lambda x: x["text_score"], reverse=True)
            hits = hits[:top_k]

            for i, h in enumerate(hits):
                h["rank"] = i + 1

            return {
                "query": query,
                "search_type": "simple_text",
                "total_found": len(hits),
                "memories": hits,
            }
        except Exception as exc:
            print(f"Error in simple text search: {exc}")
            return {
                "query": query,
                "search_type": "simple_text",
                "total_found": 0,
                "memories": [],
                "error": str(exc),
            }

    def get_memory_by_id(self, memory_id: str) -> Optional[Dict]:
        """Return the single memory with the given id, or ``None`` if absent."""
        try:
            outcome = self.collection.get(ids=[memory_id], include=["metadatas", "documents"])
            if outcome["documents"]:
                return {
                    "id": memory_id,
                    "memory": outcome["documents"][0],
                    "metadata": outcome["metadatas"][0],
                }
            return None
        except Exception as exc:
            print(f"Error getting memory by ID: {exc}")
            return None

    def get_memories_by_user(self, user_id: str, limit: Optional[int] = None) -> Dict:
        """Return all memories belonging to ``user_id``."""
        try:
            kwargs = {
                "where": {"user_id": user_id},
                "include": ["metadatas", "documents"],
            }
            if limit:
                kwargs["limit"] = limit

            outcome = self.collection.get(**kwargs)

            memories = []
            for i, (doc, meta) in enumerate(zip(outcome["documents"], outcome["metadatas"])):
                memories.append({
                    "index": i,
                    "id": outcome["ids"][i],
                    "memory": doc,
                    "metadata": meta,
                })

            return {
                "user_id": user_id,
                "total_found": len(memories),
                "memories": memories,
            }
        except Exception as exc:
            print(f"Error getting memories for user {user_id}: {exc}")
            return {"user_id": user_id, "total_found": 0, "memories": []}

    def list_users(self) -> List[str]:
        """Return the sorted set of distinct ``user_id`` values present in the metadata."""
        try:
            outcome = self.collection.get(include=["metadatas"])
            users = set()
            for meta in outcome["metadatas"]:
                if "user_id" in meta:
                    users.add(meta["user_id"])
            return sorted(list(users))
        except Exception as exc:
            print(f"Error listing users: {exc}")
            return []

    def list_collections(self) -> List[Dict[str, Any]]:
        """Return ``(name, count)`` for every collection known to the client."""
        try:
            collections = self.client.list_collections()
            return [
                {"name": coll.name, "count": coll.count()}
                for coll in collections
            ]
        except Exception as exc:
            print(f"Error listing collections: {exc}")
            return []


def main():
    parser = argparse.ArgumentParser(description="View and inspect Chroma database contents")
    parser.add_argument("--db-path", required=True, help="Path to Chroma database")
    parser.add_argument("--collection", default=None, help="Collection name (auto-detects if not specified)")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("stats", help="Show database statistics")

    list_parser = subparsers.add_parser("list", help="List all memories")
    list_parser.add_argument("--limit", type=int, help="Limit number of results")
    list_parser.add_argument("--offset", type=int, default=0, help="Offset for pagination")

    search_parser = subparsers.add_parser("search", help="Search memories (uses simple text search by default)")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--topk", type=int, default=10, help="Number of results to return")
    search_parser.add_argument("--user-id", help="Filter by user ID")
    search_parser.add_argument("--semantic", action="store_true", help="Use semantic search instead of simple text matching")

    get_parser = subparsers.add_parser("get", help="Get memory by ID")
    get_parser.add_argument("memory_id", help="Memory ID to retrieve")

    indices_parser = subparsers.add_parser("indices", help="Get memories by index positions")
    indices_parser.add_argument("indices", nargs="+", type=int, help="Index positions to retrieve (e.g., 0 1 5 10)")

    user_parser = subparsers.add_parser("user", help="Get memories for a specific user")
    user_parser.add_argument("user_id", help="User ID")
    user_parser.add_argument("--limit", type=int, help="Limit number of results")

    subparsers.add_parser("users", help="List all user IDs")

    subparsers.add_parser("collections", help="List all collections in the database")

    args = parser.parse_args()

    db_path = args.db_path

    if not os.path.exists(db_path):
        print(f"Database path does not exist: {db_path}")
        return

    # Dispatch table for the parsed sub-command.
    try:
        viewer = ChromaViewer(db_path, args.collection)

        cmd = args.command

        if cmd == "stats":
            print(json.dumps(viewer.get_stats(), indent=2))

        elif cmd == "list":
            outcome = viewer.list_all_memories(args.limit, getattr(args, 'offset', 0))
            print(json.dumps(outcome, indent=2))

        elif cmd == "search":
            use_semantic = getattr(args, 'semantic', False)
            outcome = viewer.search_memories(args.query, args.topk, args.user_id, use_semantic)
            print(json.dumps(outcome, indent=2))

        elif cmd == "indices":
            outcome = viewer.get_memories_by_indices(args.indices)
            print(json.dumps(outcome, indent=2))

        elif cmd == "get":
            outcome = viewer.get_memory_by_id(args.memory_id)
            if outcome:
                print(json.dumps(outcome, indent=2))
            else:
                print(f"Memory with ID '{args.memory_id}' not found")

        elif cmd == "user":
            outcome = viewer.get_memories_by_user(args.user_id, args.limit)
            print(json.dumps(outcome, indent=2))

        elif cmd == "users":
            print(json.dumps({"users": viewer.list_users()}, indent=2))

        elif cmd == "collections":
            print(json.dumps({"collections": viewer.list_collections()}, indent=2))

        else:
            parser.print_help()

    except Exception as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()

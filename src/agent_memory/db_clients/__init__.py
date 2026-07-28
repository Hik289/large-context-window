"""Vector database client implementations.

Re-exports the concrete clients along with the abstract base class and
the configuration-driven factory.
"""
from agent_memory.db_clients.base import VectorDBClient
from agent_memory.db_clients.chromadb_client import ChromaDBClient
from agent_memory.db_clients.redis_client import RedisVectorDBClient
from agent_memory.db_clients.factory import create_vector_db_client

__all__ = [
    "VectorDBClient",
    "ChromaDBClient",
    "RedisVectorDBClient",
    "create_vector_db_client",
]

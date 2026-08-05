"""Vector database client implementations.

Re-exports the concrete clients along with the abstract base class and
the configuration-driven factory.
"""
from ultramem.db_clients.base import VectorDBClient
from ultramem.db_clients.chromadb_client import ChromaDBClient
from ultramem.db_clients.redis_client import RedisVectorDBClient
from ultramem.db_clients.factory import create_vector_db_client

__all__ = [
    "VectorDBClient",
    "ChromaDBClient",
    "RedisVectorDBClient",
    "create_vector_db_client",
]

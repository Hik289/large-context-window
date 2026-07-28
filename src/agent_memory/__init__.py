"""
agent_memory — high-level memory layer for AI assistants.

Bundles storage, retrieval and analysis primitives behind a small public
surface. Re-exports the ``browser`` and ``utils`` subpackages.
"""

from . import browser
from . import utils

__version__ = "0.1.0"
__all__ = ["browser", "utils"]

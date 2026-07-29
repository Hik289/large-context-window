"""Public package surface for the large-context agent-memory artifact."""

from importlib import import_module
from typing import Any

__version__ = "0.1.0"
__all__ = ["browser", "methods", "utils", "__version__"]


def __getattr__(name: str) -> Any:
    """Load optional subpackages only when requested.

    Keeping the top-level import light makes version checks, packaging, and the
    representation-level tests independent of optional retrieval backends.
    """
    if name in {"browser", "methods", "utils"}:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Public package surface for UltraMem."""

from importlib import import_module
from typing import Any

__version__ = "0.1.0"
__all__ = [
    "DualIndex",
    "DualNode",
    "DualNodeError",
    "MemoryClient",
    "TokenLedger",
    "browser",
    "methods",
    "utils",
    "validate_batch",
    "validate_one",
    "__version__",
]

_EXPORTS = {
    "DualIndex": ("agent_memory.methods", "DualIndex"),
    "DualNode": ("agent_memory.methods", "DualNode"),
    "DualNodeError": ("agent_memory.methods", "DualNodeError"),
    "MemoryClient": ("agent_memory.client", "MemoryClient"),
    "TokenLedger": ("agent_memory.methods", "TokenLedger"),
    "validate_batch": ("agent_memory.methods", "validate_batch"),
    "validate_one": ("agent_memory.methods", "validate_one"),
}


def __getattr__(name: str) -> Any:
    """Load optional subpackages only when requested.

    Keeping the top-level import light makes version checks, packaging, and the
    representation-level tests independent of optional retrieval backends.
    """
    if name in {"browser", "methods", "utils"}:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    if name in _EXPORTS:
        module_name, attribute = _EXPORTS[name]
        value = getattr(import_module(module_name), attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

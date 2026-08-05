"""Public package surface for UltraMem."""

from importlib import import_module
from typing import Any

__version__ = "0.1.0"
# Star imports expose only interfaces supported by the core installation.
__all__ = [
    "DualNode",
    "DualNodeError",
    "MemoryClient",
    "TokenLedger",
    "validate_batch",
    "validate_one",
    "__version__",
]

_EXPORTS = {
    "DualIndex": ("ultramem.methods", "DualIndex"),
    "DualNode": ("ultramem.methods", "DualNode"),
    "DualNodeError": ("ultramem.methods", "DualNodeError"),
    "MemoryClient": ("ultramem.client", "MemoryClient"),
    "TokenLedger": ("ultramem.methods", "TokenLedger"),
    "validate_batch": ("ultramem.methods", "validate_batch"),
    "validate_one": ("ultramem.methods", "validate_one"),
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


def __dir__() -> list[str]:
    """List both core exports and lazily available optional interfaces."""
    return sorted(set(globals()) | set(_EXPORTS) | {"browser", "methods", "utils"})

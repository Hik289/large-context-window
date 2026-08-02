"""Model alias resolver for UltraMem.

The resolver maps each alias in models.yaml to one concrete general chat API
model. Missing aliases and unresolved providers fail loudly so large runs cannot
silently drift across model families.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


class ModelAliasError(RuntimeError):
    pass


CONFIG_DIR = Path(__file__).parent
MODELS_YAML = CONFIG_DIR / "models.yaml"

_CACHE: Dict[str, Any] = {}


def _load_models_yaml() -> Dict[str, Any]:
    global _CACHE
    if _CACHE:
        return _CACHE
    with open(MODELS_YAML) as f:
        _CACHE = yaml.safe_load(f) or {}
    return _CACHE


def resolve(alias: str) -> Dict[str, Any]:
    """Return the spec for an alias."""
    cfg = _load_models_yaml()
    aliases = cfg.get("aliases", {})
    if alias not in aliases:
        raise ModelAliasError(
            f"unknown model alias {alias!r}. Known: {sorted(aliases.keys())}"
        )
    spec = aliases[alias]
    if str(spec.get("provider", "")).upper() == "UNRESOLVED":
        raise ModelAliasError(
            f"alias {alias!r} is UNRESOLVED. "
            f"Blocker: {spec.get('blocker', 'unspecified')}."
        )
    if str(spec.get("provider", "")).lower() != "general":
        raise ModelAliasError(
            f"alias {alias!r} uses provider={spec.get('provider')!r}; "
            "only provider='general' is accepted by this artifact."
        )
    return spec


def assert_active(alias: str) -> None:
    """Raise if an alias is unavailable."""
    resolve(alias)


def _env_value(spec: Dict[str, Any], key: str, default: str = "") -> str:
    env_name = spec.get(key, "")
    return os.environ.get(env_name, default) if env_name else default


def smoke_test_general(
    alias: str,
    spec: Dict[str, Any],
    message: str = "Say 'pong' and nothing else.",
) -> Dict[str, Any]:
    """Perform a one-message chat API smoke test for an alias."""
    from openai import OpenAI

    base_url = spec.get("base_url") or _env_value(spec, "base_url_env")
    api_key = spec.get("api_key") or _env_value(spec, "api_key_env")
    model = spec.get("model", "")
    if not base_url or not api_key:
        raise ModelAliasError(
            f"no API base/key for {alias} "
            f"(base_env={spec.get('base_url_env')}, key_env={spec.get('api_key_env')})"
        )
    if not model or model.startswith("YOUR_"):
        raise ModelAliasError(f"no concrete model configured for alias {alias}")

    client = OpenAI(base_url=base_url, api_key=api_key)
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "temperature": 0.0,
    }
    token_param = spec.get("token_parameter", "max_tokens")
    kwargs[token_param] = 20

    t0 = time.time()
    resp = client.chat.completions.create(**kwargs)
    latency = time.time() - t0
    text = resp.choices[0].message.content or ""
    return {
        "alias": alias,
        "provider": "general",
        "model": model,
        "base_url": base_url,
        "reply": text,
        "latency_seconds": round(latency, 3),
        "ok": True,
    }


def smoke_test(alias: str) -> Dict[str, Any]:
    spec = resolve(alias)
    return smoke_test_general(alias, spec)


def list_aliases() -> Dict[str, str]:
    """Return {alias: status} for diagnostics."""
    cfg = _load_models_yaml()
    out = {}
    for alias, spec in cfg.get("aliases", {}).items():
        if str(spec.get("provider", "")).upper() == "UNRESOLVED":
            out[alias] = f"UNRESOLVED ({spec.get('blocker', '?')})"
        else:
            out[alias] = f"{spec.get('provider')}::{spec.get('model')}"
    return out

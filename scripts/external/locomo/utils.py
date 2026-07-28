import time
import json

import requests
from requests.exceptions import RequestException
from omegaconf import DictConfig

TECHNIQUES = ["mem0", "rag", "langmem", "zep", "openai"]

METHODS = ["add", "search"]


def init_mem0_client(cfg: DictConfig):
    """
    Build a mem0 ``Memory`` client wired up with the general API configuration.

    Args:
        cfg: An OmegaConf ``DictConfig`` carrying model settings.

    Returns:
        A ``Memory`` client instance.
    """
    from mem0 import Memory

    api_base = cfg.openai.llm_api_base
    managed_identity = cfg.openai.managed_identity
    model = cfg.openai.model
    embedding_model = cfg.openai.embedding_model

    # Wire LLM, embedder and vector store through the configured API.
    config = {
        "llm": {
            "provider": "general",
            "config": {
                "model": model,
                "temperature": 0.1,
                "max_tokens": 2000,
                # All settings inline; no env-var lookups
                "api_kwargs": {
                    "base_url": api_base,
                    "client_id": managed_identity,
                },
            },
        },
        "embedder": {
            "provider": "general",
            "config": {
                "model": embedding_model,
                "api_kwargs": {
                    "base_url": api_base,
                    "client_id": managed_identity,
                },
            },
        },
        # Vector store
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": "agent_memory",
                "path": cfg.memory.persist_path,
                # ChromaDB Cloud (optional):
                # "api_key": "your-chroma-cloud-api-key",
                # "tenant": "your-chroma-cloud-tenant-id",
            },
        },
    }

    client = Memory.from_config(config)
    return client


def load_data(data_path: str, subset_idx: int = -1) -> list:

    with open(data_path, "r") as fh:
        data = json.load(fh)

    if subset_idx > 0:
        data = data[subset_idx - 1 : subset_idx]

    return data


def generate_debug_data(data, conversion_idx=0, session_idx=1, num_sessions=3) -> list:
    """
    Build a small slice of *data* for debugging.

    Args:
        data (list): Original dataset.
        conversion_idx (int, optional): Defaults to 0.
        session_idx (int, optional): Defaults to 1.
        num_sessions (int, optional): Defaults to 1.

    Returns:
        list: Subset of the original data suited to debugging.
    """
    data = data[: conversion_idx + 1]

    # Keep only the first 10 questions
    data[0]["qa"] = data[0]["qa"][:10]

    # Take the first ``num_sessions`` sessions
    conversation = data[0]["conversation"]
    new_conversation = {
        "speaker_a": conversation["speaker_a"],
        "speaker_b": conversation["speaker_b"],
    }

    # Resolve the requested session indices
    if isinstance(session_idx, int):
        indices = [session_idx]
    elif isinstance(session_idx, str):
        indices = [int(piece) for piece in session_idx.split(",")]
    else:
        raise ValueError("session_idx must be int or comma-separated string")

    for start in indices:
        for offset in range(start, start + num_sessions):
            key = f"session_{offset}"
            if key in conversation:
                new_conversation[key] = conversation[key]
                new_conversation[f"{key}_date_time"] = conversation[f"{key}_date_time"]
    data[0]["conversation"] = new_conversation
    return data


def measure_execution_time(func, *args, **kwargs):
    """
    Helper that measures how long a function takes.

    Args:
        func: Callable to invoke.
        *args: Positional args forwarded to *func*.
        **kwargs: Keyword args forwarded to *func*.

    Returns:
        tuple: ``(result, duration_in_seconds)``.
    """
    started_at = time.time()
    outcome = func(*args, **kwargs)
    finished_at = time.time()
    duration = finished_at - started_at
    return outcome, duration


def format_duration(duration: float) -> str:
    """
    Format *duration* as a simple seconds string.

    Args:
        duration: Duration in seconds.

    Returns:
        Formatted duration string, e.g. ``"3.21 seconds"``.
    """
    return f"{duration:.2f} seconds"


def get_session_num(conversation):
    last = 0
    for idx in range(1, 100):
        key = f"session_{idx}"
        if key not in conversation:
            break
        last += 1
    return last


def is_valid_image_url(url: str, timeout: int = 5) -> bool:
    """
    Probe an image URL and report whether it is reachable.

    Args:
        url: The image URL to validate.
        timeout: Per-request timeout in seconds.

    Returns:
        bool: ``True`` when the URL is reachable, otherwise ``False``.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        # Try HEAD first (cheaper than GET)
        response = requests.head(url, timeout=timeout, allow_redirects=True, headers=headers)

        # Some servers reject HEAD — fall back to streaming GET
        if response.status_code == 405 or response.status_code == 403:
            response = requests.get(url, timeout=timeout, allow_redirects=True, headers=headers, stream=True)
            # Close the connection right after grabbing the headers
            response.close()

        # Treat as success if both status and content-type look right
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')
            # Either Content-Type starts with ``image/`` or the URL has a known image extension
            is_image_type = content_type.startswith('image/')
            is_image_extension = url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'))
            return is_image_type or is_image_extension
        return False
    except RequestException as exc:
        print(f"Error validating URL {url}: {exc}")
        return False

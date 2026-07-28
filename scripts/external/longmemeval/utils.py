import time
import json

import requests
from requests.exceptions import RequestException
from omegaconf import DictConfig

METHODS = ["add", "search"]



def measure_execution_time(func, *args, **kwargs):
    """
    Run ``func`` and return its result alongside the wall-clock time it took
    to complete (in seconds).

    Args:
        func: Callable to invoke.
        *args: Positional arguments forwarded to ``func``.
        **kwargs: Keyword arguments forwarded to ``func``.

    Returns:
        Tuple ``(value, seconds)`` where ``value`` is whatever ``func``
        returned and ``seconds`` is the elapsed time.
    """
    started = time.time()
    outcome = func(*args, **kwargs)
    elapsed = time.time() - started
    return outcome, elapsed


def format_duration(duration: float) -> str:
    """
    Render a duration as a short human-readable string (in seconds).

    Args:
        duration: Number of seconds.

    Returns:
        A formatted string such as ``"1.23 seconds"``.
    """
    return f"{duration:.2f} seconds"


def get_session_num(conversation):
    """Count how many ``session_<n>`` keys are present in ``conversation``."""
    count = 0
    for pos in range(1, 100):
        if f"session_{pos}" not in conversation:
            break
        count += 1
    return count

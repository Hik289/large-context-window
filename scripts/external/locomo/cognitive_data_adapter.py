"""
Adapter that loads cognitive evaluation data from ``unified_input_samples_v2.json``
and links each item back to its locomo10 conversation.

Each cognitive item exercises whether the memory system can surface the right
historical context when a fresh conversational trigger is provided. Fields:
- ``trigger``: a new utterance from a speaker (used as the retrieval query)
- ``evidence``: the past dialogue the trigger implicitly references (ground truth)
- ``time_gap``: the time gap between the trigger and the original conversation

On first use the v2 JSON (~223 MB) is parsed in full and items are mapped to
locomo10 conversations. The result is cached so subsequent runs skip the heavy
parse.

``build_augmented_locomo10`` stitches every cognitive evidence turn back into
the original locomo10 conversations, so memory stores built from the augmented
data contain the dialogue the cognitive questions actually depend on.
"""

import copy
import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CACHE_FILENAME = "cognitive_items_cache.json"


@dataclass
class CognitiveItem:
    """One cognitive evaluation item, mapped to a locomo10 conversation."""
    trigger: str
    evidence: str
    time_gap: str
    speaker_a: str
    speaker_b: str
    conv_idx: int
    raw_trigger: str
    evidence_after_session: int = -1  # session to append evidence after; -1 = last

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CognitiveItem":
        return cls(**d)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _load_locomo10(locomo10_path: str) -> list:
    with open(locomo10_path, "r") as fh:
        return json.load(fh)


def _build_speaker_mapping(
    locomo10: list,
) -> Dict[frozenset, Tuple[str, str, int, dict]]:
    """Map ``frozenset({speaker_a, speaker_b})`` to ``(spk_a, spk_b, idx, conv)``."""
    mapping: Dict[frozenset, Tuple[str, str, int, dict]] = {}
    for pos, entry in enumerate(locomo10):
        conv = entry["conversation"]
        spk_a = conv["speaker_a"]
        spk_b = conv["speaker_b"]
        mapping[frozenset({spk_a, spk_b})] = (spk_a, spk_b, pos, conv)
    return mapping


def _find_evidence_session(
    input_prompt: str, evidence: str, conversation: dict,
) -> int:
    """Identify which locomo10 session the evidence was stitched after.

    The v2 pipeline injects cue dialogue between two existing sessions. We
    locate the evidence inside ``input_prompt``, look at the line right above
    it, and match that line to the last turn of one of the sessions in
    ``conversation``.

    Returns the 1-based session index, or ``-1`` when detection fails (the
    caller is expected to fall back to the final session).
    """
    prompt_lines = input_prompt.strip().split("\n")
    ev_lines = [ln.strip() for ln in evidence.strip().split("\n") if ln.strip()]
    if not ev_lines:
        return -1

    head = ev_lines[0]
    head_text = head.split("：", 1)[1].strip() if "：" in head else head

    ev_pos = None
    for idx, line in enumerate(prompt_lines):
        if head_text[:40] in line:
            ev_pos = idx
            break

    if ev_pos is None:
        return -1

    # Evidence at the very top of the conversation → prepend before session 1
    if ev_pos == 0:
        return 0

    prev_line = prompt_lines[ev_pos - 1]
    # input_prompt lines are formatted as: 'Speaker said, "text"' — pull out the text
    quoted = re.match(r'\w+ said, "(.+)"$', prev_line)
    prev_text = quoted.group(1) if quoted else prev_line

    n_sessions = _get_session_num(conversation)
    for s in range(1, n_sessions + 1):
        session_key = f"session_{s}"
        if session_key not in conversation:
            continue
        last_text = conversation[session_key][-1]["text"]
        if prev_text[:40] in last_text or last_text[:40] in prev_text:
            return s

    return -1


def _extract_speakers_from_prompt(input_prompt: str) -> frozenset:
    """Pull the two speaker names from the opening lines of *input_prompt*."""
    found = set()
    for hit in re.finditer(r'^(\w+) said,', input_prompt, re.MULTILINE):
        found.add(hit.group(1))
        if len(found) >= 2:
            break
    return frozenset(found)


def _parse_trigger_text(trigger: str) -> str:
    """Drop the leading 'A: ' role marker if present."""
    return trigger[3:] if trigger.startswith("A: ") else trigger


# ------------------------------------------------------------------
# Cache I/O
# ------------------------------------------------------------------

def _save_cache(items: List[CognitiveItem], cache_path: str) -> None:
    """Write processed items to disk so future runs skip the heavy v2 parse."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as fh:
        json.dump([entry.to_dict() for entry in items], fh)
    logger.info(f"Cached {len(items)} cognitive items to {cache_path}")


def _load_cache(cache_path: str) -> List[CognitiveItem]:
    """Load previously-cached cognitive items from *cache_path*."""
    with open(cache_path, "r") as fh:
        raw = json.load(fh)
    items = [CognitiveItem.from_dict(d) for d in raw]
    logger.info(f"Loaded {len(items)} cognitive items from cache ({cache_path})")
    return items


def _cache_is_fresh(cache_path: str, *source_paths: str) -> bool:
    """``True`` iff the cache exists and is newer than every source file."""
    if not os.path.exists(cache_path):
        return False
    cache_mtime = os.path.getmtime(cache_path)
    for src in source_paths:
        if os.path.exists(src) and os.path.getmtime(src) > cache_mtime:
            return False
    return True


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def load_cognitive_data(
    v2_path: str,
    locomo10_path: str,
    cache_dir: Optional[str] = None,
    force_reload: bool = False,
) -> List[CognitiveItem]:
    """Load cognitive items, leveraging a cache file when available.

    On the first call we parse the full v2 JSON, map every item to its
    locomo10 conversation and persist the result to
    ``<cache_dir>/cognitive_items_cache.json``. Later calls just read the
    cache (< 1 MB versus ~223 MB).

    The cache is invalidated automatically whenever either source file is
    modified after the cache was written. Pass ``force_reload=True`` to
    skip the cache.

    Args:
        v2_path: Path to ``unified_input_samples_v2.json``.
        locomo10_path: Path to ``locomo10.json``.
        cache_dir: Directory for the cache file. Defaults to the parent
            directory of *v2_path*.
        force_reload: When ``True``, ignore any cache and rebuild.

    Returns:
        A list of ``CognitiveItem`` objects already mapped to a conversation.
    """
    if cache_dir is None:
        cache_dir = os.path.dirname(v2_path)
    cache_path = os.path.join(cache_dir, CACHE_FILENAME)

    if not force_reload and _cache_is_fresh(cache_path, v2_path, locomo10_path):
        return _load_cache(cache_path)

    logger.info("Cache miss — parsing v2 data and mapping to locomo10 conversations …")
    locomo10 = _load_locomo10(locomo10_path)
    speaker_mapping = _build_speaker_mapping(locomo10)

    with open(v2_path, "r") as fh:
        v2_data = json.load(fh)

    cognitive_raw = [entry for entry in v2_data if entry.get("category") == "Cognitive"]
    logger.info(f"Found {len(cognitive_raw)} cognitive items in v2 data")

    items: List[CognitiveItem] = []
    unmapped = 0
    session_detected = 0
    for raw_item in cognitive_raw:
        speakers = _extract_speakers_from_prompt(raw_item["input_prompt"])

        if speakers not in speaker_mapping:
            unmapped += 1
            logger.warning(f"Could not map speakers {speakers} to locomo10 conversation")
            continue

        spk_a, spk_b, conv_idx, conversation = speaker_mapping[speakers]
        target_session = _find_evidence_session(
            raw_item["input_prompt"], raw_item["evidence"], conversation,
        )
        if target_session > 0:
            session_detected += 1

        items.append(CognitiveItem(
            trigger=_parse_trigger_text(raw_item["trigger"]),
            evidence=raw_item["evidence"],
            time_gap=raw_item.get("time_gap", ""),
            speaker_a=spk_a,
            speaker_b=spk_b,
            conv_idx=conv_idx,
            raw_trigger=raw_item["trigger"],
            evidence_after_session=target_session,
        ))

    if unmapped:
        logger.warning(f"{unmapped} cognitive items could not be mapped")

    n_convs = len({entry.conv_idx for entry in items})
    logger.info(f"Loaded {len(items)} cognitive items across {n_convs} conversations")
    logger.info(
        f"Evidence session detected for {session_detected}/{len(items)} items "
        f"(rest will fall back to last session)"
    )

    _save_cache(items, cache_path)

    return items


def get_user_ids(
    item: CognitiveItem, use_combined_user: bool,
) -> Tuple[str, Optional[str]]:
    """Return user IDs that match the convention used during memory building."""
    if use_combined_user:
        return f"{item.speaker_a}_{item.speaker_b}_{item.conv_idx}", None
    return f"{item.speaker_a}_{item.conv_idx}", f"{item.speaker_b}_{item.conv_idx}"


def parse_evidence_lines(evidence: str) -> List[Tuple[str, str]]:
    """Split the evidence field into ``(speaker, text)`` pairs.

    Evidence uses a full-width colon (``：``) as separator, e.g.::

        Caroline：After learning to say 'no', I've felt a lot less stressed.
        Melanie：That's a great skill to develop; protecting your time is important.

    Returns:
        Ordered list of ``(speaker_name, dialogue_text)`` tuples.
    """
    pairs: List[Tuple[str, str]] = []
    for raw in evidence.strip().split("\n"):
        raw = raw.strip()
        if not raw:
            continue
        if "：" in raw:
            spk, body = raw.split("：", 1)
            pairs.append((spk.strip(), body.strip()))
        else:
            pairs.append(("", raw))
    return pairs


# ------------------------------------------------------------------
# Augmented locomo10 builder
# ------------------------------------------------------------------

def _get_session_num(conversation: dict) -> int:
    """Return the highest session index present inside *conversation*."""
    last = 0
    for idx in range(1, 100):
        if f"session_{idx}" not in conversation:
            break
        last = idx
    return last


def build_augmented_locomo10(
    locomo10_path: str,
    items: List[CognitiveItem],
    output_path: str,
) -> str:
    """Write an augmented copy of locomo10 with cognitive evidence stitched in.

    Evidence turns are inserted at the **same session position** the v2
    pipeline originally used (taken from ``evidence_after_session`` on each
    item). When detection failed (``-1``) the evidence falls back to the
    last session. Duplicate evidence lines (same speaker + text within the
    same target session) are dropped.

    Args:
        locomo10_path: Path to the original ``locomo10.json``.
        items: Already-loaded list of ``CognitiveItem`` objects.
        output_path: Where to write the augmented JSON file.

    Returns:
        ``output_path`` (returned for convenience).
    """
    with open(locomo10_path, "r") as fh:
        locomo10 = json.load(fh)

    augmented = copy.deepcopy(locomo10)

    # Group unique evidence turns by (conv_idx, target_session).
    # key = (conv_idx, session_num) → ordered list of (speaker, text)
    session_evidence: Dict[Tuple[int, int], List[Tuple[str, str]]] = defaultdict(list)
    seen: Dict[Tuple[int, int], set] = defaultdict(set)

    for item in items:
        conv = augmented[item.conv_idx]["conversation"]
        target_session = item.evidence_after_session
        if target_session < 0:
            target_session = _get_session_num(conv)

        for spk, body in parse_evidence_lines(item.evidence):
            dedup_key = (spk, body)
            bucket = (item.conv_idx, target_session)
            if dedup_key in seen[bucket]:
                continue
            seen[bucket].add(dedup_key)
            session_evidence[bucket].append((spk, body))

    total_added = 0
    for (conv_idx, session_num), pairs in session_evidence.items():
        conv = augmented[conv_idx]["conversation"]

        if session_num == 0:
            # Evidence preceding session_1 → prepend
            session_key = "session_1"
            insert_pos = 0
        else:
            session_key = f"session_{session_num}"
            if session_key not in conv:
                session_key = f"session_{_get_session_num(conv)}"
            insert_pos = len(conv[session_key])  # append

        for offset, (spk, body) in enumerate(pairs):
            turn = {
                "speaker": spk,
                "text": body,
                "dia_id": f"cognitive_evidence_{conv_idx}_{total_added}",
            }
            if insert_pos == 0:
                conv[session_key].insert(offset, turn)
            else:
                conv[session_key].append(turn)
            total_added += 1

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(augmented, fh, indent=2)

    n_slots = len(session_evidence)
    n_convs = len({ci for ci, _ in session_evidence})
    logger.info(
        f"Augmented locomo10 written to {output_path} "
        f"({total_added} evidence turns across {n_slots} session slots "
        f"in {n_convs} conversations)"
    )
    return output_path

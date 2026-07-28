"""
Retrieval recall metrics shared by locomo10 QA and the cognitive evaluation.

We compute two complementary signals:
  - **session_recall** (binary): did *any* retrieved memory come from the same
    conversation session as the ground-truth evidence?
  - **text_recall** (0–1): of the ground-truth evidence turns, what fraction
    were matched by at least one retrieved memory?

Both pipelines (``run_experiments`` / ``run_cognitive_eval``) call into this
module so the metric definitions stay aligned.
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Cosine-similarity threshold above which an evidence turn counts as recalled.
DEFAULT_SIMILARITY_THRESHOLD = 0.8

# Matches a "SpeakerName: " prefix at the start of a line (greedy until first colon).
_SPEAKER_PREFIX_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_ ]*:\s*", re.MULTILINE)

# Matches "[timestamp]\n" blocks injected by ``update_memory``.
_TIMESTAMP_BRACKET_RE = re.compile(r"\[[^\[\]]{4,60}\]\n?")


def strip_metadata(text: str) -> str:
    """Drop speaker prefixes and ``[timestamp]`` blocks from *text*.

    Both evidence dialogue (which carries ``"Speaker: "`` prefixes) and
    memory values (which may carry ``"Speaker: "`` prefixes plus ``[timestamp]``
    blocks merged in by ``update_memory``) need this so comparison focuses on
    actual content.
    """
    text = _TIMESTAMP_BRACKET_RE.sub("", text)
    text = _SPEAKER_PREFIX_RE.sub("", text)
    return text.strip()


def _get_sentence_model():
    """Reuse the lazily-loaded SentenceTransformer from ``metrics.utils``."""
    from metrics.utils import _get_sentence_model
    return _get_sentence_model()


# ------------------------------------------------------------------
# Locomo10 evidence reference helpers (format: ``D{session}:{turn}``)
# ------------------------------------------------------------------

_EVIDENCE_REF_RE = re.compile(r"D(\d+):(\d+)")


def parse_evidence_refs(evidence_list: List[str]) -> List[Tuple[int, int]]:
    """Convert locomo10-style evidence refs into ``(session, turn)`` tuples.

    >>> parse_evidence_refs(["D1:3", "D2:8"])
    [(1, 3), (2, 8)]
    """
    pairs: List[Tuple[int, int]] = []
    for ref in evidence_list:
        m = _EVIDENCE_REF_RE.search(str(ref))
        if m:
            pairs.append((int(m.group(1)), int(m.group(2))))
    return pairs


def evidence_session_numbers(evidence_list: List[str]) -> Set[int]:
    """Return the distinct session numbers referenced by *evidence_list*."""
    return {sess for sess, _ in parse_evidence_refs(evidence_list)}


def resolve_evidence_turns(
    evidence_refs: List[str],
    conversation: dict,
) -> List[str]:
    """Return one dialogue string per evidence ref (looked up from *conversation*).

    *conversation* is a locomo10 conversation dict that has ``session_1``,
    ``session_2``, ... keys, each pointing at a list of turn dicts with
    ``"text"`` and ``"speaker"`` fields.
    """
    turns: List[str] = []
    for session_num, turn_num in parse_evidence_refs(evidence_refs):
        session_key = f"session_{session_num}"
        session_turns = conversation.get(session_key, [])
        if 0 < turn_num <= len(session_turns):
            turn = session_turns[turn_num - 1]
            turns.append(f"{turn.get('speaker', '')}: {turn.get('text', '')}")
    return turns


def split_evidence_into_turns(evidence_text: str) -> List[str]:
    """Break a raw cognitive evidence string into individual turns.

    Cognitive evidence is typically a short dialogue with lines like
    ``"Speaker A: ...\\nSpeaker B: ..."``. We split on newlines and keep
    every non-empty line.
    """
    return [ln.strip() for ln in evidence_text.split("\n") if ln.strip()]


# ------------------------------------------------------------------
# Core recall computations
# ------------------------------------------------------------------

def compute_session_recall(
    evidence_sessions: Set[int],
    retrieved_memories: list,
    conv_idx: int,
) -> int:
    """Binary recall: ``1`` iff any retrieved memory is from an evidence session.

    Memories must carry ``source_conv_idx`` and ``source_session`` metadata
    (populated during memory building).
    """
    if not evidence_sessions:
        return 0
    for mem in retrieved_memories:
        if (
            mem.get("source_conv_idx") == conv_idx
            and mem.get("source_session") in evidence_sessions
        ):
            return 1
    return 0


def compute_text_recall(
    evidence_turns: List[str],
    retrieved_memory_values: List[str],
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> Optional[float]:
    """Evidence-level recall: fraction of evidence turns matched by a memory.

    Computed via sentence-embedding cosine similarity. For each evidence turn
    we test whether *any single* retrieved memory clears the threshold.
    Speaker prefixes and timestamp brackets are stripped before encoding.

    Returns ``None`` when SentenceTransformer is unavailable or every
    evidence turn is empty after stripping.
    """
    model = _get_sentence_model()
    if model is None:
        logger.warning("SentenceTransformer unavailable; text_recall will be None")
        return None

    # Strip metadata from both sides
    ev_texts = [strip_metadata(t) for t in evidence_turns]
    ev_texts = [t for t in ev_texts if t]
    if not ev_texts:
        return None

    mem_texts = [strip_metadata(v) for v in retrieved_memory_values]
    mem_texts = [t for t in mem_texts if t]
    if not mem_texts:
        return 0.0

    # Encode in a single batch
    ev_embeddings = model.encode(ev_texts, convert_to_numpy=True, show_progress_bar=False)
    mem_embeddings = model.encode(mem_texts, convert_to_numpy=True, show_progress_bar=False)

    # Cosine similarity matrix of shape (num_evidence, num_memories).
    # SentenceTransformer outputs are already L2-normalised, so dot product == cosine.
    sim_matrix = np.dot(ev_embeddings, mem_embeddings.T)

    # For each evidence turn, check whether the best memory clears the threshold.
    max_sims = sim_matrix.max(axis=1)  # shape: (num_evidence,)
    matched = int((max_sims >= threshold).sum())

    return round(matched / len(ev_texts), 4)


# ------------------------------------------------------------------
# Unified entry points
# ------------------------------------------------------------------

def compute_recall_for_locomo_qa(
    evidence_refs: List[str],
    retrieved_memories: list,
    conv_idx: int,
    conversation: Optional[dict] = None,
) -> Dict[str, object]:
    """Recall calculation for a standard locomo10 QA item.

    Args:
        evidence_refs: e.g. ``["D1:3", "D2:8"]``.
        retrieved_memories: list of memory dicts (with ``source_conv_idx``,
            ``source_session`` and ``value`` fields).
        conv_idx: conversation index.
        conversation: optional conversation dict, required for text recall.

    Returns:
        ``{"session_recall": 0|1, "text_recall": float|None}``.
    """
    sessions = evidence_session_numbers(evidence_refs)
    session_hit = compute_session_recall(sessions, retrieved_memories, conv_idx)

    text_recall = None
    if conversation is not None:
        evidence_turns = resolve_evidence_turns(evidence_refs, conversation)
        if evidence_turns:
            mem_values = [m.get("value", "") for m in retrieved_memories]
            text_recall = compute_text_recall(evidence_turns, mem_values)

    return {"session_recall": session_hit, "text_recall": text_recall}


def compute_recall_for_cognitive(
    evidence_text: str,
    evidence_after_session: int,
    retrieved_memories: list,
    conv_idx: int,
) -> Dict[str, object]:
    """Recall calculation for a cognitive evaluation item.

    Args:
        evidence_text: raw evidence dialogue string.
        evidence_after_session: session that evidence was injected into
            (``0`` means prepended to session 1, ``-1`` means unknown).
        retrieved_memories: list of memory dicts.
        conv_idx: conversation index.

    Returns:
        ``{"session_recall": 0|1, "text_recall": float|None}``.
    """
    target_session = evidence_after_session
    if target_session == 0:
        target_session = 1
    elif target_session == -1:
        target_session = None

    session_hit = 0
    if target_session is not None:
        session_hit = compute_session_recall(
            {target_session}, retrieved_memories, conv_idx,
        )

    evidence_turns = split_evidence_into_turns(evidence_text)
    mem_values = [m.get("value", "") for m in retrieved_memories]
    text_recall = compute_text_recall(evidence_turns, mem_values)

    return {"session_recall": session_hit, "text_recall": text_recall}

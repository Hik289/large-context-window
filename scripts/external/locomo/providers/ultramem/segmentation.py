"""
Conversation Segmentation Utilities

Provides LLM-driven (and batch-fallback) segmentation that splits a sequence of
chat messages into coherent topical episodes.
"""

import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from ultramem.utils.llm import ChatCompletionModel

logger = logging.getLogger(__name__)


# ---- Structured-output schema -------------------------------------------------
class Episode(BaseModel):
    """One topical chunk of a longer conversation."""
    indices: List[int] = Field(description="List of message indices (0-based) in this episode")
    topic: str = Field(description="Brief description of the topic discussed in this episode")


class SegmentationOutput(BaseModel):
    """Top-level container returned by the segmentation LLM call."""
    episodes: List[Episode] = Field(description="List of topical episodes identified in the conversation")


# Prompt adapted from the Nemori formulation. Do NOT alter wording — it is the
# LLM input and changing it would break experiment comparability.
SEGMENTATION_PROMPT_TEMPLATE = """You are an expert conversation segmentation specialist. Your goal is to analyze a series of messages in a conversation and segment them into coherent topical episodes.

# TASK:
Read the conversation carefully, and identify points where the topic shifts significantly. Group messages that discuss similar subject, event, or theme into a single episode.

An episode is defined as a sequence of messages that revolve around a core topic or theme. Your task is to segment the conversation into such episodes.

## Output Format
Your output should be a JSON object with the following structure:
{{
    "episodes": [
        {{ 
            "topic": "<brief topic description>", 
            "indices": [<list of message indices in this episode>] 
        }},
        ...
    ]
}}

Where each episode contains:
- topic: A brief description (a few words) summarizing the main topic of the episode
- indices: A list of 1-based indices of messages that belong to this episode


# GUIDELINES:

## Segmentation Criteria
- Topical shift: Identify topic shifts in the messages. Does it introduce a new subject, event, or theme? Break the episode there. Be sensitive to subtle shifts in topic.
- Transitions: Look for transition phrases that signal a new episode, such as "By the way", "Changing the subject", or "On another note".
- Time gaps: Significant time lapses between messages may indicate a new episode.
- Setting changes: Changes in speaker, location, or context can signal a new episode.
- Topical grouping: Group consecutive messages into the same episode if they discuss the same topic or theme.


## Episode Length
- An episode should typically contain 2-8 messages.
- Combine messages into larger episodes when they discuss the same topic. 
- Avoid having long episodes (more than 8 messages) that cover multiple sub-topics.
- Avoid treating a single message as a standalone episode unless it clearly marks a shift in topic.
- When in doubt, split into smaller episodes.


## Formatting Rules
- Use 1-based indexing for message indices (i.e., the first message is index 1).
- Ensure that all messages are included in exactly one episode (no gaps or overlaps).
- Indices within each episode should be consecutive, reflecting the order of messages in the conversation.
- Episodes should cover all messages exactly once (no gaps, no overlaps).

Example output:
{{
    "episodes": [
        {{
            "topic": "General introduction and greetings",
            "indices": [1, 2, 3, 4],
        }},
        {{
            "topic": "Discussion about vacation plans",
            "indices": [5, 6],
        }},
        {{
            "topic": "Recap of last year's events",
            "indices": [7, 8, 9, 10, 11, 12],
        }}, 
        ...
    ]
}}

Respond only with the JSON object and no additional text.

Segment the following conversation:

{messages}

Output:
"""


class ConversationSegmenter:
    """Splits a conversation into topical episodes via LLM (or fixed batches)."""

    def __init__(self, cfg):
        """Wire up the segmenter.

        Args:
            cfg: Configuration object holding LLM/segmentation options.
        """
        self.cfg = cfg
        self.enable_segmentation = cfg.memory.get("enable_segmentation", False)
        self.batch_size = 2  # default chunk size used when LLM mode is off

        if self.enable_segmentation:
            logger.info("LLM-based segmentation enabled")
            self.llm = ChatCompletionModel(cfg)
        else:
            logger.info("Batch-based segmentation enabled")
            self.llm = None

    def _format_messages_for_prompt(
        self,
        messages: List[Dict[str, Any]],
        start_idx: int = 0,
        is_context: bool = False
    ) -> str:
        """
        Render messages as a numbered, role-labeled text block suitable for the
        segmentation prompt.

        Args:
            messages: Sequence of message dicts.
            start_idx: Offset added to the displayed numbering.
            is_context: Mark the block as background-only context if True.

        Returns:
            A single string with one rendered message per line.
        """
        rendered = []
        prefix = "[CONTEXT] " if is_context else ""

        for offset, msg in enumerate(messages):
            display_idx = start_idx + offset + 1  # 1-based index for the prompt
            content = msg.get("content", "")

            if isinstance(content, list):
                # Multimodal content: collect text parts and flag images
                pieces = [item.get("text", "") for item in content if item.get("type") == "text"]
                text_content = " ".join(pieces)
                if any(item.get("type") == "image_url" for item in content):
                    text_content += " [Contains image]"
            else:
                text_content = content

            role = msg.get("role", "user")
            rendered.append(f"{prefix}{display_idx}. {role}: {text_content}")

        return "\n".join(rendered)

    def _create_segmentation_prompt(
        self,
        messages: List[Dict[str, Any]]
    ) -> str:
        """Build the full prompt string handed to the LLM.

        Args:
            messages: Conversation messages to segment.

        Returns:
            A prompt ready to be sent to the segmentation LLM.
        """
        rendered_block = self._format_messages_for_prompt(messages, start_idx=0, is_context=False)
        return SEGMENTATION_PROMPT_TEMPLATE.format(messages=rendered_block)

    def _call_llm_for_segmentation(self, prompt: str) -> SegmentationOutput:
        """Invoke the LLM and parse its response into a structured object.

        Args:
            prompt: The fully assembled segmentation prompt.

        Returns:
            A populated ``SegmentationOutput`` instance.
        """
        try:
            return self.llm.invoke(
                input=prompt,
                response_format=SegmentationOutput,
                source="ConversationSegmenter",
                temperature=0.0
            )

        except Exception as exc:
            logger.error(f"LLM segmentation failed: {exc}")
            raise

    def segment_conversation(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Produce topical segments for a conversation.

        Falls back to batch-based chunking whenever LLM segmentation is disabled
        or the LLM call fails / returns invalid output. Always returns a usable
        result.

        Args:
            messages: Conversation messages.

        Returns:
            A list of segments, each with ``indices`` and a ``topic`` label.
        """
        if not messages:
            return []

        if not self.enable_segmentation:
            return self._segment_with_batches(messages)

        segments = self._segment_with_llm(messages)
        if segments is None:
            logger.warning("LLM segmentation failed, falling back to batch mode")
            return self._segment_with_batches(messages)
        return segments

    def _segment_with_llm(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]] | None:
        """Run LLM segmentation and validate the result.

        Returns:
            The segments list on success, ``None`` if the LLM call fails or
            produces an invalid layout.
        """
        total = len(messages)
        logger.info(f"Segmenting conversation with LLM ({total} messages)")

        try:
            prompt = self._create_segmentation_prompt(messages)
            parsed = self._call_llm_for_segmentation(prompt)

            # Convert pydantic objects into plain dicts and shift to 0-based indices.
            segments = [
                {"indices": [i - 1 for i in ep.indices], "topic": ep.topic}
                for ep in parsed.episodes
            ]

            if not self._validate_segments(segments, total):
                logger.error(f"LLM returned invalid segments for {total} messages")
                return None

            logger.info(f"Created {len(segments)} LLM segments from {total} messages")
            return segments

        except Exception as exc:
            logger.error(f"LLM segmentation error: {exc}")
            return None

    def _segment_with_batches(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Slice messages into fixed-size chunks (no LLM call)."""
        total = len(messages)
        logger.info(f"Segmenting conversation with batch mode ({total} messages, batch_size={self.batch_size})")

        segments = []
        for start in range(0, total, self.batch_size):
            stop = min(start + self.batch_size, total)
            segments.append({
                "indices": list(range(start, stop)),
                "topic": f"Conversation batch {start // self.batch_size + 1}"
            })

        logger.info(f"Created {len(segments)} batch segments")
        return segments

    def _validate_segments(self, segments: List[Dict[str, Any]], num_messages: int) -> bool:
        """Sanity-check that segments form a valid partition of all messages.

        Args:
            segments: Candidate segments with ``indices`` lists.
            num_messages: Total messages in the source conversation.

        Returns:
            ``True`` when the partition is valid, ``False`` otherwise.
        """
        if not segments:
            logger.warning("No segments returned")
            return False

        gathered = []
        for seg in segments:
            indices = seg.get("indices", [])
            if not indices:
                logger.error(f"Segment has empty indices: {seg}")
                return False
            gathered.extend(indices)

        # Bounds check: every index must address a real message.
        for i in gathered:
            if i < 0 or i >= num_messages:
                logger.error(f"Index {i} out of range [0, {num_messages-1}]")
                return False

        # Detect duplicates.
        if len(gathered) != len(set(gathered)):
            duplicates = [i for i in set(gathered) if gathered.count(i) > 1]
            logger.error(f"Duplicate indices found: {duplicates}")
            return False

        # Coverage check: no gaps and no extras.
        expected = set(range(num_messages))
        actual = set(gathered)

        missing = expected - actual
        if missing:
            logger.error(f"Missing message indices: {sorted(missing)}")
            return False

        extra = actual - expected
        if extra:
            logger.error(f"Extra indices not in conversation: {sorted(extra)}")
            return False

        # Soft check: warn (do not fail) if a segment's indices are not contiguous.
        for seg in segments:
            ordered = sorted(seg["indices"])
            for j in range(len(ordered) - 1):
                if ordered[j + 1] - ordered[j] != 1:
                    logger.warning(f"Non-consecutive indices in segment '{seg['topic']}': {ordered}")
                    # logged as a warning only — still considered acceptable

        logger.info("Segment validation passed")
        return True

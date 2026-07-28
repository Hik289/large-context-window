"""
Topic-based conversation segmentation.

Provides ``ConversationSegmenter`` which slices a conversation into
topical episodes either via an LLM (when enabled) or by simple
fixed-size batches as a fallback.
"""

import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from agent_memory.utils.llm import ChatCompletionModel

logger = logging.getLogger(__name__)


# Pydantic schemas used as the structured-output target for the LLM.
class Episode(BaseModel):
    """A single topical episode within the conversation."""
    indices: List[int] = Field(description="List of message indices (0-based) in this episode")
    topic: str = Field(description="Brief description of the topic discussed in this episode")


class SegmentationOutput(BaseModel):
    """Wrapper carrying the list of segmented episodes."""
    episodes: List[Episode] = Field(description="List of topical episodes identified in the conversation")


# Prompt for asking the LLM to perform the segmentation (Nemori-style).
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
    """Cuts a conversation into topical episodes using an LLM (or fixed batches)."""

    def __init__(self, cfg):
        """Initialise the segmenter from a config object.

        Args:
            cfg: Config exposing ``cfg.memory.enable_segmentation`` plus the
                OpenAI settings used by ``ChatCompletionModel``.
        """
        self.cfg = cfg
        self.enable_segmentation = cfg.memory.get("enable_segmentation", False)
        self.batch_size = 2  # Default batch size in fallback mode.

        # Only spin up the LLM when LLM-based segmentation is enabled.
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
        is_context: bool = False,
    ) -> str:
        """
        Render messages as ``"<idx>. <role>: <text>"`` lines for the prompt.

        Args:
            messages: Conversation messages.
            start_idx: Offset added to the (1-based) numbering.
            is_context: When True, lines are prefixed with ``[CONTEXT]``.

        Returns:
            Newline-joined string of formatted messages.
        """
        lines = []
        prefix = "[CONTEXT] " if is_context else ""

        for i, msg in enumerate(messages):
            line_idx = start_idx + i + 1  # 1-based numbering inside the prompt.
            content = msg.get("content", "")

            # Multimodal content arrives as a list of text/image dicts.
            if isinstance(content, list):
                text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
                text_content = " ".join(text_parts)
                has_image = any(item.get("type") == "image_url" for item in content)
                if has_image:
                    text_content += " [Contains image]"
            else:
                text_content = content

            role = msg.get("role", "user")
            lines.append(f"{prefix}{line_idx}. {role}: {text_content}")

        return "\n".join(lines)

    def _create_segmentation_prompt(
        self,
        messages: List[Dict[str, Any]],
    ) -> str:
        """
        Build the prompt that will be handed to the LLM.

        Args:
            messages: Conversation messages to segment.

        Returns:
            The fully-rendered prompt string.
        """
        rendered = self._format_messages_for_prompt(messages, start_idx=0, is_context=False)
        return SEGMENTATION_PROMPT_TEMPLATE.format(messages=rendered)

    def _call_llm_for_segmentation(self, prompt: str) -> SegmentationOutput:
        """Invoke the LLM with the structured ``SegmentationOutput`` schema.

        Args:
            prompt: Pre-rendered segmentation prompt.

        Returns:
            Parsed ``SegmentationOutput`` instance.
        """
        try:
            outcome = self.llm.invoke(
                input=prompt,
                response_format=SegmentationOutput,
                source="ConversationSegmenter",
                temperature=0.0,
            )
            return outcome

        except Exception as exc:
            logger.error(f"LLM segmentation failed: {exc}")
            raise

    def segment_conversation(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Slice a conversation into topical segments.

        When LLM-based segmentation is enabled, that path is used; otherwise
        — or if it fails / produces invalid output — fixed-size batches are
        returned. Either way the function always yields a valid segmentation.

        Args:
            messages: Conversation messages.

        Returns:
            A list of dicts each with:
                - ``indices``: message indices belonging to the segment
                - ``topic``: short topic label (or generic label for batch mode)
        """
        num_messages = len(messages)

        if num_messages == 0:
            return []

        # Fallback path — batch mode.
        if not self.enable_segmentation:
            return self._segment_with_batches(messages)

        # LLM path; on any issue fall back to batch segmentation.
        segments = self._segment_with_llm(messages)
        if segments is not None:
            return segments

        logger.warning("LLM segmentation failed, falling back to batch mode")
        return self._segment_with_batches(messages)

    def _segment_with_llm(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]] | None:
        """Run the LLM-based segmentation path.

        Returns:
            The validated list of segments on success, or ``None`` if the LLM
            call failed or its output didn't pass validation.
        """
        num_messages = len(messages)
        logger.info(f"Segmenting conversation with LLM ({num_messages} messages)")

        try:
            prompt = self._create_segmentation_prompt(messages)
            outcome = self._call_llm_for_segmentation(prompt)

            # Convert pydantic models into dicts and shift to 0-based indices.
            segments = [{"indices": [pos - 1 for pos in ep.indices], "topic": ep.topic} for ep in outcome.episodes]

            if not self._validate_segments(segments, num_messages):
                logger.error(f"LLM returned invalid segments for {num_messages} messages")
                return None

            logger.info(f"Created {len(segments)} LLM segments from {num_messages} messages")
            return segments

        except Exception as exc:
            logger.error(f"LLM segmentation error: {exc}")
            return None

    def _segment_with_batches(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback: slice the conversation into fixed-size chunks."""
        num_messages = len(messages)
        logger.info(f"Segmenting conversation with batch mode ({num_messages} messages, batch_size={self.batch_size})")

        segments = []
        for i in range(0, num_messages, self.batch_size):
            end_idx = min(i + self.batch_size, num_messages)
            segments.append({
                "indices": list(range(i, end_idx)),
                "topic": f"Conversation batch {i // self.batch_size + 1}",
            })

        logger.info(f"Created {len(segments)} batch segments")
        return segments

    def _validate_segments(self, segments: List[Dict[str, Any]], num_messages: int) -> bool:
        """Check that ``segments`` is a well-formed partition of the messages.

        Args:
            segments: Candidate segmentation.
            num_messages: Total number of messages in the conversation.

        Returns:
            True if the segmentation is valid, False otherwise.
        """
        if not segments:
            logger.warning("No segments returned")
            return False

        # Flatten the indices across all segments.
        all_indices = []
        for seg in segments:
            seg_indices = seg.get("indices", [])
            if not seg_indices:
                logger.error(f"Segment has empty indices: {seg}")
                return False
            all_indices.extend(seg_indices)

        # Bounds check.
        for pos in all_indices:
            if pos < 0 or pos >= num_messages:
                logger.error(f"Index {pos} out of range [0, {num_messages-1}]")
                return False

        # Duplicate check.
        if len(all_indices) != len(set(all_indices)):
            duplicates = [pos for pos in set(all_indices) if all_indices.count(pos) > 1]
            logger.error(f"Duplicate indices found: {duplicates}")
            return False

        # Coverage check (every message must appear in some segment).
        expected_indices = set(range(num_messages))
        actual_indices = set(all_indices)

        missing = expected_indices - actual_indices
        if missing:
            logger.error(f"Missing message indices: {sorted(missing)}")
            return False

        extra = actual_indices - expected_indices
        if extra:
            logger.error(f"Extra indices not in conversation: {sorted(extra)}")
            return False

        # Soft check: indices inside a segment should normally be consecutive.
        for seg in segments:
            sorted_indices = sorted(seg["indices"])
            for i in range(len(sorted_indices) - 1):
                if sorted_indices[i + 1] - sorted_indices[i] != 1:
                    logger.warning(f"Non-consecutive indices in segment '{seg['topic']}': {sorted_indices}")
                    # Not a hard error — just log it.

        logger.info("Segment validation passed")
        return True

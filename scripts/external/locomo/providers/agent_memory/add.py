import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from omegaconf import DictConfig
from tqdm import tqdm

from agent_memory.client import MemoryClient
from utils import generate_debug_data, get_session_num, load_data, is_valid_image_url
from providers.agent_memory.segmentation import ConversationSegmenter

logger = logging.getLogger(__name__)


class AgentMemoryAdd:

    def __init__(
        self,
        cfg: DictConfig,
        data_path=None,
        batch_size=2,
        data_config=None,
    ):

        self.cfg = cfg
        self.data_config = data_config

        self.batch_size = batch_size
        self.data_path = data_path
        self.data = None
        if data_path:
            self.load_data()

        # Per-user MemoryClient cache (lazily populated under a lock).
        self.cached_clients = {}
        self._client_lock = threading.Lock()

        # Always instantiate; segmenter handles both LLM and batch fallback paths.
        self.segmenter = ConversationSegmenter(cfg)

    def get_memory_client(self, user_id: str):
        cached = self.cached_clients.get(user_id)
        if cached is not None:
            return cached

        # Lock prevents concurrent ChromaDB initialization races.
        with self._client_lock:
            cached = self.cached_clients.get(user_id)
            if cached is not None:
                return cached

            client = MemoryClient(self.cfg, user_id=user_id)
            self.cached_clients[user_id] = client
            return client

    def load_data(self):

        self.data = load_data(self.data_path, subset_idx=self.cfg.eval.subset_idx)

        if self.data_config:
            self.data = generate_debug_data(
                self.data,
                self.data_config["conversion_idx"],
                self.data_config["session_idx"],
                self.data_config["num_sessions"],
            )
        elif self.cfg.general.debug:
            self.data = generate_debug_data(self.data)

        return self.data

    def add_memory(self, user_id, message, metadata, retries=3):

        client = self.get_memory_client(user_id)
        for attempt in range(retries):
            try:
                _ = client.add(message, builder="chat", metadata=metadata)
                return
            except Exception as exc:
                if attempt >= retries - 1:
                    raise exc
                time.sleep(1)  # back off briefly before retrying

    def add_memories_for_speaker(self, speaker, chunks, desc):
        """Persist a list of (messages, metadata) pairs for a given speaker.

        Args:
            speaker: Speaker user id used as the storage key.
            chunks: Iterable of ``(messages, metadata)`` tuples.
            desc: Label shown on the tqdm progress bar.
        """
        for msgs, meta in tqdm(chunks, desc=desc):
            self.add_memory(speaker, msgs, metadata=meta)

    def process_conversation(self, item, idx):

        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]
        conv_idx = idx  # keep a stable copy — the inner loop reuses ``idx``

        use_combined_user = self.cfg.eval.get("use_combined_user", False)

        if use_combined_user:
            # Single shared user id across both speakers.
            combined_user_id = f"{speaker_a}_{speaker_b}_{idx}"
            client_combined = self.get_memory_client(combined_user_id)
            client_combined.clear()
        else:
            # Per-speaker user ids.
            speaker_a_user_id = f"{speaker_a}_{idx}"
            speaker_b_user_id = f"{speaker_b}_{idx}"

            client_speaker_a = self.get_memory_client(speaker_a_user_id)
            client_speaker_b = self.get_memory_client(speaker_b_user_id)

            # Wipe any prior memory belonging to either speaker.
            client_speaker_a.clear()
            client_speaker_b.clear()

        num_sessions = get_session_num(conversation)
        for idx in range(1, num_sessions + 1):

            print("=" * 60, f" Processing session {idx}/{num_sessions} ", "=" * 60)
            key = f"session_{idx}"

            date_time_key = key + "_date_time"
            timestamp = conversation[date_time_key]
            if not timestamp:
                print("No timestamp found, using current time")
            chats = conversation[key]

            messages = []
            messages_reverse = None if use_combined_user else []
            for chat in chats:
                text_content = chat["text"]
                speaker = chat["speaker"]

                if "blip_caption" in chat:
                    image_description = chat["blip_caption"]
                    if "query" in chat:
                        image_description += f", {chat['query']}"
                    text_content = text_content + f" [Image Description: {image_description}]"

                if "img_url" in chat:
                    image_url = chat["img_url"]
                    if isinstance(image_url, list):
                        image_url = image_url[0] if image_url else None

                    # Build a multimodal payload only when the URL passes validation.
                    if image_url and is_valid_image_url(image_url):
                        message = [
                            {"type": "text", "text": f"{speaker}: {text_content}"},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    else:
                        # Otherwise, send a plain text message.
                        message = f"{speaker}: {text_content}"
                else:
                    message = f"{speaker}: {text_content}"

                if use_combined_user:
                    # All entries share the ``user`` role in combined mode.
                    messages.append({"role": "user", "content": message})
                else:
                    # Mirror the speaker -> role mapping for the two threads.
                    chat_speaker = chat["speaker"]
                    if chat_speaker == speaker_a:
                        messages.append({"role": "user", "content": message})
                        messages_reverse.append({"role": "assistant", "content": message})
                    elif chat_speaker == speaker_b:
                        messages.append({"role": "assistant", "content": message})
                        messages_reverse.append({"role": "user", "content": message})
                    else:
                        raise ValueError(f"Unknown speaker: {chat_speaker}")

            # Segmenter takes care of both LLM-based and batch-based chunking.
            logger.info(f"Processing session {idx} with {len(messages)} messages")
            segments = self.segmenter.segment_conversation(messages)
            logger.info(f"Created {len(segments)} segments")

            if use_combined_user:
                # One unified set of chunks for the combined user.
                chunks_combined = []
                for seg_idx, segment in enumerate(segments):
                    seg_msgs = [messages[i] for i in segment["indices"]]

                    metadata = {
                        "timestamp": timestamp,
                        "segment_topic": segment.get("topic", "conversation"),
                        "segment_index": seg_idx,
                        "source_conv_idx": conv_idx,
                        "source_session": idx,
                    }

                    chunks_combined.append((seg_msgs, metadata))

                self.add_memories_for_speaker(
                    combined_user_id, chunks_combined, "Adding Memories for Combined User"
                )
            else:
                # Two parallel chunk lists, one per speaker perspective.
                chunks_a = []
                chunks_b = []
                for seg_idx, segment in enumerate(segments):
                    seg_msgs = [messages[i] for i in segment["indices"]]
                    seg_msgs_reverse = [messages_reverse[i] for i in segment["indices"]]

                    metadata = {
                        "timestamp": timestamp,
                        "segment_topic": segment.get("topic", "conversation"),
                        "segment_index": seg_idx,
                        "source_conv_idx": conv_idx,
                        "source_session": idx,
                    }

                    chunks_a.append((seg_msgs, metadata))
                    chunks_b.append((seg_msgs_reverse, metadata))

                # Spin up one writer thread per speaker.
                thread_a = threading.Thread(
                    target=self.add_memories_for_speaker,
                    args=(speaker_a_user_id, chunks_a, "Adding Memories for Speaker A"),
                )
                thread_b = threading.Thread(
                    target=self.add_memories_for_speaker,
                    args=(speaker_b_user_id, chunks_b, "Adding Memories for Speaker B"),
                )

                thread_a.start()
                thread_b.start()
                thread_a.join()
                thread_b.join()

        print("Messages added successfully")

    def process_all_conversations(self):
        if not self.data:
            raise ValueError("No data loaded. Please set data_path and call load_data() first.")
        with ThreadPoolExecutor(max_workers=self.cfg.eval.max_workers) as executor:
            futures = [executor.submit(self.process_conversation, item, idx) for idx, item in enumerate(self.data)]

            for fut in futures:
                fut.result()

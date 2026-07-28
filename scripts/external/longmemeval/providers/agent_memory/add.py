import json
import logging
import time
from datetime import datetime
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import os

from importlib_metadata import metadata
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from agent_memory.client import MemoryClient
from openai import ContentFilterFinishReasonError

logger = logging.getLogger(__name__)


class AgentMemoryAdd:

    def __init__(
        self,
        cfg: DictConfig,
        data_path=None,
        batch_size=32,
        data_config=None,
    ):
        self.cfg = cfg
        self.data_config = data_config
        self.batch_size = batch_size
        self.data_path = data_path
        self.data = None
        if data_path:
            self.load_data()

        # Per-user MemoryClient cache so we don't recreate clients on every call.
        self.cached_clients = {}
        self._client_lock = threading.Lock()

    def get_memory_client(self, user_id: str):
        if user_id in self.cached_clients:
            return self.cached_clients[user_id]

        client = MemoryClient(self.cfg, user_id=user_id)

        # Cache for subsequent lookups.
        self.cached_clients[user_id] = client

        return client

    def load_data(self):
        with open(self.data_path, "r") as f:
            payload = json.load(f)

        subset_idx = self.cfg.eval.subset_idx
        if subset_idx > 0:
            payload = payload[subset_idx - 1: subset_idx]

        self.data = payload

        if self.cfg.general.debug:
            self.data = self.data[:1]

        logger.info(f"Loaded {len(self.data)} questions for processing.")

        return self.data

    def add_memory(self, client, user_id: str, chunks, retries=3):
        for messages, meta in chunks:
            for attempt in range(retries):
                try:
                    _ = client.add(messages, type="chat", metadata=meta)
                    return True

                except ContentFilterFinishReasonError as exc:
                    logger.warning(
                        f"Content filter rejected memory for user {user_id}. Skipping."
                    )
                    return False

                except Exception as exc:
                    if attempt < retries - 1:
                        time.sleep(1)
                        continue
                    raise exc

        return False

    def process_conversation(self, item, idx):
        """Build memories for a longmemeval question using SESSION-BASED segmentation."""

        question_id = item.get("question_id", "")
        user_id = f"question_{question_id}"

        logger.info(f"\nProcessing question ({idx+1}/{len(self.data)}): {question_id}")

        if "haystack_sessions" not in item:
            logger.warning(f"Item {idx} doesn't have 'haystack_sessions' - skipping")
            return

        client = self.get_memory_client(user_id)
        client.clear()

        haystack_sessions = item.get("haystack_sessions", [])
        haystack_dates = item.get("haystack_dates", [])
        haystack_session_ids = item.get("haystack_session_ids", [])

        num_sessions = len(haystack_sessions)
        total_messages = 0

        # Treat each haystack session as one segment.
        for session_idx, (session, date, session_id) in enumerate(
            zip(haystack_sessions, haystack_dates, haystack_session_ids)
        ):
            logger.info(f"  Processing session {session_idx + 1}/{len(haystack_sessions)}: {session_id}")

            messages = []

            # Reformat the per-session date into a more readable string.
            try:
                ts_obj = datetime.strptime(date, '%Y/%m/%d (%a) %H:%M')

                fmt = "%-I:%M %p on %-d %B, %Y"
                rendered = ts_obj.strftime(fmt).lower()
                parts = rendered.split(' ')
                parts[4] = parts[4].capitalize()

                timestamp = ' '.join(parts)

            except Exception as exc:
                print(f"    Warning: Failed to parse date '{date}': {exc}, using original timestamp.")
                timestamp = date

            for turn_idx, turn in enumerate(session):
                role = turn['role']
                content = turn['content']

                messages.append({"role": role, "content": content})

            print(f"Total messages in session {session_idx + 1}: {len(messages)}")

            chunks = []

            # Split a session into batch_size-sized chunks if it's too long.
            if len(messages) > self.batch_size:
                for i in range(0, len(messages), self.batch_size):
                    sub_msgs = messages[i: i + self.batch_size]
                    chunk_meta = {
                        "timestamp": timestamp,
                        "segment_index": str(session_id) + "_" + str(i),
                    }
                    chunks.append((sub_msgs, chunk_meta))
            else:
                chunk_meta = {
                    "timestamp": timestamp,
                    "segment_index": str(session_id),
                }
                chunks.append((messages, chunk_meta))

            # Hand the chunks to MemoryClient.add().
            try:
                self.add_memory(client, user_id, chunks)

            except Exception as exc:
                logger.error(f"Error adding session {session_idx} for question {question_id}: {exc}")
                continue

        # Per-question summary banner.
        print(f"\n{'='*60}")
        print(f"Question {question_id} Summary:")
        print(f"  Total sessions (segments): {num_sessions}")
        print(f"  Total messages: {total_messages}")
        print(f"{'='*60}\n")

    def process_all_conversations(self):
        if not self.data:
            raise ValueError("No data loaded. Please set data_path and call load_data() first.")

        print(f"[DEBUG] Total questions to process: {len(self.data)}")

        # Each question is independent, so fan out to a thread pool.
        with ThreadPoolExecutor(max_workers=self.cfg.eval.max_workers) as pool:
            pending = [
                pool.submit(self.process_conversation, item, idx)
                for idx, item in enumerate(self.data)
            ]

            for fut in pending:
                fut.result()  # Surface exceptions raised by workers.

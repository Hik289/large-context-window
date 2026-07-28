import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from omegaconf import DictConfig
from tqdm import tqdm

from utils import generate_debug_data, init_mem0_client, load_data
from mem0 import Memory
load_dotenv()


# Memory-extraction guidance used by the underlying LLM.
custom_instructions = """
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name, do not use "user" while creating memories
   - Personal details (career aspirations, hobbies, life circumstances)
   - Emotional states and reactions
   - Ongoing journeys or future plans
   - Specific dates when events occurred

2. Include meaningful personal narratives focusing on:
   - Identity and self-acceptance journeys
   - Family planning and parenting
   - Creative outlets and hobbies
   - Mental health and self-care activities
   - Career aspirations and education goals
   - Important life events and milestones

3. Make each memory rich with specific details rather than general statements
   - Include timeframes (exact dates when possible)
   - Name specific activities (e.g., "charity race for mental health" rather than just "exercise")
   - Include emotional context and personal growth elements

4. Extract memories only from user messages, not incorporating assistant responses

5. Format each memory as a paragraph with a clear narrative structure that captures the person's experience, challenges, and aspirations
"""


class MemoryADD:

    def __init__(
        self,
        cfg: DictConfig,
        memory: Memory,
        data_path=None,
        batch_size=2,
        is_graph=False,
    ):
        self.memory = memory
        self.cfg = cfg

        self.batch_size = batch_size
        self.data_path = data_path
        self.data = None
        self.is_graph = is_graph
        if data_path:
            self.load_data()

    def load_data(self):

        self.data = load_data(self.data_path, subset_idx=self.cfg.eval.subset_idx)

        # Optionally restrict to a small slice for fast iteration.
        if self.cfg.general.debug:
            self.data = generate_debug_data(self.data)
        return self.data

    def add_memory(self, user_id, message, metadata, retries=3):
        for attempt in range(retries):
            try:
                _ = self.memory.add(
                    message, user_id=user_id, metadata=metadata
                )
                return
            except Exception as exc:
                if attempt >= retries - 1:
                    raise exc
                time.sleep(1)  # back off briefly before trying again

    def add_memories_for_speaker(self, speaker, messages, timestamp, desc):
        total = len(messages)
        for i in tqdm(range(0, total, self.batch_size), desc=desc):
            batch = messages[i:i + self.batch_size]
            self.add_memory(speaker, batch, metadata={"timestamp": timestamp})

    def process_conversation(self, item, idx):
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        speaker_a_user_id = f"{speaker_a}_{idx}"
        speaker_b_user_id = f"{speaker_b}_{idx}"

        # Wipe any prior memory for both users before re-ingesting.
        self.memory.delete_all(user_id=speaker_a_user_id)
        self.memory.delete_all(user_id=speaker_b_user_id)

        for key in conversation.keys():
            # Skip metadata keys; only the per-session message lists are processed.
            if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                continue

            date_time_key = key + "_date_time"
            timestamp = conversation[date_time_key]
            chats = conversation[key]

            messages = []
            messages_reverse = []
            for chat in chats:
                speaker = chat["speaker"]
                text = chat["text"]
                if speaker == speaker_a:
                    messages.append({"role": "user", "content": f"{speaker_a}: {text}"})
                    messages_reverse.append({"role": "assistant", "content": f"{speaker_a}: {text}"})
                elif speaker == speaker_b:
                    messages.append({"role": "assistant", "content": f"{speaker_b}: {text}"})
                    messages_reverse.append({"role": "user", "content": f"{speaker_b}: {text}"})
                else:
                    raise ValueError(f"Unknown speaker: {speaker}")

            # Two writer threads — one for each speaker perspective.
            thread_a = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(speaker_a_user_id, messages, timestamp, "Adding Memories for Speaker A"),
            )
            thread_b = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(speaker_b_user_id, messages_reverse, timestamp, "Adding Memories for Speaker B"),
            )

            thread_a.start()
            thread_b.start()
            thread_a.join()
            thread_b.join()

        print("Messages added successfully")

    def process_all_conversations(self, max_workers=1):
        if not self.data:
            raise ValueError("No data loaded. Please set data_path and call load_data() first.")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_conversation, item, idx) for idx, item in enumerate(self.data)]

            for fut in futures:
                fut.result()

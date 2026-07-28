import json
import os
import time
from collections import defaultdict

from jinja2 import Template
from omegaconf import DictConfig
from tqdm import tqdm

from utils import load_data
from agent_memory.utils.llm import get_general_chat_completion_client

# Shares the answer-prompt template used by agent_memory and Mem0 so the three
# baselines remain directly comparable.
PROMPT = """
    You are an intelligent memory assistant that answers questions based on conversation history.

    # CONTEXT:
    You have access to the full conversation history between two speakers. The conversation contains timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze the conversation history
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the conversation
    4. If the conversation contains contradictory information, prioritize the most recent information in terms of timestamp
    5. If there is a question about time references (like "last year", "two months ago", etc.), calculate the actual date based on the conversation timestamp. For example, if a conversation from 4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example, convert "last year" to "2022" or "two months ago" to "March 2023" based on the conversation timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the conversation. Do not confuse character names mentioned in conversation with the actual speakers.
    8. The answer should be less than 5-6 words.


    # APPROACH (Think step by step):
    1. First, examine all parts of the conversation that contain information related to the question
    2. Examine the timestamps and content carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the conversation
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Conversation History:

    {{CONTEXT}}

    Question: {{QUESTION}}

    Answer:
    """


class FullContextManager:
    """
    Baseline that feeds the entire conversation history into the LLM as context,
    skipping any retrieval or memory-extraction step.
    """

    def __init__(self, cfg: DictConfig, data_path: str):
        self.cfg = cfg
        self.model = cfg.llm.model  # generation model (e.g. gpt-4.1-mini)
        self.llm_client = get_general_chat_completion_client(cfg)
        self.data_path = data_path

    def generate_response(self, question, context):
        """Run the LLM with the full conversation as the user prompt."""
        template = Template(PROMPT)
        prompt = template.render(CONTEXT=context, QUESTION=question)

        max_retries = 3
        attempt = 0

        while attempt <= max_retries:
            try:
                t1 = time.time()
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that can answer questions based on the provided context."
                        ,
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                )
                t2 = time.time()
                return response.choices[0].message.content.strip(), t2 - t1
            except Exception as exc:
                attempt += 1
                if attempt > max_retries:
                    raise exc
                time.sleep(1)  # brief pause before retrying

    def clean_chat_history(self, chat_history):
        """Flatten chat-history dicts into a single human-readable string."""
        cleaned = ""
        for chat in chat_history:
            ts = chat.get("timestamp", "")
            speaker = chat.get("speaker", "")
            text = chat.get("text", "")
            cleaned += f"{ts} | {speaker}: {text}\n"
        return cleaned

    def process_all_conversations(self, output_file_path):
        """Iterate over every conversation in the dataset and answer its QAs."""
        data = load_data(self.data_path)

        FINAL_RESULTS = {}
        for idx, item in tqdm(enumerate(data), desc="Processing conversations", total=len(data)):
            # Use ``sample_id`` when available; otherwise fall back to the index.
            key = item.get("sample_id", f"conversation_{idx}")

            # Flatten every session into one chat history list.
            conversation = item["conversation"]
            chat_history = []

            session_idx = 1
            while f"session_{session_idx}" in conversation:
                session_chats = conversation[f"session_{session_idx}"]
                session_datetime = conversation.get(f"session_{session_idx}_date_time", "")

                for chat in session_chats:
                    chat_with_ts = chat.copy()
                    if session_datetime and "timestamp" not in chat_with_ts:
                        chat_with_ts["timestamp"] = session_datetime
                    chat_history.append(chat_with_ts)

                session_idx += 1

            # Materialize the entire conversation as a single context string.
            full_context = self.clean_chat_history(chat_history)

            questions = item["qa"]

            FINAL_RESULTS[key] = []
            for qa_item in tqdm(questions, desc="Answering questions", leave=False):
                question = qa_item["question"]
                answer = qa_item.get("answer", "")
                category = str(qa_item["category"])

                # Whole conversation acts as context for every QA.
                response, response_time = self.generate_response(question, full_context)

                FINAL_RESULTS[key].append(
                    {
                        "question": question,
                        "answer": answer,
                        "category": category,
                        "context": full_context,  # retain full context for later inspection
                        "response": response,
                        "response_time": response_time,
                    }
                )

            # Snapshot after each conversation so progress is preserved on crash.
            with open(output_file_path, "w+") as f:
                json.dump(FINAL_RESULTS, f, indent=4)

        # Final write at the very end.
        with open(output_file_path, "w+") as f:
            json.dump(FINAL_RESULTS, f, indent=4)

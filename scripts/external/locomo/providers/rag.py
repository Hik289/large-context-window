import json
import os
import time
from collections import defaultdict

import numpy as np
import tiktoken
from dotenv import load_dotenv
from jinja2 import Template
from omegaconf import DictConfig
from tqdm import tqdm

from utils import load_data
from ultramem.utils.llm import get_general_chat_completion_client
from ultramem.utils.embedding import get_general_embedding_client

load_dotenv()

# Re-uses the same answer-prompt structure as ultramem/Mem0 for an apples-
# to-apples comparison.
PROMPT = """
    You are an intelligent memory assistant that answers questions based on retrieved context.

    # CONTEXT:
    You have access to retrieved context from a conversation between two speakers. This context contains timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze the provided context
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the context
    4. If the context contains contradictory information, prioritize the most recent information in terms of timestamp
    5. If there is a question about time references (like "last year", "two months ago", etc.), calculate the actual date based on the context timestamp. For example, if context from 4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example, convert "last year" to "2022" or "two months ago" to "March 2023" based on the context timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the context. Do not confuse character names mentioned in context with the actual speakers.
    8. The answer should be less than 5-6 words.


    # APPROACH (Think step by step):
    1. First, examine all parts of the context that contain information related to the question
    2. Examine the timestamps and content carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the context
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Retrieved Context:

    {{CONTEXT}}

    Question: {{QUESTION}}

    Answer:
    """


class RAGManager:
    def __init__(self, cfg: DictConfig, data_path="dataset/locomo10.json", chunk_size=500, k=1):
        self.cfg = cfg
        self.model = cfg.llm.model  # generation model (e.g. gpt-4.1-mini)
        self.embedding_model = cfg.openai.embedding_model
        self.llm_client = get_general_chat_completion_client(cfg)
        self.embedding_client = get_general_embedding_client(cfg)
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.k = k

    def generate_response(self, question, context):
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
        cleaned = ""
        for c in chat_history:
            cleaned += f"{c['timestamp']} | {c['speaker']}: {c['text']}\n"
        return cleaned

    def calculate_embedding(self, document):
        response = self.embedding_client.embeddings.create(model=self.embedding_model, input=document)
        return response.data[0].embedding

    def calculate_similarity(self, embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def search(self, query, chunks, embeddings, k=1):
        """
        Return the top-k most similar chunks (concatenated) along with the
        elapsed search time.

        Args:
            query: Query string.
            chunks: Pre-chunked document text.
            embeddings: Embeddings parallel to ``chunks``.
            k: How many chunks to keep (default ``1``).

        Returns:
            Tuple ``(combined_chunks, search_time)``.
        """
        t1 = time.time()
        query_embedding = self.calculate_embedding(query)
        similarities = [self.calculate_similarity(query_embedding, e) for e in embeddings]

        # Pick the top-k indices (single-best fast path for k==1).
        if k == 1:
            top_indices = [np.argmax(similarities)]
        else:
            top_indices = np.argsort(similarities)[-k:][::-1]

        combined_chunks = "\n<->\n".join([chunks[i] for i in top_indices])

        t2 = time.time()
        return combined_chunks, t2 - t1

    def create_chunks(self, chat_history, chunk_size=500):
        """
        Build chunks using ``tiktoken`` for token-accurate sizing.
        """
        encoding = tiktoken.encoding_for_model(self.embedding_model)

        documents = self.clean_chat_history(chat_history)

        if chunk_size == -1:
            return [documents], []

        chunks = []

        tokens = encoding.encode(documents)

        # Slice the token stream into ``chunk_size``-sized segments.
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + chunk_size]
            chunks.append(encoding.decode(chunk_tokens))

        embeddings = [self.calculate_embedding(c) for c in chunks]

        return chunks, embeddings

    def process_all_conversations(self, output_file_path):
        data = load_data(self.data_path)

        FINAL_RESULTS = {}
        for idx, item in tqdm(enumerate(data), desc="Processing conversations", total=len(data)):
            key = item.get("sample_id", f"conversation_{idx}")

            # Flatten the per-session messages into one chat history list.
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

            questions = item["qa"]

            chunks, embeddings = self.create_chunks(chat_history, self.chunk_size)

            FINAL_RESULTS[key] = []
            for qa_item in tqdm(questions, desc="Answering questions", leave=False):
                question = qa_item["question"]
                answer = qa_item.get("answer", "")
                category = str(qa_item["category"])

                if self.chunk_size == -1:
                    context = chunks[0]
                    search_time = 0
                else:
                    context, search_time = self.search(question, chunks, embeddings, k=self.k)
                response, response_time = self.generate_response(question, context)

                FINAL_RESULTS[key].append(
                    {
                        "question": question,
                        "answer": answer,
                        "category": category,
                        "context": context,
                        "response": response,
                        "search_time": search_time,
                        "response_time": response_time,
                    }
                )

            # Snapshot after each conversation so progress survives crashes.
            with open(output_file_path, "w+") as f:
                json.dump(FINAL_RESULTS, f, indent=4)

        # Final save.
        with open(output_file_path, "w+") as f:
            json.dump(FINAL_RESULTS, f, indent=4)

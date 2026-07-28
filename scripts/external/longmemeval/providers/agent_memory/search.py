import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import logging

from jinja2 import Template
from omegaconf import DictConfig, OmegaConf
from agent_memory.client import MemoryClient
from agent_memory.utils.llm import get_general_chat_completion_client
from prompts import ANSWER_PROMPT_NEMORI_LONGMEMEVAL
from tqdm import tqdm
from openai import BadRequestError

logger = logging.getLogger(__name__)


class AgentMemorySearch:

    def __init__(self, cfg: DictConfig, output_path="results.json", top_k=30):

        self.cfg = cfg
        self.top_k = top_k
        self.results = defaultdict(list)
        self.output_path = output_path
        self.ANSWER_PROMPT = ANSWER_PROMPT_NEMORI_LONGMEMEVAL
        self.llm_client = get_general_chat_completion_client(cfg)
        self.enable_hybrid_search = self.cfg.memory.get("enable_hybrid_search", False)

        # Reusable per-user client cache.
        self.cached_clients = {}

    def get_memory_client(self, user_id: str):
        if user_id in self.cached_clients:
            return self.cached_clients[user_id]

        # Mirror the per-user persist path layout used by add.py.
        user_cfg = OmegaConf.to_container(self.cfg, resolve=True)
        base_path = user_cfg["memory"]["persist_path"]
        user_cfg["memory"]["persist_path"] = os.path.join(base_path, user_id)
        user_cfg = OmegaConf.create(user_cfg)

        client = MemoryClient(user_cfg, user_id=user_id)
        self.cached_clients[user_id] = client
        return client

    def search_memory(self, user_id, query, max_retries=3, retry_delay=1):
        started = time.time()
        attempts = 0

        client = self.get_memory_client(user_id)
        while attempts < max_retries:
            try:
                memories = client.query(
                    query,
                    top_k=self.top_k,
                    enable_hybrid_search=self.enable_hybrid_search,
                    where={"memory_type": {"$eq": "factual"}},
                )
                break
            except Exception as exc:
                print(f"Retrying...{exc}")
                attempts += 1
                if attempts >= max_retries:
                    return [], 0.0
                time.sleep(retry_delay)

        ended = time.time()

        semantic_memories = []
        for mem in memories:
            mem_dict = {
                "index": mem.index,
                "value": mem.get_memory_value(),
                "timestamp": mem.timestamp,
                "score": round(mem.score, 2),
                "memory_type": mem.memory_type,
            }
            # Optionally surface attached image URLs (multimodal mode).
            if self.multimodal_support and mem.image_urls:
                mem_dict["image_urls"] = mem.image_urls

            if mem.episodic_memory_ids:
                mem_dict["episodic_memory_ids"] = mem.episodic_memory_ids
            semantic_memories.append(mem_dict)

        return semantic_memories, ended - started

    def process_question(self, item, idx):
        """Generate an answer for a single LongMemEval question."""
        question_id = item.get("question_id", "")
        question = item.get("question", "")
        answer = item.get("answer", "")
        question_type = item.get("question_type", "")
        question_date = item.get("question_date", "")
        answer_session_ids = item.get("answer_session_ids", [])

        user_id = f"question_{question_id}"

        memories, memory_time = self.search_memory(user_id, question)

        # Apply episodic-context formatting if it is enabled.
        formatted_memories = self._format_memories_with_episodic_context(memories, user_id)

        # Collect any image URLs attached to retrieved memories.
        all_image_urls = set()
        for mem_item in memories:
            if mem_item.get('image_urls', None):
                for img_url in mem_item['image_urls']:
                    all_image_urls.add(img_url)

        all_images = [
            {"type": "image_url", "image_url": {"url": url}}
            for url in all_image_urls
        ][:500]

        # Render the prompt template.
        template = Template(self.ANSWER_PROMPT)
        prompt = template.render(
            memories="\n".join(formatted_memories),
            question=question,
            question_date=question_date,
        )

        t1 = time.time()
        # Multimodal path: include image content blocks if we have any.
        if self.multimodal_support and all_images:
            user_content = [{"type": "text", "text": prompt}]
            user_content.extend(all_images)

            try:
                response = self.llm_client.chat.completions.create(
                    model=self.cfg.openai.model,
                    messages=[{"role": "user", "content": user_content}],
                    temperature=0.0,
                    seed=self.cfg.openai.seed,
                )
            except BadRequestError as exc:
                if "403" in str(exc) or "can not be accessed" in str(exc):
                    logger.warning(f"Image URL access failed (403 error) during query: {exc}. Falling back to text-only mode.")
                    response = self.llm_client.chat.completions.create(
                        model=self.cfg.openai.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        seed=self.cfg.openai.seed,
                    )
                else:
                    raise
        else:
            response = self.llm_client.chat.completions.create(
                model=self.cfg.openai.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                seed=self.cfg.openai.seed,
            )

        t2 = time.time()
        response_time = t2 - t1

        outcome = {
            "question_id": question_id,
            "question": question,
            "answer": answer,
            "question_type": question_type,
            "question_date": question_date,
            "answer_session_ids": answer_session_ids,
            "response": response.choices[0].message.content,
            "memories": memories,
            "formatted_memories": formatted_memories,
            "num_memories": len(memories),
            "memory_time": memory_time,
            "response_time": response_time,
        }

        return outcome

    def process_data_file(self, file_path):
        """Walk a LongMemEval data file question-by-question."""

        data = load_data(file_path, subset_idx=self.cfg.eval.subset_idx)

        if self.cfg.general.debug:
            data = generate_debug_data(data)

        # In LongMemEval each entry corresponds to one question.
        for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing questions"):
            outcome = self.process_question(item, idx)
            qid = outcome["question_id"]
            self.results[qid] = [outcome]

            # Persist after every question so we never lose progress.
            with open(self.output_path, "w") as f:
                json.dump(self.results, f, indent=4)

        # Final write.
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

    def _retrieve_and_cluster_episodic_memories(self, memories, user_id):
        """
        Pull the episodic entries that are linked to ``memories`` and bucket the
        factual ones by their tuple of episodic-ids. Memories that share the
        same set of episodes form a multi-episode cluster.

        Args:
            memories: List of factual memory dicts.
            user_id: User id used to look up the memory client.

        Returns:
            ``(sorted_clusters, orphans, episodic_dict)`` where ``sorted_clusters``
            is a list of ``(episodic_ids_tuple, cluster_data)`` pairs ordered by
            ``max_score`` descending.
        """
        client = self.get_memory_client(user_id)

        # 1) Gather every episodic id we need to fetch.
        all_episodic_ids = set()
        for mem in memories:
            ep_ids = mem.get('episodic_memory_ids', [])
            if ep_ids:
                all_episodic_ids.update(ep_ids)
            elif mem.get('memory_type') == 'episodic':
                # Episodic memory pulled directly by the search.
                all_episodic_ids.add(mem['index'])

        # 2) Fetch the episodic entries.
        episodic_memories_dict = {}
        for ep_id in all_episodic_ids:
            ep_entry = client.get(ep_id)
            if ep_entry:
                episodic_memories_dict[ep_id] = ep_entry

        # 3) Cluster factual memories by their episodic-id tuple.
        episodic_clusters = {}  # tuple[ep_id, ...] -> {'memories': [...], 'max_score': float}
        orphan_memories = []

        for mem in memories:
            ep_ids = mem.get('episodic_memory_ids', [])
            if ep_ids:
                # Tuple preserves order so identical sequences cluster together.
                cluster_key = tuple(ep_ids)
                if cluster_key in episodic_clusters:
                    # Bump the running max for the cluster if this score is higher.
                    episodic_clusters[cluster_key]['max_score'] = max(
                        episodic_clusters[cluster_key]['max_score'],
                        mem['score'],
                    )
                else:
                    episodic_clusters[cluster_key] = {
                        'memories': [],
                        'max_score': mem['score'],
                    }
                episodic_clusters[cluster_key]['memories'].append(mem)
            else:
                # No episodic links — could be a pure factual or a directly-
                # retrieved episodic memory.
                orphan_memories.append(mem)

        # 4) Promote standalone episodic memories (retrieved directly but not
        # referenced by any factual memory) into their own clusters.
        remaining_orphans = []
        for mem in orphan_memories:
            if mem.get('memory_type') == 'episodic' and mem['index'] in episodic_memories_dict:
                # Skip if some cluster already references this episodic memory.
                already_referenced = any(mem['index'] in cluster_key for cluster_key in episodic_clusters.keys())
                if not already_referenced:
                    cluster_key = (mem['index'],)
                    episodic_clusters[cluster_key] = {
                        'memories': [],
                        'max_score': mem['score'],
                    }
            else:
                remaining_orphans.append(mem)

        orphan_memories = remaining_orphans

        # 5) Sort each cluster's factual memories by score (descending).
        for cluster_data in episodic_clusters.values():
            cluster_data['memories'].sort(key=lambda x: x['score'], reverse=True)

        # 6) Order clusters by their max score (descending).
        sorted_clusters = sorted(
            episodic_clusters.items(),
            key=lambda x: x[1]['max_score'],
            reverse=True,
        )

        return sorted_clusters, orphan_memories, episodic_memories_dict

    def _format_memories_with_episodic_context(self, memories, user_id):
        """
        Render memories for the prompt, optionally interleaving the linked
        episodic context. The clustering logic mirrors the locomo provider
        for consistency.
        """
        episodic_enabled = self.cfg.memory.get("enable_episodic_memory", False)

        if not episodic_enabled:
            # Plain (non-episodic) rendering.
            return [f"- [{m['timestamp']}] {m['value']}" for m in memories]

        # Episodic rendering: cluster, then format.
        episodic_clusters, orphan_memories, episodic_memories_dict = \
            self._retrieve_and_cluster_episodic_memories(memories, user_id)

        formatted_memories = []

        # Walk clusters in descending max_score order.
        for episodic_ids_tuple, cluster_data in episodic_clusters:
            # Resolve the actual episodic entries for this cluster.
            valid_episodic_entries = []
            for episodic_id in episodic_ids_tuple:
                episodic_entry = episodic_memories_dict.get(episodic_id)
                if episodic_entry:
                    valid_episodic_entries.append(episodic_entry)

            if valid_episodic_entries:
                # First emit the episodic memories.
                for ep_entry in valid_episodic_entries:
                    formatted_memories.append(
                        f"- [{ep_entry.timestamp}] {ep_entry.value}"
                    )

                # Then the factual memories belonging to this cluster (already
                # ordered by score above).
                for factual_mem in cluster_data['memories']:
                    formatted_memories.append(
                        f"- [{factual_mem['timestamp']}] {factual_mem['value']}"
                    )

                # Visual separator between clusters.
                formatted_memories.append("")
            else:
                # No usable episodic entries — demote the factual memories to orphans.
                orphan_memories.extend(cluster_data['memories'])

        # Append the orphan memories at the end of the output.
        for mem in orphan_memories:
            formatted_memories.append(
                f"- [{mem['timestamp']}] {mem['value']}"
            )

        return formatted_memories

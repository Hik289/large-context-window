import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import logging

from jinja2 import Template
from omegaconf import DictConfig
from utils import generate_debug_data, load_data
from agent_memory.client import MemoryClient
from agent_memory.utils.llm import get_general_chat_completion_client
from agent_memory.utils.latency import LatencyTracker, count_memories_tokens
from metrics.retrieval_recall import compute_recall_for_locomo_qa
from prompts import ANSWER_PROMPT, ANSWER_PROMPT_COMBINED, ANSWER_PROMPT_EVERMEMOS
from tqdm import tqdm
from openai import BadRequestError

logger = logging.getLogger(__name__)


class AgentMemorySearch:

    def __init__(self, cfg: DictConfig, output_path="results.json", top_k=30, retrieval_strategy="semantic"):
        self.cfg = cfg
        self.top_k = top_k
        self.results = defaultdict(list)
        self.output_path = output_path
        self.retrieval_strategy = retrieval_strategy
        self.use_combined_user = self.cfg.eval.get("use_combined_user", False)

        # Pick the appropriate answer-prompt template depending on user mode.
        if self.use_combined_user:
            # In combined mode, the prompt template is configurable.
            prompt_template = self.cfg.eval.get("prompt_template", "mem0")
            prompt_lookup = {
                "mem0": ANSWER_PROMPT_COMBINED,
                "evermemos": ANSWER_PROMPT_EVERMEMOS,
            }
            chosen = prompt_lookup.get(prompt_template)
            if chosen is None:
                logger.warning(f"Invalid prompt_template '{prompt_template}', defaulting to 'mem0'.")
                chosen = ANSWER_PROMPT_COMBINED
            self.ANSWER_PROMPT = chosen
        else:
            # Split-user runs always use the canonical answer prompt.
            self.ANSWER_PROMPT = ANSWER_PROMPT

        self.llm_client = get_general_chat_completion_client(cfg)

        self.enable_hybrid_search = self.cfg.memory.get("enable_hybrid_search", False)

        # Default to True so multimodal payloads are passed through unchanged.
        self.multimodal_support = self.cfg.memory.get("multimodal_support", True)

        # When True, only the most recent episode for each factual memory is shown.
        self.use_latest_episode_only = False

        self.cached_clients = {}

    def get_memory_client(self, user_id: str) -> MemoryClient:
        if user_id in self.cached_clients:
            return self.cached_clients[user_id]

        client = MemoryClient(self.cfg, user_id=user_id)
        self.cached_clients[user_id] = client

        return client

    def search_memory(self, user_id, query, top_k=None, max_retries=3, retry_delay=1, latency_tracker=None):
        start_time = time.time()
        attempts = 0

        client = self.get_memory_client(user_id)

        if top_k is None:
            top_k = self.top_k

        while attempts < max_retries:
            try:
                memories = client.advance_query(
                    query,
                    top_k=top_k,
                    query_type=self.retrieval_strategy,  # "semantic" or "prompt"
                    latency_tracker=latency_tracker,
                )
                break

            except Exception as exc:
                print(f"Retrying...{exc}")
                attempts += 1
                if attempts >= max_retries:
                    return [], 0.0
                time.sleep(retry_delay)

        end_time = time.time()

        semantic_memories = []
        for mem in memories:
            entry = {
                "index": mem.index,
                "value": mem.get_memory_value(),
                "timestamp": mem.timestamp,
                "score": round(mem.score, 2),
                "memory_type": mem.memory_type,
            }
            # Surface image URLs only when multimodal support is on and any exist.
            if self.multimodal_support and mem.image_urls:
                entry["image_urls"] = mem.image_urls
            if mem.episodic_memory_ids:
                entry["episodic_memory_ids"] = mem.episodic_memory_ids
            # Carry through provenance for downstream recall computation.
            extra = mem.extra_metadata or {}
            if "source_conv_idx" in extra:
                entry["source_conv_idx"] = extra["source_conv_idx"]
            if "source_session" in extra:
                entry["source_session"] = extra["source_session"]
            semantic_memories.append(entry)

        return semantic_memories, end_time - start_time

    def _retrieve_and_cluster_episodic_memories(self, memories, user_id):
        """
        Fetch episodic memories and group factual memories that share the same
        episode set into a single cluster.

        Args:
            memories: List of factual memory dictionaries.
            user_id: User id used to obtain the appropriate memory client.

        Returns:
            Tuple ``(episodic_clusters, orphan_memories, episodic_memories_dict)``,
            where ``episodic_clusters`` is a list of ``(ids_tuple, cluster_data)``.
        """
        client = self.get_memory_client(user_id)

        # 1. Collect every episodic id we will need to retrieve.
        all_episodic_ids = set()
        for mem in memories:
            episodic_ids_list = mem.get('episodic_memory_ids', [])
            if episodic_ids_list:
                if self.use_latest_episode_only:
                    all_episodic_ids.add(episodic_ids_list[-1])
                else:
                    all_episodic_ids.update(episodic_ids_list)
            elif mem.get('memory_type') == 'episodic':
                # Episodic memory was returned directly by the retriever.
                all_episodic_ids.add(mem['index'])

        # 2. Fetch the episodic entries.
        episodic_memories_dict = {}
        for episodic_id in all_episodic_ids:
            episodic_entry = client.get(episodic_id)
            if episodic_entry:
                episodic_memories_dict[episodic_id] = episodic_entry

        # 3. Group factual memories by their referenced episodic ids.
        episodic_clusters = {}  # tuple(ids) -> {memories: [...], max_score: float}
        orphan_memories = []  # memories with no episodic reference

        for mem in memories:
            episodic_ids_list = mem.get('episodic_memory_ids', [])
            if not episodic_ids_list:
                # No episodic link — could be factual or directly retrieved episodic.
                orphan_memories.append(mem)
                continue

            # Use the (ordered) tuple of ids as the cluster key.
            if self.use_latest_episode_only:
                cluster_key = (episodic_ids_list[-1],)
            else:
                cluster_key = tuple(episodic_ids_list)

            bucket = episodic_clusters.get(cluster_key)
            if bucket is None:
                episodic_clusters[cluster_key] = {
                    'memories': [mem],
                    'max_score': mem['score'],
                }
            else:
                bucket['max_score'] = max(bucket['max_score'], mem['score'])
                bucket['memories'].append(mem)

        # 4. Promote standalone episodic memories that aren't already covered.
        remaining_orphans = []
        for mem in orphan_memories:
            if (
                mem.get('memory_type') == 'episodic'
                and mem['index'] in episodic_memories_dict
            ):
                already_referenced = any(
                    mem['index'] in cluster_key for cluster_key in episodic_clusters.keys()
                )
                if not already_referenced:
                    cluster_key = (mem['index'],)
                    episodic_clusters[cluster_key] = {
                        'memories': [],
                        'max_score': mem['score'],
                    }
                    continue
            remaining_orphans.append(mem)

        orphan_memories = remaining_orphans

        # 5. Sort each cluster's factual memories by score (descending).
        for cluster_data in episodic_clusters.values():
            cluster_data['memories'].sort(key=lambda x: x['score'], reverse=True)

        # 6. Order clusters globally by their best score.
        sorted_clusters = sorted(
            episodic_clusters.items(),
            key=lambda x: x[1]['max_score'],
            reverse=True,
        )

        return sorted_clusters, orphan_memories, episodic_memories_dict

    def _format_speaker_memories(
        self, memories, user_id, enable_episodic
    ):
        """
        Render a speaker's memories as a list of strings, optionally injecting
        episodic context blocks ahead of the related factual memories.

        Args:
            memories: Memory dicts for this speaker (already include
                ``episodic_memory_ids`` when relevant).
            user_id: Used to resolve a ``MemoryClient`` for episodic lookups.
            enable_episodic: Whether to include episodic context.

        Returns:
            A list of formatted memory strings.
        """

        if not enable_episodic:
            # Plain rendering — no episodic context.
            formatted_memories = []
            for item in memories:
                value = item['value']
                # If the merged value already embeds timestamps, do not prefix again.
                if value.startswith("[") and "]\n" in value:
                    formatted_memories.append(value)
                else:
                    formatted_memories.append(f"{item['timestamp']}: {value}")

            return formatted_memories

        # Group memories by their episodic references.
        episodic_clusters, orphan_memories, episodic_memories_dict = \
            self._retrieve_and_cluster_episodic_memories(memories, user_id)

        formatted_memories = []

        # Walk clusters in score order.
        for episodic_ids_tuple, cluster_data in episodic_clusters:
            # Resolve every episodic entry for this cluster.
            valid_episodic_entries = []
            for episodic_id in episodic_ids_tuple:
                entry = episodic_memories_dict.get(episodic_id)
                if entry:
                    valid_episodic_entries.append(entry)

            if not valid_episodic_entries:
                # No episodic context resolved — treat factual memories as orphans.
                orphan_memories.extend(cluster_data['memories'])
                continue

            # Emit the episodic context block.
            for episodic_entry in valid_episodic_entries:
                ts = episodic_entry.timestamp
                if ts is not None:
                    date = ts.split(" on ")[-1] if " on " in ts else ts
                else:
                    date = "Unknown date"
                    logger.warning(f"Episodic memory {episodic_entry.index} has no timestamp.")
                formatted_memories.append(f"[{date}]\n{episodic_entry.value}")

            formatted_memories.append("Details:")  # divider before the factual lines
            # Factual memories are already sorted by score within the cluster.
            for factual_mem in cluster_data['memories']:
                value = factual_mem['value']
                if value.startswith("[") and "]\n" in value:
                    formatted_memories.append(value)
                else:
                    formatted_memories.append(f"{factual_mem['timestamp']}: {value}")

            # Blank line between clusters.
            formatted_memories.append("\n")

        # Trailing orphan memories (no associated episodic context).
        for item in orphan_memories:
            value = item['value']
            if value.startswith("[") and "]\n" in value:
                formatted_memories.append(value)
            else:
                formatted_memories.append(f"{item['timestamp']}: {value}")

        return formatted_memories

    def answer_question(self, speaker_1_user_id, speaker_2_user_id, question):
        # Per-question latency capture.
        tracker = LatencyTracker()
        tracker.start_overall_search()

        enable_episodic = self.cfg.memory.get("enable_episodic_memory", False)

        if self.use_combined_user:
            # Combined mode runs a single retrieval for the merged user id.
            speaker_1_memories, speaker_1_memory_time = self.search_memory(
                speaker_1_user_id, question, latency_tracker=tracker
            )

            with tracker.track("format_memories"):
                search_1_memory = self._format_speaker_memories(
                    speaker_1_memories, speaker_1_user_id, enable_episodic
                )

            # No data for "speaker 2" in combined mode.
            speaker_2_memories = []
            speaker_2_memory_time = 0.0
            search_2_memory = []

        else:
            # Split mode performs two retrievals — one per speaker — using half
            # of ``top_k`` for each.
            speaker_1_memories, time_1 = self.search_memory(
                speaker_1_user_id, question, top_k=self.top_k // 2, latency_tracker=tracker
            )
            speaker_2_memories, time_2 = self.search_memory(
                speaker_2_user_id, question, top_k=self.top_k // 2, latency_tracker=tracker
            )

            with tracker.track("format_memories"):
                search_1_memory = self._format_speaker_memories(
                    speaker_1_memories, speaker_1_user_id, enable_episodic
                )
                search_2_memory = self._format_speaker_memories(
                    speaker_2_memories, speaker_2_user_id, enable_episodic
                )

            speaker_1_name = speaker_1_user_id.split("_")[0]
            speaker_2_name = speaker_2_user_id.split("_")[0]
            speaker_1_memory_time = time_1
            speaker_2_memory_time = time_2

        tracker.end_overall_search()

        # Aggregate any image URLs across the retrieved memories.
        all_image_urls = set()
        for mem in speaker_1_memories + speaker_2_memories:
            if mem.get('image_urls'):
                all_image_urls.update(mem['image_urls'])

        all_images = [
            {"type": "image_url", "image_url": {"url": url}}
            for url in all_image_urls
        ][:500]  # cap at 500 images

        # Render the prompt for the active mode.
        template = Template(self.ANSWER_PROMPT)
        if self.use_combined_user:
            answer_prompt = template.render(
                memories=json.dumps(search_1_memory, indent=4),
                question=question,
            )
        else:
            answer_prompt = template.render(
                speaker_1_user_id=speaker_1_name,
                speaker_2_user_id=speaker_2_name,
                speaker_1_memories=json.dumps(search_1_memory, indent=4),
                speaker_2_memories=json.dumps(search_2_memory, indent=4),
                question=question,
            )

        # Token accounting over the memory strings (not the full prompt).
        all_memory_strings = search_1_memory + search_2_memory
        try:
            token_stats = count_memories_tokens(
                all_memory_strings,
                model=self.cfg.llm.get("model", "YOUR_CHAT_MODEL"),
            )
            token_stats["num_images"] = len(all_images)
            tracker.set_prompt_stats(token_stats)
        except Exception as exc:
            logger.warning(f"Failed to count tokens: {exc}")

        # Choose between text-only and multimodal payload.
        use_multimodal = self.multimodal_support and bool(all_images)
        if use_multimodal:
            user_content = [{"type": "text", "text": answer_prompt}] + all_images
        else:
            user_content = answer_prompt

        # Call the LLM with retries for transient errors.
        response = None
        max_llm_retries = 3

        for llm_attempt in range(max_llm_retries):
            try:
                with tracker.track("llm_generation"):
                    if use_multimodal:
                        try:
                            response = self.llm_client.chat.completions.create(
                                model=self.cfg.llm.model,
                                messages=[{"role": "user", "content": user_content}],
                                temperature=0.0,
                                seed=self.cfg.llm.seed,
                            )
                        except BadRequestError as exc:

                            if "403" in str(exc) or "can not be accessed" in str(exc):

                                logger.warning(
                                    f"Image access failed (403 error): {exc}. Retrying with text-only."
                                )

                                # Fallback to text-only on image access errors.
                                response = self.llm_client.chat.completions.create(
                                    model=self.cfg.llm.model,
                                    messages=[{"role": "user", "content": answer_prompt}],
                                    temperature=0.0,
                                    seed=self.cfg.llm.seed,
                                )

                            else:
                                raise
                    else:
                        response = self.llm_client.chat.completions.create(
                            model=self.cfg.llm.model,
                            messages=[{"role": "user", "content": user_content}],
                            temperature=0.0,
                            seed=self.cfg.llm.seed,
                        )
                break  # request succeeded

            except BadRequestError:
                raise  # not a transient error — bubble up

            except Exception as exc:
                if llm_attempt < max_llm_retries - 1:
                    logger.warning(
                        f"LLM call attempt {llm_attempt + 1}/{max_llm_retries} failed: {exc}. Retrying..."
                    )
                    time.sleep(1 * (llm_attempt + 1))
                else:
                    logger.error(f"LLM call failed after {max_llm_retries} attempts: {exc}")
                    response_time = tracker.get_timing("llm_generation")
                    latency_summary = tracker.get_summary()
                    return (
                        f"ERROR: LLM call failed.",
                        speaker_1_memories,
                        speaker_2_memories,
                        speaker_1_memory_time,
                        speaker_2_memory_time,
                        response_time,
                        search_1_memory,
                        search_2_memory,
                        latency_summary,
                    )

        response_time = tracker.get_timing("llm_generation")

        # Pull the answer text out of the response object.
        if response and response.choices and response.choices[0].message.content:
            result = response.choices[0].message.content.strip()
            if not result:
                result = "ERROR: Empty response from LLM."
                logger.error("Empty response from LLM.")
        else:
            result = "ERROR: No response from LLM."
            logger.error("No response from LLM.")

        if "FINAL ANSWER:" in result:
            parts = result.split("FINAL ANSWER:")
            result = parts[1].strip() if len(parts) > 1 else result.strip()
        else:
            result = result.strip()

        latency_summary = tracker.get_summary()

        return (
            result,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            response_time,
            search_1_memory,
            search_2_memory,
            latency_summary,
        )

    def process_question(self, val, speaker_a_user_id, speaker_b_user_id,
                         conv_idx=None, conversation=None):
        question = val.get("question", "")
        answer = val.get("answer", "")
        category = val.get("category", -1)
        evidence = val.get("evidence", [])
        adversarial_answer = val.get("adversarial_answer", "")

        (
            response,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            response_time,
            formatted_speaker_1_memories,
            formatted_speaker_2_memories,
            latency_summary,
        ) = self.answer_question(speaker_a_user_id, speaker_b_user_id, question)

        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "response": response,
            "adversarial_answer": adversarial_answer,
            "speaker_1_memories": speaker_1_memories,
            "speaker_2_memories": speaker_2_memories,
            "num_speaker_1_memories": len(speaker_1_memories),
            "num_speaker_2_memories": len(speaker_2_memories),
            "speaker_1_memory_time": speaker_1_memory_time,
            "speaker_2_memory_time": speaker_2_memory_time,
            "response_time": response_time,
            "formatted_speaker_1_memories": formatted_speaker_1_memories,
            "formatted_speaker_2_memories": formatted_speaker_2_memories,
            "latency_breakdown": latency_summary,
        }

        # Compute retrieval recall if evidence and source provenance are available.
        if conv_idx is not None and evidence:
            all_memories = speaker_1_memories + speaker_2_memories
            recall = compute_recall_for_locomo_qa(
                evidence, all_memories, conv_idx, conversation=conversation,
            )
            result["session_recall"] = recall["session_recall"]
            result["text_recall"] = recall["text_recall"]

        # Persist results after processing each question.
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

        return result

    def process_data_file(self, file_path):

        data = load_data(file_path, subset_idx=self.cfg.eval.subset_idx)

        if self.cfg.general.debug:
            data = generate_debug_data(data)

        for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing conversations"):
            qa = item["qa"]
            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]

            if self.use_combined_user:
                # Combined mode uses a single user id.
                speaker_a_user_id = f"{speaker_a}_{speaker_b}_{idx}"
                speaker_b_user_id = None  # unused in combined mode
            else:
                speaker_a_user_id = f"{speaker_a}_{idx}"
                speaker_b_user_id = f"{speaker_b}_{idx}"

            for question_item in tqdm(
                qa, total=len(qa), desc=f"Processing questions for conversation {idx}", leave=False
            ):
                result = self.process_question(
                    question_item, speaker_a_user_id, speaker_b_user_id,
                    conv_idx=idx, conversation=conversation,
                )
                self.results[idx].append(result)

                # Save results after each question is processed.
                with open(self.output_path, "w") as f:
                    json.dump(self.results, f, indent=4)

        # Persist a final snapshot at the end.
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

    def process_questions_parallel(self, qa_list, speaker_a_user_id, speaker_b_user_id,
                                    max_workers=1, conv_idx=None, conversation=None):
        def process_single_question(val):
            outcome = self.process_question(
                val, speaker_a_user_id, speaker_b_user_id,
                conv_idx=conv_idx, conversation=conversation,
            )
            with open(self.output_path, "w") as f:
                json.dump(self.results, f, indent=4)
            return outcome

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(executor.map(process_single_question, qa_list), total=len(qa_list), desc="Answering Questions")
            )

        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

        return results

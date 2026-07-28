"""
Locomo Experiment Runner

Provides an experimental harness for evaluating different memory techniques
on the Locomo dataset. Supports several memory backends — agent_memory, Mem0,
and a RAG baseline — driven by configurable test cases.
"""

import os
import sys
import yaml
from typing import Dict, List, Any

import hydra
from omegaconf import DictConfig

from agent_memory.utils.llm import get_general_chat_completion_client

# Make the parent directory importable so we can resolve sibling modules.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Provider implementations and evaluation helpers.
from metrics.llm_judge import evaluate_llm_judge
from providers.agent_memory.add import AgentMemoryAdd
from providers.agent_memory.search import AgentMemorySearch
from evals import evaluate, generate_scores
from agent_memory.utils.log import configure_logging


def load_test_cases(config_file: str = "test_cases.yaml") -> Dict[str, Any]:
    """
    Read the YAML file containing test-case definitions and return its parsed
    contents.

    Args:
        config_file: File name of the YAML configuration to load (resolved
            relative to this module's directory).

    Returns:
        The parsed test-case configuration as a dictionary.

    Raises:
        FileNotFoundError: When the YAML file cannot be located.
        yaml.YAMLError: When the YAML file fails to parse.
    """
    config_path = os.path.join(os.path.dirname(__file__), config_file)

    try:
        with open(config_path, 'r', encoding='utf-8') as fh:
            return yaml.safe_load(fh)
    except FileNotFoundError:
        print(f"Error: Test cases file '{config_file}' not found at {config_path}")
        raise
    except yaml.YAMLError as exc:
        print(f"Error: Invalid YAML format in '{config_file}': {exc}")
        raise


def get_test_case_by_name(test_config: Dict[str, Any], test_name: str) -> Dict[str, Any]:
    """
    Find a single test-case definition by its ``name`` field.

    Args:
        test_config: Parsed test-case configuration.
        test_name: Name of the test case to retrieve.

    Returns:
        The matching test-case dictionary.

    Raises:
        ValueError: When no test case has the requested name.
    """
    for tc in test_config.get("test_cases", []):
        if tc.get("name") == test_name:
            return tc

    raise ValueError(f"Test case '{test_name}' not found in configuration")


def run_single_test_case(cfg: DictConfig, test_case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute one test case and return its judge score.

    Args:
        cfg: Hydra configuration object.
        test_case: A single test-case definition.

    Returns:
        The LLM-judge score for the answer produced by the memory system.
    """
    print(f"\n{'='*60}")
    print(f"Running Test Case: {test_case['name']}")
    print(f"Description: {test_case.get('description', 'No description provided')}")
    print(f"Question: {test_case['question']}")
    print(f"{'='*60}")

    # Pull the parameters that drive this test case.
    data_file = os.path.join(cfg.general.data_path, "locomo10.json")
    memory_store = test_case.get("memory_store", "")

    if memory_store:
        # Reuse an existing memory store — no ingestion needed.
        print(f"Using existing memory store: {memory_store}")
        cfg.memory.persist_path = f"{cfg.general.memory_store_path}/{memory_store}"
        memory_manager = AgentMemoryAdd(cfg, data_path=data_file)
        conversation_idx = test_case.get("conversation_idx", 0)
    else:
        # Build the memory store from scratch for this test case.
        conversation_idx = test_case["conversation_idx"]
        session_idx = test_case["session_idx"]
        num_sessions = test_case.get("num_sessions", 2)

        # Ingest only the conversation slice this test case needs.
        data_config = {
            "conversion_idx": conversation_idx,
            "session_idx": session_idx,
            "num_sessions": num_sessions,
        }

        print("\n==== Building Agent Memory ====")

        # Per-test-case persist path — overwrite if necessary.
        cfg.memory.persist_path = f"{cfg.general.memory_store_path}/chroma_debug"
        memory_manager = AgentMemoryAdd(cfg, data_path=data_file, data_config=data_config)

        # Run the ingestion pass.
        for idx, item in enumerate(memory_manager.data):
            memory_manager.process_conversation(item, idx)

    question = test_case["question"]
    gt_answer = test_case["answer"]

    # Build the searcher and evaluate the answer it produces.
    memory_searcher = AgentMemorySearch(
        cfg, "dummy_result.json", cfg.memory.top_k
    )
    data = memory_manager.data[0]
    conversation = data["conversation"]
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]

    speaker_a_user_id = f"{speaker_a}_{conversation_idx}"
    speaker_b_user_id = f"{speaker_b}_{conversation_idx}"

    (
        response,
        speaker_1_memories,
        speaker_2_memories,
        speaker_1_memory_time,
        speaker_2_memory_time,
        response_time,
    ) = memory_searcher.answer_question(speaker_a_user_id, speaker_b_user_id, question)
    # Print a short summary of what happened.
    print(f"\n==== Test Results ====")
    print(f"Question: {question}")
    print(f"Ground Truth Answer: {gt_answer}")
    print(f"Model Response: {response}")
    print(f"Speaker 1 ({speaker_a}) Memories: {len(speaker_1_memories)} items")
    print(f"Speaker 2 ({speaker_b}) Memories: {len(speaker_2_memories)} items")

    # Score with the LLM judge.
    model_client = get_general_chat_completion_client(cfg)
    llm_score = evaluate_llm_judge(model_client, question, gt_answer, response)
    return llm_score


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run(cfg: DictConfig):
    """
    Entry point that runs a chosen test case end-to-end.

    Loads the YAML test-case definitions, picks one (currently hardcoded), and
    runs it through the full ``AgentMemoryAdd`` -> ``AgentMemorySearch`` ->
    judge pipeline.

    Args:
        cfg: Hydra configuration object.

    Raises:
        ValueError: If the requested test-case name does not exist.
        FileNotFoundError: If the test-case YAML file is missing.
    """
    configure_logging()

    # Force debug mode while running the unit test driver.
    cfg.general.debug = True

    try:
        test_config = load_test_cases()

        # Hardcoded test selection; edit to run a different test case.
        test_name = "Speaker interaction"  # Melanie shopping, Caroline portrait

        print(f"Running single test case: {test_name}")
        test_case = get_test_case_by_name(test_config, test_name)
        result = run_single_test_case(cfg, test_case)
        print("PASS" if result == 1 else "FAIL")
        print(f"\nCompleted test case '{test_name}' successfully.")
    except Exception as exc:
        print(f"Error in experiment execution: {exc}")
        raise


if __name__ == "__main__":
    run()

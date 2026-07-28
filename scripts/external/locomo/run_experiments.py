"""
Locomo Experiment Runner

Comprehensive experiment harness for evaluating different memory techniques
on the Locomo dataset. Supports multiple memory backends including the
agent_memory backend, Mem0, and RAG-based pipelines.
"""

from datetime import datetime
import logging
import os
from venv import logger
# Disable telemetry to keep no background threads alive
import os
os.environ["MEM0_TELEMETRY"] = "False"   # disable mem0 telemetry

from datetime import datetime
import time
try:
    from dotenv import load_dotenv
    load_dotenv()  # default: load from current working directory
except Exception:
    # Fallback: a no-dependency minimal .env loader.
    # Reads variables from the .env file next to this script (if any).
    def _load_dotenv_fallback(dotenv_path: str) -> None:
        try:
            with open(dotenv_path, "r", encoding="utf-8") as fh:
                for raw_line in fh:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    # Don't clobber existing environment variables
                    os.environ.setdefault(key, val)
        except FileNotFoundError:
            return

    _load_dotenv_fallback(os.path.join(os.path.dirname(__file__), ".env"))
import hydra
from omegaconf import DictConfig

from evals import evaluate, generate_scores
from agent_memory.utils.log import configure_logging
from mem0 import Memory

# Memory system implementations
from providers.agent_memory.search import AgentMemorySearch
from providers.agent_memory.add import AgentMemoryAdd
from providers.memzero.add import MemoryADD
from providers.memzero.search import MemorySearch

from providers.rag import RAGManager
from providers.full_context import FullContextManager
# from providers.langmem import LangMemManager
# from providers.openai.predict import OpenAIPredict
# from providers.zep.add import ZepAdd
# from providers.zep.search import ZepSearch
logger = logging.getLogger(__name__)

from utils import format_duration, init_mem0_client, measure_execution_time


def _run_agent_memory_experiment(cfg: DictConfig, data_file: str):
    """
    Run the agent_memory technique end-to-end.

    Steps:
    1. Build the memory store from conversations.
    2. Search the store and evaluate retrieval performance.

    Args:
        cfg: Configuration object containing experiment parameters.
        data_file: Path to the dataset file.
    """
    build_memory = (
        not os.path.exists(cfg.memory.persist_path) or cfg.memory.force_rebuild
    )

    if build_memory:
        logger.info("\n==== Building Agent Memory ====")
        print(f"\n==== Building Agent Memory in {cfg.memory.persist_path} ====")
        memory_manager = AgentMemoryAdd(cfg, data_path=data_file)
        _, build_duration = measure_execution_time(memory_manager.process_all_conversations)
        logger.info(f"Memory processing completed in {format_duration(build_duration)}")
    else:
        logger.info(f"\n==== Skipping memory build — persist path exists: {cfg.memory.persist_path} ====")
        print(f"\n==== Skipping memory build (already exists at {cfg.memory.persist_path}). Set memory.force_rebuild=true to rebuild. ====")

    logger.info("\n==== Searching Agent Memory and Evaluating ====")

    # Output folder is keyed by method name + retrieval strategy + timestamp.
    method = "external_baseline"
    retrieval_strategy = cfg.retrieval.get("strategy", "semantic")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(cfg.general.results_path, f"{method}_{retrieval_strategy}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n==== Output Directory: {output_dir} ====")

    # Output files inside the timestamped directory
    output_file = os.path.join(output_dir, f"{method}_output.json")
    eval_file = os.path.join(output_dir, f"{method}_eval.json")
    score_file = os.path.join(output_dir, f"{method}_scores.json")

    memory_searcher = AgentMemorySearch(cfg, output_file, cfg.memory.top_k, retrieval_strategy=retrieval_strategy)

    # Time the inference phase
    _, eval_duration = measure_execution_time(
        memory_searcher.process_data_file, data_file
    )
    logger.info(f"Memory inference completed in {format_duration(eval_duration)}")

    evaluate(cfg, output_file, eval_file)

    generate_scores(eval_file, score_file)


def _run_mem0_experiment(cfg: DictConfig, data_file: str, output_file: str):
    """
    Run the Mem0 memory technique experiment.

    Args:
        cfg: Configuration object containing experiment parameters.
        data_file: Path to the dataset file.
    """
    method = "mem0"

    # Output folder name: <method>_<subset>_<date>
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = datetime.now().strftime("%Y%m%d")
    subset = cfg.eval.subset_idx if cfg.eval.subset_idx >= 0 else "full"
    output_dir = os.path.join(cfg.general.results_path, f"{method}_{subset}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n==== Output Directory: {output_dir} ====")

    output_file = os.path.join(output_dir, f"{method}_output.json")

    build_memory = (
        not os.path.exists(cfg.memory.persist_path) or cfg.memory.force_rebuild
    )

    memory: Memory = init_mem0_client(cfg)

    if build_memory:
        # Build the memory store and time it
        print(f"\n==== Building Mem0 Memory in {cfg.memory.persist_path} ====")
        memory_manager = MemoryADD(
            cfg, memory, data_path=data_file, is_graph=cfg.memory.is_graph
        )

        tic = time.time()
        for idx, item in enumerate(memory_manager.data):
            memory_manager.process_conversation(item, idx)
        build_duration = time.time() - tic
        print(f"Mem0 memory processing completed in {build_duration:.2f} seconds")

    print("\n==== Searching Mem0 Memory and Evaluating ====")
    memory_searcher = MemorySearch(
        cfg,
        memory,
        output_file,
        cfg.memory.top_k,
        cfg.memory.filter_memories,
        cfg.memory.is_graph,
    )
    _, eval_duration = measure_execution_time(
        memory_searcher.process_data_file, data_file
    )
    print(f"Mem0 evaluation completed in {format_duration(eval_duration)}")

    # Run evaluation
    eval_file = os.path.join(output_dir, f"{method}_eval.json")
    evaluate(cfg, output_file, eval_file)

    # Generate scores
    score_file = os.path.join(output_dir, f"{method}_scores.json")
    generate_scores(eval_file, score_file)


def _run_rag_experiment(cfg: DictConfig, rag_data_file: str):
    """
    Run the RAG (Retrieval-Augmented Generation) experiment.

    Args:
        cfg: Configuration object containing experiment parameters.
        rag_data_file: Path to the RAG-specific dataset file.
    """
    print("\n==== Running RAG Experiment ====")

    method = "rag"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(cfg.general.results_path, f"{method}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n==== Output Directory: {output_dir} ====")

    output_file_path = os.path.join(
        output_dir,
        f"{method}_output.json",
    )

    rag_manager = RAGManager(
        cfg=cfg,
        data_path=rag_data_file,
        chunk_size=cfg.memory.chunk_size,
        k=cfg.memory.num_chunks,
    )
    _, duration = measure_execution_time(
        rag_manager.process_all_conversations, output_file_path
    )
    print(f"RAG generation completed in {format_duration(duration)}")

    print("\n==== Running Evaluation ====")
    eval_file = os.path.join(output_dir, f"{method}_eval.json")
    evaluate(cfg, output_file_path, eval_file)

    print("\n==== Generating Scores ====")
    score_file = os.path.join(output_dir, f"{method}_scores.json")
    generate_scores(eval_file, score_file)


def _run_full_context_experiment(cfg: DictConfig, data_file: str):
    """
    Run the Full-Context baseline.

    The baseline simply hands the entire conversation history to the LLM with
    no extraction or retrieval — useful as a reference point.

    Args:
        cfg: Configuration object containing experiment parameters.
        data_file: Path to the dataset file.
    """
    print("\n==== Running Full Context Baseline ====")

    method = "full_context"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(cfg.general.results_path, f"{method}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n==== Output Directory: {output_dir} ====")

    output_file_path = os.path.join(output_dir, f"{method}_output.json")

    full_context_manager = FullContextManager(cfg=cfg, data_path=data_file)
    _, duration = measure_execution_time(
        full_context_manager.process_all_conversations, output_file_path
    )
    print(f"Full context generation completed in {format_duration(duration)}")

    print("\n==== Running Evaluation ====")
    eval_file = os.path.join(output_dir, f"{method}_eval.json")
    evaluate(cfg, output_file_path, eval_file)

    print("\n==== Generating Scores ====")
    score_file = os.path.join(output_dir, f"{method}_scores.json")
    generate_scores(eval_file, score_file)


def _run_langmem_experiment(cfg: DictConfig):
    """
    Run the LangMem memory technique experiment.

    Args:
        cfg: Configuration object containing experiment parameters.
    """
    print("\n==== Running LangMem Experiment ====")

    output_file_path = os.path.join(cfg.general.output_path, "langmem_results.json")

    langmem_manager = LangMemManager(dataset_path="dataset/locomo10_rag.json")

    _, duration = measure_execution_time(
        langmem_manager.process_all_conversations, output_file_path
    )
    print(f"LangMem experiment completed in {format_duration(duration)}")


def _run_zep_experiment(cfg: DictConfig, data_file: str):
    """
    Run the Zep memory technique experiment.

    Both ``add`` and ``search`` Zep operations are supported.

    Args:
        cfg: Configuration object containing experiment parameters.
        data_file: Path to the dataset file.
    """
    print("\n==== Running Zep Experiment ====")

    if cfg.method == "add":
        print("Zep Mode: Adding conversations to memory")
        zep_manager = ZepAdd(data_path=data_file)

        _, duration = measure_execution_time(zep_manager.process_all_conversations, "1")
        print(f"Zep add operation completed in {format_duration(duration)}")

    elif cfg.method == "search":
        print("Zep Mode: Searching and evaluating memory")
        output_file_path = os.path.join(
            cfg.general.output_path, "zep_search_results.json"
        )
        zep_manager = ZepSearch()

        _, duration = measure_execution_time(
            zep_manager.process_data_file, data_file, "1", output_file_path
        )
        print(f"Zep search operation completed in {format_duration(duration)}")

    else:
        raise ValueError(
            f"Invalid Zep method: '{cfg.method}'. Supported methods: ['add', 'search']"
        )


def _run_openai_experiment(cfg: DictConfig, data_file: str):
    """
    Run the OpenAI prediction experiment.

    Args:
        cfg: Configuration object containing experiment parameters.
        data_file: Path to the dataset file.
    """
    print("\n==== Running OpenAI Experiment ====")

    output_file_path = os.path.join(cfg.general.output_path, "openai_results.json")

    openai_manager = OpenAIPredict()

    _, duration = measure_execution_time(
        openai_manager.process_data_file, data_file, output_file_path
    )
    print(f"OpenAI experiment completed in {format_duration(duration)}")


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def run(cfg: DictConfig):
    """
    Hydra entry point.

    Drives the requested memory-technique experiment based on the provided
    Hydra config and routes execution to the matching backend.

    Args:
        cfg: Hydra configuration object containing all experiment parameters.

    Raises:
        ValueError: If an unsupported memory technique type is specified.
    """
    # Configure logging — emit a per-run log file under logs_path
    logs_dir = cfg.general.get("logs_path", os.path.join(cfg.general.output_path, "logs"))
    log_file = configure_logging(log_dir=logs_dir)

    print("=" * 60)
    print("LOCOMO MEMORY EXPERIMENT RUNNER")
    print("=" * 60)
    print(f"Memory Technique: {cfg.memory.type}")
    print(f"Dataset Path: {cfg.general.data_path}")
    print(f"Output Path: {cfg.general.output_path}")
    if log_file:
        print(f"Log File: {log_file}")
    print("=" * 60)

    data_file = os.path.join(cfg.general.data_path, "locomo10.json")
    rag_data_file = os.path.join(cfg.general.data_path, "locomo10_rag.json")

    output_file = os.path.join(
        cfg.general.output_path, f"{cfg.memory.type}_output.json"
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Dispatch to the chosen backend.
    technique = cfg.memory.type
    if technique == "external_baseline":
        _run_agent_memory_experiment(cfg, data_file)
    elif technique == "mem0":
        _run_mem0_experiment(cfg, data_file, output_file)
    elif technique == "rag":
        _run_rag_experiment(cfg, data_file)
    elif technique == "full_context":
        _run_full_context_experiment(cfg, data_file)
    elif technique == "langmem":
        _run_langmem_experiment(cfg)
    elif technique == "zep":
        _run_zep_experiment(cfg, data_file)
    elif technique == "openai":
        _run_openai_experiment(cfg, data_file)
    else:
        raise ValueError(
            f"Invalid memory technique type: '{technique}'. "
            f"Supported types: ['external_baseline', 'mem0', 'rag', 'full_context', 'langmem', 'zep', 'openai']"
        )


if __name__ == "__main__":
    run()

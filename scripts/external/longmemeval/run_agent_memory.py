"""
LongMemEval experiment driver.

Provides a thin orchestration layer that builds memories with the agent_memory
backend, runs the search step, and finally invokes the LLM-judge evaluation.
"""

import logging
import os
import hydra
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Backend implementations
from providers.agent_memory.add import AgentMemoryAdd
from providers.agent_memory.search import AgentMemorySearch
from evals import evaluate
from agent_memory.utils.log import configure_logging

logger = logging.getLogger(__name__)


def run_agent_memory_experiment(
    cfg: DictConfig, debug: bool, subset_idx: int, agent_memory_store: str, retrieval_strategy: str = "semantic",
):
    """Drive a single agent_memory experiment end-to-end."""
    # Initialise logging.
    log_level = cfg.general.get("log_level", "INFO")
    configure_logging(log_level)

    logger.info("\n===== Memory Config =====\n" + OmegaConf.to_yaml(cfg.memory))
    logger.info(f"\n===== Retrieval Strategy: {retrieval_strategy} =====")

    # Push runtime overrides into the cfg so downstream code sees them.
    cfg.general.debug = debug
    cfg.eval.subset_idx = subset_idx
    cfg.memory.memory_store = agent_memory_store

    # Resolve the dataset file for this experiment.
    data_file = os.path.join(cfg.general.data_path, "longmemeval_s_cleaned.json")

    # Decide whether to (re)build the memory store.
    needs_build = cfg.memory.force_rebuild or not os.path.exists(cfg.memory.persist_path)

    if needs_build:
        print(f"\n==== Building agent_memory store in {cfg.memory.persist_path} ====")

        try:
            memory_manager = AgentMemoryAdd(cfg, data_path=data_file)
            # Add step is parallelised across questions internally.
            memory_manager.process_all_conversations()

        except Exception as exc:
            logger.error(f"Error building memory: {str(exc)}")
            raise exc

    logger.info(f"\n==== Memory building completed ====")

    # Compose the per-run output directory.
    method = f"agent_memory_{retrieval_strategy}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(cfg.general.results_path, f"{method}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"\n==== Output Directory: {output_dir} ====")

    # Snapshot the resolved config alongside results.
    params_file = os.path.join(output_dir, "parameters.yaml")
    with open(params_file, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Experiment parameters saved to: {params_file}")

    output_file = os.path.join(output_dir, f"{method}_output.json")

    # Search step.
    try:
        memory_searcher = AgentMemorySearch(cfg, output_file, cfg.memory.top_k, retrieval_strategy=retrieval_strategy)
        memory_searcher.process_data_file(data_file)
    except Exception as exc:
        logger.error(f"Error during memory search and evaluation: {str(exc)}")
        raise exc

    # Evaluation step.
    logger.info(f"\n==== Start evaluation ====")
    eval_file = os.path.join(output_dir, f"{method}_eval.json")
    evaluate(cfg, output_file, eval_file, use_question_type_eval=cfg.eval.use_question_type_eval)


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def run(cfg: DictConfig):
    """
    Hydra entry point that simply forwards the resolved config into the
    main experiment runner.
    """

    run_agent_memory_experiment(
        cfg,
        debug=cfg.general.debug,
        subset_idx=cfg.eval.subset_idx,
        agent_memory_store=cfg.memory.memory_store,
        retrieval_strategy=cfg.retrieval.strategy,
    )


if __name__ == "__main__":
    run()

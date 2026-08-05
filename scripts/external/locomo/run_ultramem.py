"""
Locomo Experiment Runner

Comprehensive experiment harness for evaluating different memory techniques
on the Locomo dataset. Supports multiple memory backends including the
ultramem backend, Mem0, and RAG-based pipelines.
"""

import logging
import os
import time
import json
import hydra
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Memory system implementations
from providers.ultramem.add import UltraMemAdd
from providers.ultramem.search import UltraMemSearch
from evals import evaluate, generate_scores
from ultramem.utils.log import configure_logging
from analyze_latency import analyze_and_save_latency

logger = logging.getLogger(__name__)


def run_ultramem_experiment(
    cfg: DictConfig, debug: bool, subset_idx: int, ultramem_store: str, retrieval_strategy: str = "semantic",
):
    """Top-level driver for an ultramem experiment run."""
    log_level = cfg.general.get("log_level", "INFO")
    configure_logging(log_level)

    logger.info("\n===== Memory Config =====\n" + OmegaConf.to_yaml(cfg.memory))
    logger.info(f"\n===== Retrieval Strategy: {retrieval_strategy} =====")

    cfg.general.debug = debug
    cfg.eval.subset_idx = subset_idx
    cfg.memory.memory_store = ultramem_store

    data_file = os.path.join(cfg.general.data_path, "locomo10.json")

    # Set up the output directory up-front so timing artefacts can land in it
    method = f"ultramem_{retrieval_strategy}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(cfg.general.results_path, f"{method}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"\n==== Output Directory: {output_dir} ====")

    build_memory = (
        not os.path.exists(cfg.memory.persist_path) or cfg.memory.force_rebuild
    )

    if build_memory:
        # Time the memory-building stage
        print(f"\n==== Building UltraMem in {cfg.memory.persist_path} ====")

        build_start_time = time.time()
        print(f"Memory building started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            memory_manager = UltraMemAdd(cfg, data_path=data_file)
            # for idx, item in tqdm(enumerate(memory_manager.data)):
            #     memory_manager.process_conversation(item, idx)

            # Process all conversations in parallel
            memory_manager.process_all_conversations()

        except Exception as exc:
            logger.error(f"Error building memory: {str(exc)}")
            raise exc

        build_end_time = time.time()
        build_duration = build_end_time - build_start_time
        print(f"Memory building completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        # Render duration as h:m:s
        build_duration_str = time.strftime("%H:%M:%S", time.gmtime(build_duration))

        timing_file = os.path.join(output_dir, "build_timing.json")

        timing_data = {
            "start_time": datetime.fromtimestamp(build_start_time).strftime('%Y-%m-%d %H:%M:%S'),
            "end_time": datetime.fromtimestamp(build_end_time).strftime('%Y-%m-%d %H:%M:%S'),
            "duration": build_duration_str,
            "memory_store": cfg.memory.memory_store,
        }

        with open(timing_file, 'w') as fh:
            json.dump(timing_data, fh, indent=2)

        logger.info(f"Memory building timing saved to: {timing_file}")

    logger.info(f"\n==== Memory building completed ====")

    # Persist the experiment parameters
    params_file = os.path.join(output_dir, "parameters.yaml")
    with open(params_file, "w") as fh:
        fh.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Experiment parameters saved to: {params_file}")

    output_file = os.path.join(output_dir, f"{method}_output.json")

    try:
        memory_searcher = UltraMemSearch(cfg, output_file, cfg.memory.top_k, retrieval_strategy=retrieval_strategy)
        memory_searcher.process_data_file(data_file)
    except Exception as exc:
        logger.error(f"Error during memory search and evaluation: {str(exc)}")
        raise exc

    logger.info(f"\n==== Start evaluation ====")
    eval_file = os.path.join(output_dir, f"{method}_eval.json")
    evaluate(cfg, output_file, eval_file)

    logger.info(f"\n==== Start generating score table ====")
    score_file = os.path.join(output_dir, f"{method}_scores.json")
    generate_scores(eval_file, score_file)

    logger.info(f"\n==== Start latency analysis ====")
    analyze_and_save_latency(output_file, output_dir)


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def run(cfg: DictConfig):
    """
    Hydra entry point.

    Drives execution of the memory-technique experiments configured by the
    Hydra config. Backends and timing/logging are dispatched downstream.

    Args:
        cfg: Hydra configuration object containing all experiment parameters.

    Raises:
        ValueError: If an unsupported memory technique type is specified.
    """

    run_ultramem_experiment(
        cfg,
        debug=cfg.general.debug,
        subset_idx=cfg.eval.subset_idx,
        ultramem_store=cfg.memory.memory_store,
        retrieval_strategy=cfg.retrieval.strategy,
    )


if __name__ == "__main__":
    run()

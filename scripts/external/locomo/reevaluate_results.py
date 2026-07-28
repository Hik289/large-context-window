"""
Re-evaluation Script for agent_memory Results

Use this script to re-run the evaluation step against existing output files
without re-running memory extraction or answering. It's handy when you want
to swap the evaluation model or other eval-time parameters.

It can also regenerate answers with a different prompt template (e.g. switch
from the mem0 template to evermemos) without retrieving memories again.

Usage:
    # Re-run evaluation with a different judge model only
    python reevaluate_results.py --results_dir <path_to_results_folder> --eval_model <model_name>

    # Regenerate answers with a different template AND re-run evaluation
    python reevaluate_results.py --results_dir <path_to_results_folder> --regenerate_answers --answer_template evermemos

    # Regenerate answers using a different generation model
    python reevaluate_results.py --results_dir <path_to_results_folder> --regenerate_answers --answer_model YOUR_CHAT_MODEL

Example:
    python reevaluate_results.py --results_dir results/run_eval --eval_model YOUR_JUDGE_MODEL
    python reevaluate_results.py --results_dir results/run_eval --regenerate_answers --answer_template evermemos --eval_model YOUR_JUDGE_MODEL
    python reevaluate_results.py --results_dir results/run_eval --regenerate_answers --answer_model YOUR_CHAT_MODEL --eval_model YOUR_JUDGE_MODEL
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import hydra
from omegaconf import DictConfig, OmegaConf
from jinja2 import Template
from tqdm import tqdm

# Make sure local modules are importable
sys.path.insert(0, str(Path(__file__).parent))

from evals import evaluate, generate_scores
from agent_memory.utils.log import configure_logging
from agent_memory.utils.llm import get_general_chat_completion_client
from prompts import ANSWER_PROMPT_COMBINED, ANSWER_PROMPT_EVERMEMOS

logger = logging.getLogger(__name__)


def regenerate_answer(
    question: str,
    formatted_memories: list,
    template_name: str,
    llm_client,
    llm_model: str,
    llm_seed: int,
) -> tuple:
    """
    Build a fresh answer for *question* from already-stored memories.

    Args:
        question: The question to answer.
        formatted_memories: Pre-formatted memory strings to feed the LLM.
        template_name: Either ``'mem0'`` or ``'evermemos'``.
        llm_client: Shared LLM client.
        llm_model: Model name used for generation.
        llm_seed: Random seed for deterministic output.

    Returns:
        Tuple of (answer, response_time).
    """
    # Pick the template
    if template_name == "mem0":
        template_str = ANSWER_PROMPT_COMBINED
    elif template_name == "evermemos":
        template_str = ANSWER_PROMPT_EVERMEMOS
    else:
        raise ValueError(f"Unknown template: {template_name}. Use 'mem0' or 'evermemos'.")

    template = Template(template_str)
    answer_prompt = template.render(
        memories=json.dumps(formatted_memories, indent=4),
        question=question,
    )

    started_at = time.time()
    try:
        response = llm_client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": answer_prompt}],
            temperature=0.0,
            seed=llm_seed,
        )
    except Exception as exc:
        logger.error(f"LLM call failed for question '{question}': {exc}")
        finished_at = time.time()
        return f"ERROR: LLM call failed - {str(exc)}", finished_at - started_at

    finished_at = time.time()
    response_time = finished_at - started_at

    if response and response.choices and response.choices[0].message.content:
        outcome = response.choices[0].message.content.strip()
        if not outcome:
            outcome = "ERROR: Empty response from LLM."
            logger.error("Empty response from LLM.")
    else:
        outcome = "ERROR: No response from LLM."
        logger.error("No response from LLM.")

    if "FINAL ANSWER:" in outcome:
        chunks = outcome.split("FINAL ANSWER:")
        if len(chunks) > 1:
            outcome = chunks[1].strip()

    return outcome, response_time


def regenerate_answers_from_output(
    input_file: str,
    output_file: str,
    template_name: str,
    llm_client,
    llm_model: str,
    llm_seed: int,
):
    """
    Walk over a previous output file and regenerate every answer using the
    stored memories and the requested template.

    Args:
        input_file: Path to the original output JSON file.
        output_file: Path to write the new output (with regenerated answers).
        template_name: Either ``'mem0'`` or ``'evermemos'``.
        llm_client: LLM client used for generation.
        llm_model: Model name used for generation.
        llm_seed: Random seed for reproducibility.
    """
    logger.info(f"\n==== Regenerating Answers ====")
    logger.info(f"Template: {template_name}")
    logger.info(f"Answer generation model: {llm_model}")

    with open(input_file, 'r') as fh:
        data = json.load(fh)

    new_results = defaultdict(list)

    for conv_id, qa_items in tqdm(data.items(), desc="Processing conversations"):
        for entry in tqdm(qa_items, desc=f"Regenerating answers for conversation {conv_id}", leave=False):
            question = entry['question']

            # Pull the pre-formatted memories that were stored on the original run
            formatted_memories = entry.get('formatted_speaker_1_memories', [])

            # If we have no formatted memories there's nothing to regenerate against
            if not formatted_memories:
                logger.warning(f"No formatted memories found for question: {question}")
                new_results[conv_id].append(entry)
                continue

            new_response, response_time = regenerate_answer(
                question=question,
                formatted_memories=formatted_memories,
                template_name=template_name,
                llm_client=llm_client,
                llm_model=llm_model,
                llm_seed=llm_seed,
            )

            new_entry = entry.copy()
            new_entry['response'] = new_response
            new_entry['response_time'] = response_time
            new_entry['regenerated_with_template'] = template_name
            new_entry['regenerated_with_model'] = llm_model

            new_results[conv_id].append(new_entry)

    with open(output_file, 'w') as fh:
        json.dump(new_results, fh, indent=4)

    logger.info(f"Regenerated answers saved to: {output_file}")


def reevaluate_results(
    results_dir: str,
    eval_model: str = None,
    max_workers: int = None,
    output_suffix: str = None,
    regenerate_answers: bool = False,
    answer_template: str = None,
    answer_model: str = None,
):
    """
    Re-evaluate an existing agent_memory results folder using a different
    evaluation model. Optionally regenerate answers with a new template first.

    Args:
        results_dir: Path to the results directory containing the output file.
        eval_model: Evaluation model identifier.
        max_workers: Number of evaluation workers.
        output_suffix: Optional suffix for the new files (defaults to a timestamp).
        regenerate_answers: When ``True``, regenerate answers with a new template.
        answer_template: Either ``'mem0'`` or ``'evermemos'`` (required when
            ``regenerate_answers`` is True).
        answer_model: Model used for answer generation (overrides config default).
    """

    results_path = Path(results_dir)
    if not results_path.exists():
        raise ValueError(f"Results directory does not exist: {results_dir}")

    # Output file is the JSON ending with ``_output.json``
    output_files = list(results_path.glob("*_output.json"))
    if not output_files:
        raise ValueError(f"No output file found in {results_dir}")

    if len(output_files) > 1:
        logger.warning(f"Multiple output files found, using: {output_files[0]}")

    original_output_file = str(output_files[0])
    logger.info(f"Found output file: {original_output_file}")

    with hydra.initialize(version_base=None, config_path="./conf"):
        cfg = hydra.compose(config_name="config")

    if eval_model:
        cfg.eval.model = eval_model
        logger.info(f"Using evaluation model: {eval_model}")
    else:
        logger.info(f"Using default evaluation model from config: {cfg.eval.model}")

    if max_workers:
        cfg.eval.max_workers = max_workers
        logger.info(f"Using max_workers: {max_workers}")

    if output_suffix is None:
        output_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_name = output_files[0].stem.replace("_output", "")

    if regenerate_answers:
        if not answer_template:
            raise ValueError("Must specify --answer_template when using --regenerate_answers")

        answer_gen_model = answer_model if answer_model else cfg.llm.model

        logger.info(f"\n==== Regenerating Answers ====")
        logger.info(f"Answer template: {answer_template}")
        logger.info(f"Answer generation model: {answer_gen_model}")

        regenerated_output_file = str(
            results_path / f"{base_name}_output_{answer_template}_{output_suffix}.json"
        )

        llm_client = get_general_chat_completion_client(cfg)

        regenerate_answers_from_output(
            input_file=original_output_file,
            output_file=regenerated_output_file,
            template_name=answer_template,
            llm_client=llm_client,
            llm_model=answer_gen_model,
            llm_seed=cfg.llm.seed,
        )

        # Hand the regenerated output to the evaluation step
        input_file = regenerated_output_file
        eval_suffix = f"{answer_template}_{output_suffix}"
    else:
        input_file = original_output_file
        eval_suffix = output_suffix

    eval_file = str(results_path / f"{base_name}_eval_{eval_suffix}.json")
    score_file = str(results_path / f"{base_name}_scores_{eval_suffix}.json")

    logger.info(f"\n==== Starting Re-evaluation ====")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Evaluation model: {cfg.eval.model}")
    logger.info(f"Output eval file: {eval_file}")
    logger.info(f"Output score file: {score_file}")

    logger.info(f"\n==== Running evaluation ====")
    evaluate(cfg, input_file, eval_file)

    logger.info(f"\n==== Generating score table ====")
    scores_data = generate_scores(eval_file, score_file)

    logger.info(f"\n==== Re-evaluation Complete ====")
    logger.info(f"Evaluation results saved to: {eval_file}")
    logger.info(f"Score summary saved to: {score_file}")

    metadata = {
        "original_output_file": original_output_file,
        "re_evaluation_timestamp": datetime.now().isoformat(),
        "evaluation_model": cfg.eval.model,
        "max_workers": cfg.eval.max_workers,
        "eval_file": eval_file,
        "score_file": score_file,
        "regenerated_answers": regenerate_answers,
    }

    if regenerate_answers:
        metadata["answer_template"] = answer_template
        metadata["answer_generation_model"] = answer_gen_model
        metadata["regenerated_output_file"] = input_file

    metadata_file = str(results_path / f"{base_name}_reevaluation_metadata_{eval_suffix}.json")
    with open(metadata_file, 'w') as fh:
        json.dump(metadata, fh, indent=4)

    logger.info(f"Metadata saved to: {metadata_file}")

    return eval_file, score_file, scores_data


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate existing agent_memory results with optional answer regeneration"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to the results directory containing the output file",
    )
    parser.add_argument(
        "--eval_model",
        type=str,
        default=None,
        help="Evaluation model to use. If not specified, uses config default.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Number of parallel workers for evaluation. If not specified, uses config default.",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default=None,
        help="Optional suffix for output files (default: timestamp)",
    )
    parser.add_argument(
        "--regenerate_answers",
        action="store_true",
        help="Regenerate answers using stored memories with a different template",
    )
    parser.add_argument(
        "--answer_template",
        type=str,
        default=None,
        choices=["mem0", "evermemos"],
        help="Template to use for answer generation (required if --regenerate_answers is set)",
    )
    parser.add_argument(
        "--answer_model",
        type=str,
        default=None,
        help="Model to use for answer generation. If not specified, uses config default (llm.model).",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    args = parser.parse_args()

    configure_logging(args.log_level)

    if args.regenerate_answers and not args.answer_template:
        parser.error("--answer_template is required when --regenerate_answers is set")

    try:
        reevaluate_results(
            results_dir=args.results_dir,
            eval_model=args.eval_model,
            max_workers=args.max_workers,
            output_suffix=args.output_suffix,
            regenerate_answers=args.regenerate_answers,
            answer_template=args.answer_template,
            answer_model=args.answer_model,
        )
    except Exception as exc:
        logger.error(f"Re-evaluation failed: {str(exc)}")
        raise


if __name__ == "__main__":
    main()

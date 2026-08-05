#!/usr/bin/env python3
"""
Test Case Runner Utility

A small CLI for executing one or more named test cases against the Locomo
memory-evaluation framework.

Usage:
    python run_test_cases.py --test "Caroline Research Test"
    python run_test_cases.py --batch
    python run_test_cases.py --list
"""

import argparse
import os
import sys
import yaml
from typing import List, Optional

# Make the parent locomo directory importable.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.locomo.unittest.run_single_test_case import load_test_cases, get_test_case_by_name, run_single_test_case
import hydra
from omegaconf import DictConfig


def list_available_test_cases() -> None:
    """Print every test case found in the YAML configuration."""
    try:
        test_config = load_test_cases()
        test_cases = test_config.get("test_cases", [])

        print("\\n" + "=" * 60)
        print("AVAILABLE TEST CASES")
        print("=" * 60)

        for pos, tc in enumerate(test_cases, 1):
            print(f"{pos}. {tc['name']}")
            print(f"   Description: {tc['description']}")
            print(f"   Question: {tc['question']}")
            print(f"   Config: conv_idx={tc['conversation_idx']}, "
                  f"session_idx={tc['session_idx']}")
            print()

        default_test = test_config.get("default_test_case", "None")
        print(f"Default test case: {default_test}")

    except Exception as exc:
        print(f"Error loading test cases: {exc}")


def run_specific_test_case(test_name: str, cfg: DictConfig) -> None:
    """Run exactly one test case by name."""
    try:
        test_config = load_test_cases()
        test_case = get_test_case_by_name(test_config, test_name)
        result = run_single_test_case(cfg, test_case)

        print(f"\\nTest '{test_name}' completed successfully!")
        return result

    except Exception as exc:
        print(f"Error running test case '{test_name}': {exc}")
        return None


def run_batch_tests(cfg: DictConfig, selected_tests: Optional[List[str]] = None) -> None:
    """Run a list of test cases sequentially."""
    try:
        test_config = load_test_cases()

        if selected_tests is None:
            # Fall back to whatever the YAML batch config specifies.
            batch_config = test_config.get("batch_config", {})
            selected_tests = batch_config.get("selected_tests", [])

        if not selected_tests:
            print("No test cases specified for batch execution.")
            return

        results = []
        print(f"\\nRunning {len(selected_tests)} test cases in batch mode...")

        for test_name in selected_tests:
            try:
                test_case = get_test_case_by_name(test_config, test_name)
                outcome = run_single_test_case(cfg, test_case)
                results.append(outcome)
            except Exception as exc:
                print(f"Error running test case '{test_name}': {exc}")
                continue

        # Summarize the batch results.
        print(f"\\n{'=' * 60}")
        print("BATCH TEST RESULTS SUMMARY")
        print(f"{'=' * 60}")

        for pos, outcome in enumerate(results, 1):
            print(f"{pos}. {outcome['test_case_name']}")
            print(f"   Question: {outcome['question']}")
            print(f"   Answer: {outcome['answer']}")
            print()

        print(f"Completed {len(results)}/{len(selected_tests)} test cases successfully.")

    except Exception as exc:
        print(f"Error in batch execution: {exc}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Parse CLI flags and dispatch to the appropriate runner."""
    parser = argparse.ArgumentParser(
        description="Run Locomo memory evaluation test cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_test_cases.py --list
  python run_test_cases.py --test "Caroline Research Test"
  python run_test_cases.py --batch
  python run_test_cases.py --batch --tests "Caroline Research Test" "Speaker Interaction Test"
        """,
    )

    parser.add_argument("--list", action="store_true",
                        help="List all available test cases")
    parser.add_argument("--test", type=str, metavar="TEST_NAME",
                        help="Run a specific test case by name")
    parser.add_argument("--batch", action="store_true",
                        help="Run test cases in batch mode")
    parser.add_argument("--tests", nargs="+", metavar="TEST_NAME",
                        help="Specify test cases for batch mode")

    args = parser.parse_args()

    # Configure logging once before doing any work.
    from ultramem.utils.log import configure_logging
    configure_logging()
    cfg.general.debug = True

    if args.list:
        list_available_test_cases()
    elif args.test:
        run_specific_test_case(args.test, cfg)
    elif args.batch:
        run_batch_tests(cfg, args.tests)
    else:
        print("No action specified. Use --help for usage information.")
        list_available_test_cases()


if __name__ == "__main__":
    main()

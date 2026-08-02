"""
agent_memory CLI

Top-level command-line entry point for the memory framework.
"""

import sys
import argparse
import logging

from . import __version__

logger = logging.getLogger(__name__)


def _show_config(_args):
    """Print the active model config and its non-secret alias mapping."""
    from .methods.configs.model_resolver import list_aliases, models_config_path

    print(f"Model config: {models_config_path()}")
    for alias, status in sorted(list_aliases().items()):
        print(f"  {alias}: {status}")


def _dispatch_browser(args):
    """Forward the ``browser`` subcommand to the dedicated module CLI."""
    from .browser.__main__ import main as browser_main

    forwarded: list[str] = [args.db_path]

    flag_pairs = (
        ("collection", "-c"),
        ("search", "--search"),
        ("output", "--output"),
    )
    for attr, flag in flag_pairs:
        value = getattr(args, attr, None)
        if value:
            forwarded.extend([flag, value])

    if args.limit:
        forwarded.extend(["--limit", str(args.limit)])

    bool_flags = (
        ("stats", "--stats"),
        ("analyze", "--analyze"),
        ("verbose", "--verbose"),
        ("no_interactive", "--no-interactive"),
    )
    for attr, flag in bool_flags:
        if getattr(args, attr, False):
            forwarded.append(flag)

    saved_argv = sys.argv
    try:
        sys.argv = ["agent-memory-browser"] + forwarded
        browser_main()
    finally:
        sys.argv = saved_argv


def main():
    """Top-level entry point for the agent_memory CLI."""
    from .utils.log import configure_logging

    parser = argparse.ArgumentParser(
        description="UltraMem - Source-resolved memory retrieval beyond the prompt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive memory browser
  agent-memory browser /path/to/memory_store

  # Browse specific collection
  agent-memory browser /path/to/memory_store -c agent_memory

  # Quick statistics
  agent-memory browser /path/to/memory_store --stats

  # Search memories
  agent-memory browser /path/to/memory_store --search "incident management"

  # Generate analysis report
  agent-memory browser /path/to/memory_store --analyze --output report.json
        """
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'agent_memory {__version__}'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    browser_parser = subparsers.add_parser(
        'browser',
        help='Interactive memory browser',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive browser
  agent-memory browser /path/to/memory_store

  # Browse specific collection
  agent-memory browser /path/to/memory_store -c agent_memory

  # Quick statistics
  agent-memory browser /path/to/memory_store --stats

  # Search and analyze
  agent-memory browser /path/to/memory_store --search "procedures"
        """
    )

    browser_parser.add_argument(
        'db_path',
        help='Path to ChromaDB database directory'
    )
    browser_parser.add_argument(
        '--collection', '-c',
        help='Name of collection to browse'
    )
    browser_parser.add_argument(
        '--stats',
        action='store_true',
        help='Show quick statistics and exit'
    )
    browser_parser.add_argument(
        '--analyze',
        action='store_true',
        help='Generate comprehensive analysis report'
    )
    browser_parser.add_argument(
        '--search', '-s',
        help='Search query to analyze'
    )
    browser_parser.add_argument(
        '--output', '-o',
        help='Output file path for results'
    )
    browser_parser.add_argument(
        '--limit', '-l',
        type=int,
        default=10,
        help='Limit number of results (default: 10)'
    )
    browser_parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='Disable interactive mode'
    )

    browser_parser.set_defaults(func=_dispatch_browser)

    config_parser = subparsers.add_parser(
        'config',
        help='Show the active model-alias configuration',
    )
    config_parser.set_defaults(func=_show_config)

    args = parser.parse_args()

    configure_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not hasattr(args, 'func'):
        parser.print_help()
        print("\nTry: agent-memory browser /path/to/memory_store")
        sys.exit(1)

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)
    except Exception as exc:
        logger.error(f"Command failed: {exc}")
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

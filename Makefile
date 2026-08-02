.PHONY: install test self-test build figures check

install:
	python -m pip install -e ".[dev]"

test:
	pytest

self-test:
	python -m agent_memory.methods.dual_node
	python -m agent_memory.methods.token_ledger
	python -m agent_memory.methods.configs.isolation

build:
	python -m build

figures:
	python figures/make_all_figs.py

check: test self-test build

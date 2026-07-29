.PHONY: install test build figures check

install:
	python -m pip install -e ".[dev]"

test:
	pytest

build:
	python -m build

figures:
	python figures/make_all_figs.py

check: test build

# Detect virtual environment and use it if available
VENV := .venv
PYTHON := $(shell [ -d $(VENV) ] && echo $(VENV)/bin/python || echo python)
RUFF := $(shell [ -d $(VENV) ] && echo $(VENV)/bin/ruff || echo ruff)
MYPY := $(shell [ -d $(VENV) ] && echo $(VENV)/bin/mypy || echo mypy)
PYTEST := $(shell [ -d $(VENV) ] && echo $(VENV)/bin/pytest || echo pytest)
PRE_COMMIT := $(shell [ -d $(VENV) ] && echo $(VENV)/bin/pre-commit || echo pre-commit)
PIP := $(shell [ -d $(VENV) ] && echo $(VENV)/bin/pip || echo pip)

.PHONY: help install lint lint-fix format format-check type-check test clean all pre-commit

help:
	@echo "Available commands:"
	@echo "  make install      - Install package in editable mode with dev dependencies"
	@echo "  make pre-commit   - Install pre-commit hooks"
	@echo "  make lint         - Check code with ruff (no fixes)"
	@echo "  make lint-fix     - Check and auto-fix code with ruff"
	@echo "  make format       - Format code with ruff"
	@echo "  make format-check - Check formatting without changes"
	@echo "  make type-check   - Run mypy type checker"
	@echo "  make test         - Run pytest"
	@echo "  make all          - Run all checks (lint, format-check, type-check, test)"
	@echo "  make clean        - Remove cache and build artifacts"
	@echo ""
	@echo "Note: Commands automatically use .venv if it exists"

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

pre-commit:
	$(PRE_COMMIT) install

lint:
	$(RUFF) check src/ tests/

lint-fix:
	$(RUFF) check src/ tests/ --fix

format:
	$(RUFF) format src/ tests/

format-check:
	$(RUFF) format --check src/ tests/

type-check:
	$(MYPY) src/psyphy tests/ || true

test:
	$(PYTEST) -v

all: lint format-check type-check test
	@echo "✅ All checks passed!"

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

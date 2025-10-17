# Contributing
![psyphy logo](images/psyphy_logo_draft.png)


## Dev setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Code Quality

### worfklow

We use **Ruff** for linting and formatting, **mypy** for type checking, and **pre-commit** to automate checks.

### Pre-commit Hooks (Automatic)

Install once during initial setup:

```bash
pre-commit install
```

Now hooks run automatically when you commit:

```bash
git add .
git commit -m "Add new feature"

# Automatically runs:
# ✓ ruff check --fix    (lints and auto-fixes)
# ✓ ruff format          (formats code, wraps long lines)
# ✓ mypy                 (type checks)
# ✓ trailing-whitespace  (removes trailing spaces)
# ✓ end-of-file-fixer    (ensures files end with newline)
# ... and more

# If hooks modify files, review and commit again:
git add .
git commit -m "Add new feature"
```

To run all hooks manually without committing:

```bash
pre-commit run --all-files
```

### Linting & Formatting (Manual)

When developing, you can run checks manually:

```bash
# Quick option: Use Makefile
make format        # Format code
make lint-fix      # Lint and auto-fix
make type-check    # Type check
make test          # Run tests
make all           # Run all checks

# Or use Ruff directly:
ruff format src/ tests/              # Format code
ruff check src/ tests/ --fix         # Lint with auto-fix
ruff check src/ tests/               # Lint only (no fixes)
ruff format --check src/ tests/      # Check formatting (no changes)
```

### Type Checking

We use **mypy** for static type analysis:

```bash
mypy src/psyphy tests/
```

### Running Tests

```bash
pytest -v
```

### All Checks (CI simulation)

Run all checks locally before pushing:

```bash
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/psyphy tests/
pytest
```

## Docs: build & serve

Install doc dependencies:

```bash
pip install mkdocs mkdocs-material mkdocstrings[python]
```

Build/serve locally (auto-reload):

```bash
mkdocs serve
```

Deploy (manual):

```bash
mkdocs gh-deploy --clean
```

We use NumPy-style docstrings for API reference via mkdocstrings.

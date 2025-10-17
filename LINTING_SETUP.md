# Linting Setup Summary

This document summarizes the linting, formatting, and code quality setup we use for `psyphy`.


### 1. Tools & Configuration

- **Ruff** (v0.9.1): Modern, fast Python linter and formatter
  - Replaces: Black, isort, flake8, pylint
  - Configured in `pyproject.toml`

- **mypy** (v1.11.2): Static type checker
  - Configured in `pyproject.toml`

- **pre-commit**: Git hooks for automated checks
  - Configuration in `.pre-commit-config.yaml`

### 2. Files

- `.pre-commit-config.yaml`
    - Pre-commit hooks configuration
- `.github/workflows/lint.yml`
    - CI workflow for automated checks
- `Makefile`
    - Convenient commands for development


- `pyproject.toml`
    - Added Ruff, mypy, and pytest configuration
- `requirements.txt`
    - Replaced `black` with `mypy`
- `docs/CONTRIBUTING.md`
    - now also inlcudeds  linting guidelines
- All Python files in `src/` and `tests/`
     - Auto-formatted and modernized

### 3. Ruff Configuration Highlights

**Selected Rules:**
- `E`, `W` - pycodestyle (PEP 8 compliance)
- `F` - pyflakes (detect undefined names, unused imports)
- `I` - isort (import sorting)
- `B` - flake8-bugbear (find likely bugs)
- `C4` - flake8-comprehensions (better comprehensions)
- `UP` - pyupgrade (modern Python syntax)
- `SIM` - flake8-simplify (simplify code)
- `NPY` - NumPy-specific rules

**Ignored (for now):**
- `E501` - Line too long (handled by formatter)
- `ARG` - Unused arguments (prototyping phase)


## Workflow: How It All Works Together

### Visual Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Development Phase                                          │
├─────────────────────────────────────────────────────────────┤
│  You write code                                             │
│    ↓                                                        │
│  make format / ruff format    → Auto-formats code          │
│    ↓                                                        │
│  make lint-fix / ruff check --fix → Auto-fixes issues      │
│    ↓                                                        │
│  make test / pytest           → Run tests                  │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│  Commit Phase (Automated)                                   │
├─────────────────────────────────────────────────────────────┤
│  git commit -m "..."                                        │
│    ↓                                                        │
│  Pre-commit hooks run automatically:                        │
│    • ruff check --fix                                       │
│    • ruff format                                            │
│    • mypy                                                   │
│    • trailing-whitespace                                    │
│    • end-of-file-fixer                                      │
│    • check-yaml, check-toml, etc.                          │
│    ↓                                                        │
│  If files modified: review → git add → git commit again    │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│  Push Phase                                                 │
├─────────────────────────────────────────────────────────────┤
│  git push                                                   │
│    ↓                                                        │
│  GitHub Actions CI runs:                                    │
│    • ruff check (lint)                                      │
│    • ruff format --check (format check)                     │
│    • mypy (type check)                                      │
│    • pytest (tests)                                         │
│    ↓                                                        │
│  ✓ Checks pass → PR can be merged                          │
│  ✗ Checks fail → Fix locally and push again                │
└─────────────────────────────────────────────────────────────┘
```

### Pre-commit Hooks (Automatic)

The hooks run automatically when you commit. Here's what happens:

```bash
git add .
git commit -m "Your changes"

# Automatically runs (in order):
# 1. ruff check --fix    → Lints and auto-fixes issues
# 2. ruff format          → Formats code (wraps long lines, fixes spacing)
# 3. mypy                 → Type checks your code
# 4. trailing-whitespace  → Removes trailing spaces
# 5. end-of-file-fixer    → Ensures files end with newline
# 6. check-yaml           → Validates YAML files
# 7. check-toml           → Validates TOML files
# 8. check-merge-conflict → Checks for merge conflict markers
# 9. debug-statements     → Finds leftover debug statements

# If any hooks modify files:
git add .               # Stage the auto-fixed files
git commit -m "..."     # Commit again
```

**Pro tip:** The hooks will auto-fix most issues, so you just need to review and re-commit.

### Manual Development

When working on code, run checks manually:

```bash
# Option 1: Use Makefile (recommended)
make format        # Format code first
make lint-fix      # Fix linting issues
make type-check    # Check types (optional during dev)
make test          # Run tests

# Option 2: Direct commands
ruff format src/ tests/              # Format (wraps long lines, fixes spacing)
ruff check src/ tests/ --fix         # Lint and auto-fix
mypy src/psyphy tests/               # Type check
pytest                               # Test
```

### Before Pushing (CI Simulation)

Run all checks locally to match what CI will run:

```bash
make all
# Or manually:
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/psyphy tests/
pytest
```

### Manual Commands Reference

```bash
# Quick commands via Makefile
make lint          # Check for issues
make lint-fix      # Auto-fix issues
make format        # Format code
make type-check    # Run mypy
make test          # Run tests
make all           # Run all checks

# Direct commands
ruff check src/ tests/              # Lint
ruff check src/ tests/ --fix        # Lint with auto-fix
ruff format src/ tests/             # Format
mypy src/psyphy tests/              # Type check
pytest                              # Test
```

### Run All Pre-commit Hooks Manually

```bash
pre-commit run --all-files
```

### CI/CD

GitHub Actions workflow (`.github/workflows/lint.yml`) runs on:
- Push to `main`
- Pull requests to `main`

Checks that will be performed:
- Linting (ruff check)
- Format checking (ruff format --check)
- Type checking (mypy) - currently set to continue-on-error
- Tests (pytest)


## Next Steps

1. **Stricter Type Checking**: Once we add more type hints, set `disallow_untyped_defs = true` in mypy config
2. **Enable ARG Rules**: When out of prototyping, remove `ARG` from ignored rules
3. **Documentation Linting**: Add `pydocstyle` or `D` rules for docstring checks

## Troubleshooting

**Pre-commit hooks fail:**
- Review the error message
- Fix issues manually or with `ruff check --fix`
- Stage fixed files and commit again

**Skip hooks temporarily:**
```bash
git commit --no-verify
```

**Update pre-commit hooks:**
```bash
pre-commit autoupdate
```

**Clear caches:**
```bash
make clean
pre-commit clean
```

## References

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [mypy Documentation](https://mypy.readthedocs.io/)
- [pre-commit Documentation](https://pre-commit.com/)

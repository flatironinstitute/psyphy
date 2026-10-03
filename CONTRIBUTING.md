

# Contributing to psyphy

We welcome contributions from both experienced developers and researchers who may be newer to collaborative software workflows. This guide provides both a quick workflow and a more detailed, step-by-step path.

---

## Quick Start (TL;DR)

If you’re already familiar with GitHub workflows, this is the minimal path. For a detailed version scroll down to "Standard Workflow (Step-by-Step)".
```
# fork repo on GitHub first
git clone https://github.com/YOUR_USERNAME/psyphy.git   # clone your fork (origin)
cd psyphy
git remote add upstream https://github.com/flatironinstitute/psyphy.git  # add canonical repo
git checkout main
git fetch upstream
git merge upstream/main    # sync local main with upstream
git checkout -b feature/my-feature
# make changes
git add .
git commit -m "Add feature"
git push origin feature/my-feature   # push to your fork (origin)
# open PR: origin -> upstream (GitHub UI)
```
---


## Development Tools

We use **Ruff** for linting and formatting, **mypy** for type checking, and
**pre-commit** to automate both. See
[LINTING_SETUP.md](https://github.com/flatironinstitute/psyphy/blob/main/LINTING_SETUP.md)
for the full configuration.

### Pre-commit hooks (automatic)

Install once during setup:
```bash
pre-commit install
```
The hooks then run on every commit:
```bash
git add .
git commit -m "Add new feature"

# Automatically runs:
# ✓ ruff check --fix     (lints and auto-fixes)
# ✓ ruff format          (formats code, wraps long lines)
# ✓ mypy                 (type checks)
# ✓ trailing-whitespace  (removes trailing spaces)
# ✓ end-of-file-fixer    (ensures files end with newline)
# ... and more
```
If a hook modifies files, the commit aborts — review, re-stage and commit again:
```bash
git add .
git commit -m "Add new feature"
```
To run every hook without committing:
```bash
pre-commit run --all-files
```

### Running checks manually

```bash
# via the Makefile
make format        # format code
make lint-fix      # lint and auto-fix
make type-check    # type check
make test          # run tests
make all           # everything

# or the tools directly
ruff format src/ tests/              # format
ruff check src/ tests/ --fix         # lint with auto-fix
ruff check src/ tests/               # lint only
ruff format --check src/ tests/      # check formatting, change nothing
mypy src/psyphy tests/               # type check
pytest -v                            # tests
```

### Before pushing (what CI runs)

```bash
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/psyphy tests/
pytest
```
---

## Code Standards

* Keep PRs focused and reasonably small (ideally < 300 lines)
* Write clear commit messages
* Add tests when appropriate
* Update documentation if APIs change

---

## Pull Request Guidelines

A good PR should:

* Clearly describe what changed and why
* Reference related issues if applicable
* Be easy to review (avoid large unrelated changes)

---

## Issues

Use GitHub issues to report bugs or suggest features.

Include:

* clear description and acceptance criteria
* steps to reproduce (for bugs)

---

## Documentation

Build docs locally:
```
pip install -e '.[docs]'
mkdocs serve
```
Build static site — `--strict` is what CI runs, so use it locally too:
```
mkdocs build --strict
```
To regenerate the figures the docs embed, you also need the example
dependencies (matplotlib, seaborn, JupyterLab):
```
pip install -e '.[examples]'
```
Some example scripts (e.g. `hong2025_reproduction.py --mode full`,
`full_wppm_fit_example.py`) are GPU jobs. On an NVIDIA machine, including the
Flatiron cluster, add the `cuda` extra:
```
pip install -e '.[cuda]'
```
JAX bundles its own CUDA/cuDNN runtime through this extra — do **not** also
`module load cuda cudnn` on the cluster, since the system libraries then take
precedence on `LD_LIBRARY_PATH` and break GPU detection; `module load python`
is enough. **Apple Silicon GPUs are not supported**: JAX's Metal backend
cannot do `float64`, which these examples require for their
published-precision comparisons, so Mac users should expect CPU-only
execution (or `--mode quick`-style smoke tests only).

Deploy:
```
mkdocs gh-deploy --clean
```

### Code snippets in docs: the `--8<--` markers

Tutorial pages do not retype code. They quote it out of the runnable script
next to them, so the page and the script cannot drift apart. Mark a region in
the `.py`:

```python
# ;--8<-- [start:fit]
posterior = optimizer.fit(model, data, init_params=init)
# ;--8<-- [end:fit]
```

and pull it into the `.md` by path and tag:

````markdown
```python title="MAP fit"
;--8<-- "docs/examples/wppm/hong2025_reproduction.py:fit"
```
````

This is [`pymdownx.snippets`](https://facelessuser.github.io/pymdown-extensions/extensions/snippets/)
from PyMdown Extensions, configured in `mkdocs.yml`. The `8<` is a pair of
scissors — the old "cut here" convention — not something we invented.

Two things to know:

- We set `check_paths: true`, so a renamed or deleted tag **fails the build**
  rather than silently rendering nothing. `mkdocs build --strict` catches it.
- Prefer a snippet over pasting code into the Markdown. Pasted code has no
  such protection and will go stale.

---

## License

By contributing, you agree that your contributions will be licensed under the project’s [`LICENSE.md`](https://github.com/flatironinstitute/psyphy/blob/main/LICENSE.md) .

---

# Standard Workflow (Step-by-Step)

This section explains the full workflow in detail.

### 1. Fork and clone

On GitHub:

* Fork `flatironinstitute/psyphy` to your account

Then locally:
```
git clone https://github.com/YOUR_USERNAME/psyphy.git
# clones your fork -> sets origin = your GitHub repo
cd psyphy
git remote add upstream https://github.com/flatironinstitute/psyphy.git
# adds upstream -> the canonical Flatiron repo
```
Check:
```
git remote -v
# origin   -> your fork (you push here)
# upstream -> Flatiron repo (you pull from here, never push)
```


#### Summary:
```
- GitHub:

  upstream (canonical repo)
  flatironinstitute/psyphy
            │  Pull Request (PR)
            │
  origin (your fork)
  yourusername/psyphy
            │  git push
            │
- Local machine:

  your local repository
            |
            │  git commit
            │
        your changes
```

---

### 2. Development setup

Install the package and enable pre-commit hooks:
```
pip install -e .
pre-commit install
```
This project uses automated checks (formatting, linting, type checking) via pre-commit  ￼.
These run automatically when you commit.

Alternatively, [uv](https://docs.astral.sh/uv/) installs from the committed [`uv.lock`](https://github.com/flatironinstitute/psyphy/blob/main/uv.lock), giving you the exact versions everyone else resolved rather than whatever the loose ranges in `pyproject.toml` allow on the day you install:
```
module load uv          # Flatiron cluster only; skip if uv is already installed
uv sync --extra dev
uv run pre-commit install
```
Use `uv run <command>` instead of manually activating the venv (e.g. `uv run pytest`), or `source .venv/bin/activate` once `uv sync` has created it. Any time you add or bump a dependency in `pyproject.toml`, run `uv lock` and commit the updated `uv.lock` alongside it.

---

### 3. Sync with upstream before starting work
```
git checkout main
git fetch upstream
# fetch latest changes from upstream (Flatiron repo)
git merge upstream/main
# update your local main with upstream changes
git push origin main
# push updated main -> your fork (origin)
```
---

### 4. Create a feature branch
```
git checkout -b feature/my-feature
# create a new branch from updated main
```
---

### 5. Work and commit
```
git add .
git commit -m "Describe your change clearly"
# pre-commit hooks run automatically here
```
---

### 6. Push to your fork
```
git push origin feature/my-feature
# pushes your branch -> your fork (origin)
```
---

### 7. Open a Pull Request

On GitHub:

* Base repo: flatironinstitute/psyphy (upstream)
* Compare: your branch on your fork (origin)

---

### 8. Iterate on feedback
```
git add .
git commit -m "Address review feedback"
git push origin feature/my-feature
# updates PR automatically
```
---

### 9. After merge
```bash
git checkout main
git fetch upstream
git merge upstream/main
# get latest merged changes from upstream
git push origin main
# sync your fork
git branch -d feature/my-feature
# delete local branch
```
---

###  Keeping Your Branch Up-to-Date

If you’re working on a branch over time, upstream main will change.

⚠️ It’s important to regularly update your branch to avoid large conflicts later.

#### Recommended approach: rebase
```
git fetch upstream
# get latest changes from upstream
git checkout feature/my-feature
git rebase upstream/main
# reapply your work on top of latest upstream main
```
Then:
```
git push --force-with-lease origin feature/my-feature
# update your branch on your fork (origin) safely
```
---

#### When Things Diverge (Merge Conflicts)

If your branch and upstream modify the same code, Git will report a conflict.

This is normal.

What a conflict looks like
```
<<<<<<< HEAD
your code
=======
upstream code
>>>>>>> upstream/main
```
How to resolve:

1. Open the file
2. Decide how to combine the changes (your IDE, like VSCode will visualize each merge conflict making it straight forward whether to keep the upstream version of your local one)
3. Remove the conflict markers

Then:
```
git add filename.py
```
If rebasing:
```
git rebase --continue
```
Repeat until complete.

---

If conflicts are complex

Don’t try to resolve line-by-line blindly. Instead:

* Understand what upstream changed
* Understand your intended behavior
* Update your code accordingly

---

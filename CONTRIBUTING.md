

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
# open PR: origin → upstream (GitHub UI)
```
---


## Development Tools

We use:

* Ruff → formatting + linting (See [LINTING_SETUP.md](LINTING_SETUP.md) for details.)
* mypy → type checking
* pre-commit → runs checks automatically

You can also run manually:
```
pre-commit run --all-files
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

# Documentation

Build docs locally:
```
pip install mkdocs mkdocs-material 'mkdocstrings[python]'
mkdocs serve
```
Build static site:
```
mkdocs build
```
Deploy:
```
mkdocs gh-deploy --clean
```
---

# License

By contributing, you agree that your contributions will be licensed under the project’s [`LICENSE.md`](LICENSE.md) .

---

# Standard Workflow (Step-by-Step)

This section explains the full workflow in detail.

### 1. Fork and clone

On GitHub:

* Fork `flatironinstitute/psyphy` to your account

Then locally:
```
git clone https://github.com/YOUR_USERNAME/psyphy.git
# clones your fork → sets origin = your GitHub repo
cd psyphy
git remote add upstream https://github.com/flatironinstitute/psyphy.git
# adds upstream → the canonical Flatiron repo
```
Check:
```
git remote -v
# origin   → your fork
# upstream → Flatiron repo
```

```
GitHub:

  upstream (canonical repo)
  flatironinstitute/psyphy
            │  Pull Request (PR)
            │
  origin (your fork)
  yourusername/psyphy
            │  git push
            │
Local machine:

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

---

### 3. Sync with upstream before starting work
```
git checkout main
git fetch upstream
# fetch latest changes from upstream (Flatiron repo)
git merge upstream/main
# update your local main with upstream changes
git push origin main
# push updated main → your fork (origin)
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
# pushes your branch → your fork (origin)
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


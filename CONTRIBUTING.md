# Contributing

## Dev setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Lint & test

```bash
ruff check .
black --check .
pytest -q
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
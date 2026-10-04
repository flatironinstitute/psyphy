"""Keep the tutorial's opening code block honest.

``docs/examples/wppm/hong2025_reproduction.md`` opens with a short recipe. That
block used to be typed into the Markdown, and it drifted: it still said
``mc_samples=500`` long after the page had moved to the paper's 2000. It is now
quoted from ``docs/examples/wppm/hong2025_recipe.py`` via a ``pymdownx.snippets``
region, so the page and the code cannot disagree.

That leaves one gap. The snippet is always *shown* faithfully, but nothing
notices if psyphy's own API moves underneath it -- a renamed function or keyword
would leave the page quoting code that no longer runs, and ``mkdocs build`` would
still pass. These tests close that gap.

They are deliberately **static**. The recipe executes at import: its last
statement reaches ``.mean``, which downloads the OSF data and runs the full
49-point inversion at the paper's settings (~11 min). So we parse it instead of
importing it, which also means these tests need no network and no cache.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

RECIPE = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "examples"
    / "wppm"
    / "hong2025_recipe.py"
)

START = "# --8<-- [start:recipe]"
END = "# --8<-- [end:recipe]"


def _snippet() -> str:
    """The exact text the tutorial renders, dedented like snippets does."""
    text = RECIPE.read_text(encoding="utf-8")
    assert START in text and END in text, (
        f"{RECIPE.name} is missing its snippet markers. The tutorial includes "
        f"'{START[8:]}' by name; renaming it breaks the docs build."
    )
    return text.split(START, 1)[1].split(END, 1)[0]


def _calls(tree: ast.AST) -> list[ast.Call]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)]


def test_recipe_file_exists():
    assert RECIPE.is_file(), f"{RECIPE} is quoted by the tutorial but missing."


def test_snippet_region_is_valid_python():
    """A syntax error here would render fine and mislead every reader."""
    ast.parse(_snippet())


@pytest.mark.parametrize(
    ("module_path", "names"),
    [
        (
            "psyphy.posterior",
            ("MAPPosterior", "ThresholdConfig", "WPPMPredictivePosterior"),
        ),
        (
            "psyphy.data.published.hong2025",
            ("fetch", "load_reference_W", "load_sigma_table", "build_paper_model"),
        ),
    ],
)
def test_names_the_recipe_uses_still_exist(module_path, names):
    """Guard against a rename landing in the library but not in the tutorial."""
    import importlib

    module = importlib.import_module(module_path)
    for name in names:
        assert hasattr(module, name), (
            f"{module_path}.{name} is used by the tutorial recipe but no longer "
            "exists. Update docs/examples/wppm/hong2025_recipe.py."
        )


def test_keyword_arguments_are_still_accepted():
    """Every keyword the recipe passes must exist on the thing it calls.

    This is the check that catches a silently renamed parameter -- the failure
    mode a reader would otherwise hit as a TypeError after a 3 GB download.
    """
    from psyphy.data.published import hong2025
    from psyphy.posterior import (
        MAPPosterior,
        ThresholdConfig,
        WPPMPredictivePosterior,
    )

    targets: dict[str, object] = {
        "MAPPosterior": MAPPosterior,
        "ThresholdConfig": ThresholdConfig,
        "WPPMPredictivePosterior": WPPMPredictivePosterior,
    }

    checked = 0
    for call in _calls(ast.parse(_snippet())):
        func = call.func
        if isinstance(func, ast.Name):
            target = targets.get(func.id)
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id != "hong2025":  # jax/jnp calls are not our API
                continue
            target = getattr(hong2025, func.attr, None)
            assert target is not None, f"hong2025.{func.attr} no longer exists."
        else:
            continue

        if target is None:
            continue

        accepted = set(inspect.signature(target).parameters)
        for kw in call.keywords:
            if kw.arg is None:  # **kwargs splat
                continue
            assert kw.arg in accepted, (
                f"the recipe passes {kw.arg}= to "
                f"{getattr(target, '__name__', target)}, which no longer accepts it."
            )
            checked += 1

    assert checked >= 6, (
        f"only {checked} keyword arguments were checked; the AST walk is probably "
        "not finding the recipe's calls any more."
    )

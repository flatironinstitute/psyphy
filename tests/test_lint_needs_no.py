"""check_language.py asks about the flagged phrase and leaves plain wording alone."""

import pathlib
import subprocess
import sys

SCRIPT = pathlib.Path(__file__).parents[1] / "linting" / "check_language.py"
NEEDS = "needs" + " no"  # built here so the linter doesn't flag this file
NEED = "need" + " no"


def run_lint(tmp_path, text):
    f = tmp_path / "page.md"
    f.write_text(text)
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(f)], capture_output=True, text=True
    )


def test_flagged_phrase_is_reported(tmp_path):
    out = run_lint(tmp_path, f"This run {NEEDS} GPU.\nThese tests {NEED} network.\n")
    assert f"page.md:1: '{NEEDS}'" in out.stdout
    assert f"page.md:2: '{NEED}'" in out.stdout


def test_plain_wording_is_not_reported(tmp_path):
    out = run_lint(tmp_path, "This run doesn't need a GPU.\n")
    assert out.stdout == ""
    assert out.returncode == 0


def test_skip_lint_exempts_a_line(tmp_path):
    out = run_lint(tmp_path, f"It {NEEDS} key. <!-- skip-lint -->\n")
    assert out.stdout == ""

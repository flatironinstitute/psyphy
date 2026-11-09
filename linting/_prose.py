"""Shared helpers for the prose linters: collect files, pull prose out of them."""

import io
import pathlib
import tokenize

SUFFIXES = (".py", ".md")


def collect_paths(args):
    paths = []
    for p in args:
        p = pathlib.Path(p)
        if p.is_dir():
            paths.extend(q for q in p.glob("**/*") if q.suffix in SUFFIXES)
        elif p.suffix in SUFFIXES:
            paths.append(p)
    return paths


def prose_lines(path):
    """Yield (line_number, text) for every line of prose in a file.

    A markdown file is all prose. In a Python file only comments and string
    literals are, so identifiers like ``clean_data`` never match a word rule.
    """
    txt = path.read_text()
    if path.suffix == ".md":
        yield from enumerate(txt.splitlines(), start=1)
        return
    tokens = tokenize.generate_tokens(io.StringIO(txt).readline)
    for tok in tokens:
        if tok.type not in (tokenize.COMMENT, tokenize.STRING):
            continue
        for i, line in enumerate(tok.string.splitlines()):
            yield tok.start[0] + i, line


def skipped(line):
    # `<!-- skip-lint -->` in markdown, `# skip-lint` in a comment
    return "skip-lint" in line

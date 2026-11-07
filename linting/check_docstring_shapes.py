#!/usr/bin/env python3
"""Flag array parameters and returns in a numpy docstring without a shape.

Wants `covs : jnp.ndarray, shape (n_points, 2, 2)` on the type line.
"""

import ast
import pathlib
import re
import sys

ARRAY = r"\b(jnp\.ndarray|np\.ndarray|jax\.Array|ArrayLike)\b"
# `name : type` lines, or a bare type line under Returns
TYPE_LINE = re.compile(rf"^\s*(\w+(, \w+)* : .*{ARRAY}|{ARRAY}).*$")

paths = []
for p in sys.argv[1:]:
    p = pathlib.Path(p)
    if p.is_dir():
        paths.extend(p.glob("**/*.py"))
    elif p.suffix == ".py":
        paths.append(p)

missing = []
for p in paths:
    tree = ast.parse(p.read_text())
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
        ):
            continue
        doc = ast.get_docstring(node, clean=False)
        if doc is None:
            continue
        name = getattr(node, "name", p.stem)
        for line in doc.splitlines():
            # a PRNG key has no shape worth stating
            if line.strip().startswith(("key ", "rng_key ", "key, ")):
                continue
            if TYPE_LINE.match(line) and "shape" not in line and "pytree" not in line:
                missing.append((p, name, line.strip()))

if missing:
    print("The following docstrings give an array type with no shape:")
    for p, name, line in missing:
        print(f"{p}:{name}\n\t{line}")
    sys.exit(1)

"""Import-direction checks between subpackages."""

import ast
from pathlib import Path

import psyphy

SRC = Path(psyphy.__file__).parent
CORE_PACKAGES = ["inference", "posterior", "trial_placement", "viz"]
FIELD_MODULE = "psyphy.model.covariance_field"


def _imported_modules(path, package):
    """Absolute dotted names of everything one file imports."""
    tree = ast.parse(path.read_text())
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # relative import: climb `level` packages from the file's own package
                base = package.split(".")[: len(package.split(".")) - node.level + 1]
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module
            found.append(module)
            # `from psyphy.model import covariance_field` names the module too
            found += [f"{module}.{a.name}" for a in node.names]
    return found


def test_core_does_not_import_covariance_field():
    offenders = []
    for pkg in CORE_PACKAGES:
        for path in (SRC / pkg).rglob("*.py"):
            rel = path.relative_to(SRC.parent).with_suffix("")
            package = ".".join(rel.parts[:-1])
            for name in _imported_modules(path, package):
                if name == FIELD_MODULE or name.startswith(FIELD_MODULE + "."):
                    offenders.append(f"{path.relative_to(SRC.parent)} imports {name}")
    assert not offenders, offenders

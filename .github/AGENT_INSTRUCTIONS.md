# AGENT_INSTRUCTIONS.md

## 1) Mission & Constraints

**Mission:** Ship focused, high-quality changes with **test-driven development (TDD)**, strong typing, and clear documentation.

**Core principles**

* Prefer **TDD**: add/adjust a failing test → implement minimally → make green → refactor confidently.
* Favor **small, reversible PRs** (one clear concern).
* Introduce dependencies when a **failing test** or design note demonstrates clear value.
* Keep secrets safe by using environment variables and sanitized logs.
* Write **extensive inline comments** and **NumPy-style docstrings** to enable natural **type hints** and IDE help.

**Project context (psyphy)**

* Stack: **Python ≥3.9**, **JAX 0.4.28**, **Optax 0.2.4**, **NumPy ≥1.22**.
* Emphasize **explicit posterior objects** separate from model definitions; prefer functional, pure transforms.

---

## 2) Autonomy & Guardrails

You **may**:

* Create/modify tests, source, and docs in `tests/`, `src/`, and `docs/`.
* Extract small utilities (≈ ≤ 50 LOC) when they reduce duplication—include tests.
* Propose refactors with measurable benefits (clarity, complexity, performance) and a short note.

Open a PR with rationale when:

* Changing public APIs, serialization formats, or on-disk layout.
* Introducing heavier dependencies or CI/build changes.
* Making algorithmic or performance trade-offs that affect users.

---

## 3) Definition of Done (per PR)

* [ ] **Tests**: failing → passing, including edge cases and failure modes.
* [ ] **Coverage**: ≥ 90% for touched lines; validate shapes/dtypes where relevant.
* [ ] **Docs**: comprehensive **NumPy-style docstrings** + a concise changelog entry.
* [ ] **Typing/Style**: ruff, ruff-format, black-compatible formatting, and mypy/pyright pass.
* [ ] **Perf guard** (when relevant): micro-benchmark or shape/allocation assertions.
* [ ] **CI**: all checks green.

---

## 4) TDD Workflow

1. **Red**: encode desired behavior in a failing test (`tests/`).
2. **Green**: implement the minimal change in `src/`.
3. **Refactor**: improve structure while keeping tests green.
4. Iterate from happy path → edge cases → error paths.

**Testing practices**

* Use **pytest** parametrization for shape/dtype grids.
* For JAX: assert **pytree structure**, **shapes/dtypes**, and **JIT-ability** (no side-effects).
* For numerics: select `rtol/atol` proportional to magnitude; add a brief comment for unusual tolerances.
* For randomness: seed and **split PRNG** deterministically.

---

## 5) Documentation & Typing Standards

Aim for code that **reads like a paper with experiments**:

* **Docstrings**: NumPy style for public APIs and most internal helpers; include purpose, parameters (with types), returns, shapes/dtypes, exceptions, notes, and a minimal example.
* **Comments**: explain non-obvious math, invariants, stability tricks, and JIT considerations right where they matter.
* **Type hints**: annotate all public signatures and key internal helpers; use precise containers (`Mapping`, `Sequence`, `Tuple[...]`) and `jax.Array`/`jnp.ndarray` where appropriate.

**NumPy-style template**

```python
import jax.numpy as jnp

def softplus_stable(x: jnp.ndarray, limit: float = 20.0) -> jnp.ndarray:
    """
    Compute a numerically stable softplus.

    Parameters
    ----------
    x : jnp.ndarray
        Input array, arbitrary shape.
    limit : float, optional
        Threshold where the approximation `softplus(x) ≈ x` becomes active, by default 20.0.

    Returns
    -------
    jnp.ndarray
        Softplus of `x`, same shape/dtype.

    Notes
    -----
    Uses `log1p(exp(x))` for moderate values and switches to `x` for large positives
    to avoid overflow.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> softplus_stable(jnp.array([0.0, 20.0]))
    Array([ 0.6931472, 20.       ], dtype=float32)
    """
    # Large-x branch avoids overflow in exp
    return jnp.where(x > limit, x, jnp.log1p(jnp.exp(x)))
```

---

## 6) Repository Conventions

```
.
├── src/psyphy/           # public API lives here
│   ├── __init__.py       # re-export only stable symbols (ruff F401 allowed here)
│   ├── models/           # pure model defs
│   ├── posterior/        # posterior objects & inference
│   ├── utils/            # pure helpers (no I/O)
│   └── io/               # load/save; versioned formats
├── tests/
│   ├── unit/
│   └── integration/
├── benchmarks/
├── docs/
├── pyproject.toml
└── .pre-commit-config.yaml
```

**Conventions**

* Create `tests/unit/test_<module>.py` for each new module.
* Keep functions focused (≈ ≤ 50–80 LOC); extract helpers when clarity improves.

---

## 7) Tooling & CI Alignment (from repo config)

**Pre-commit (run locally and in CI)**

* Ruff linter (`ruff --fix`) and formatter (`ruff-format`).
* Mypy (with `additional_dependencies=[numpy]`, `--ignore-missing-imports`, `--no-site-packages`) targeting `src/psyphy` and `tests/`.
* General hooks: trailing whitespace, EOF fixers, YAML/TOML checks, large-file guard (≤ 1 MB), merge-conflict, and debug-statement checks.

**Ruff (style & lint)**

* Target: **py39**, line length 88, double quotes, import sorting, NumPy rules (`NPY`), pyupgrade, bugbear, simplify, comprehensions.
* Ignores: `E501` (formatter governs width), `ARG` (prototype allowance).
* Per-file ignores:

  * `tests/**/*.py`: `ARG`, `S101` (asserts encouraged in tests).
  * `__init__.py`: `F401` (re-exports).
  * `docs/**/*.py`: `ARG`, `T201` (prints permitted in examples).

**Mypy (typing)**

* Target: **Python 3.9**; `warn_return_any=false` (JAX interop).
* `ignore_missing_imports=true`; files limited to `src/psyphy`, `tests`.
* Exclusions: `build/`, `dist/`, `.venv/`.
* Overrides: silence errors for `jax.*`, `jaxlib.*`, `optax.*` (annotate interfaces at our boundary to retain type value).

**Pytest**

* Discovers tests in `tests/`; patterns: `test_*.py`, `Test*`, `test_*`.

**Build**

* Hatchling; wheels from `src/psyphy`; sdists include `src/psyphy`, `README.md`, `requirements.txt`, `tests`.

**Local loop**

* Prefer running: `pre-commit run -a && pytest -q`.
* VS Code tasks (optional):

  * `tests`: `pytest -q`
  * `lint`: `ruff check . && ruff format --check . && mypy src/psyphy tests`
  * `fix`: `ruff check . --fix && ruff format .`

---

## 8) Coding Standards

* **Python**: 3.9+; adopt `@dataclass(frozen=True)` for immutable configs where helpful.
* **JAX**: write **pure**, JIT-friendly functions; separate `init`/`apply`; prefer `vmap`, `jit`, `lax` over Python loops in hot paths; minimize host↔device transfers.
* **Errors**: raise specific exceptions with actionable messages.
* **Logging**: keep output concise and opt-in; prefer structured messages.
* **API surface**: re-export only stable symbols in `src/psyphy/__init__.py`.

---

## 9) Psyphy-Preferred Patterns

* **Explicit posterior API**

  * `prior = Model.init(config)`
  * `post = Posterior.from_data(prior, X, y)`
  * `pred = post.predict(X_star)`
* **Functional state**: return new state; avoid hidden mutation.
* **Shape-first**: tests and docstrings state expected shapes/dtypes and broadcasting.

---

## 10) Refactoring Policy

Refactor when it:

* Reduces duplication or cyclomatic complexity.
* Clarifies interfaces or invariants.
* Improves JIT-compatibility or memory footprint (add a brief note or micro-bench).

Prefer separate PRs for refactors unless the change is trivial and tightly coupled to the fix/feature.

---

## 11) Performance & Memory

* Add micro-benchmarks for hot paths; record device (CPU/GPU/TPU).
* Keep arrays on device; batch with `vmap`; fuse ops within `jit`.
* Add shape/allocation assertions in tests where regression risk is high.

---

## 12) Documentation Rules

* Public functions/classes include **NumPy-style docstrings** and, when concise, **doctest-ready examples**.
* Tricky math and stability choices include **inline derivations or citations** in comments or docstrings.
* Update `docs/CHANGELOG.md` for user-visible changes and add migration notes for behavior/API shifts.

---

## 13) Git Hygiene

* Branch names: `feat/<slug>`, `fix/<slug>`, `refactor/<slug>`, `test/<slug>`.
* Commits: imperative mood, scoped; reference issues where relevant.
* PR size: aim for focused diffs (≈ ≤ 400 LOC) and clear titles.

---

## 14) Decision Log

Capture non-obvious choices in `docs/DECISIONS.md`:

* **Context → Options → Choice → Rationale → Impact** (1–5 bullets).

---

## 15) Clarifications & Ambiguities

* Choose the **simplest TDD-backed interpretation** when specs are fuzzy.
* List open questions and proposed defaults in the PR description.
* Mark acknowledged gaps with `xfail` tests when this improves clarity.

---

### Quick Start Checklist

* [ ] Add a **failing test** that captures intent.
* [ ] Implement the **minimal** change to pass.
* [ ] **Refactor** safely; keep tests green.
* [ ] Run **pre-commit** and **pytest** locally.
* [ ] Write **NumPy-style docstrings** and inline comments.
* [ ] Update **CHANGELOG** and open a PR.

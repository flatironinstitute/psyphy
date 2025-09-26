# psyphy

Psychophysical modeling and adaptive trial placement.

## Install

```bash
# in your project root
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## What is in psyphy?

- Models: WPPM, priors, noise, task likelihoods
- Inference: MAP, Langevin, Laplace
- Posterior: wrappers and diagnostics
- Trial placement: Placement strategies (acquisition functions) (e.g., grid, staircase, info gain)
- Session: end-to-end orchestration of an experiment
# psyphy

Psychophysical modeling and adaptive trial placement. This package provides:

- Wishart Process Psychophysical Model (WPPM)
    - fit to subject's data 
    - predict psychphysical thresholds
    - optional trial placement strategy leveraging model's posterior (e.g., information gain, place next batch of trials such that model's uncertainty is maximally reduced)
- Priors and noise models
    - supports cold and warm starts where warm means initialzing with parameters from previous subjects fitted parameters
    - Noise Model: 
        - default: Gaussian
        - supports Student's T 
- Task likelihoods 
    - currently supports OddityTask, TwoAFC
- Inference engines (MAP, Langevin, Laplace)
- Posterior wrappers and diagnostics
- Trial placement strategies (grid, staircase, information gain)
    - supports online and batchwise trial placement
- Experiment session orchestration
    - reading session data and exporting next batch of trial placments



## Install (editable)

```bash
git clone https://github.com/hmd101/psyphy.git
cd psyphy
pip install -e .

```

## Quickstart (MVP)

```python
import jax
import jax.numpy as jnp
import optax

from psyphy.data.dataset import ResponseData
from psyphy.model import WPPM, Prior, OddityTask, GaussianNoise
from psyphy.inference.map_optimizer import MAPOptimizer
from psyphy.trial_placement.grid import GridPlacement
from psyphy.session.experiment_session import ExperimentSession

# --- Setup model ---
prior = Prior.default(input_dim=2)
task = OddityTask()
noise = GaussianNoise()
model = WPPM(input_dim=2, prior=prior, task=task, noise=noise)

# --- Inference engine ---
inference = MAPOptimizer(steps=200)

# --- Placement strategy ---
placement = GridPlacement(grid_points=[(0,0)])  # MVP stub

# --- Session orchestrator ---
sess = ExperimentSession(model, inference, placement)

# Initialize posterior (before any data)
posterior = sess.initialize()

# Collect data (simulated here)
batch = sess.next_batch(batch_size=5)
# subject_responses = run_trials(batch)   # user-defined
# sess.data.add_batch(batch, subject_responses)

# Update posterior with data
posterior = sess.update()

# Predict thresholds
ellipse = posterior.predict_thresholds(reference=jnp.array([0.0, 0.0]))


```


## Background

This package implements methods described in:
-  [Hong et al. (2025). *Comprehensive characterization of human color discrimination thresholds*.](https://www.biorxiv.org/content/10.1101/2025.07.16.665219v1)

While the paper above  used AEPsych (a Gaussian Process–based trial placer),
`psyphy` integrates trial placement directly with the WPPM posterior (e.g. via InfoGain/EAVC),
making the  adaptive trial placement model-aware.

## Docs

Build and preview the documentation locally:

```bash
# from repo root
source .venv/bin/activate
pip install mkdocs mkdocs-material 'mkdocstrings[python]'
mkdocs serve
```

Build the static site:

```bash
mkdocs build
```

Deploy to GitHub Pages (manual):

```bash
mkdocs gh-deploy --clean
```

For contributors, see CONTRIBUTING.md for full doc guidelines and NumPy-style docstrings.

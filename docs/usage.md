# Usage

## Quick start


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
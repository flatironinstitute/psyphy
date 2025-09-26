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

# -----------------------
# 1) Define the model
# -----------------------
prior_dist = Prior.default(input_dim=2)
decision_task = OddityTask(slope=1.5)
observer_noise = GaussianNoise(sigma=1.0)

wppm_model = WPPM(
	input_dim=2,
	prior=prior_dist,
	task=decision_task,
	noise=observer_noise,
)

# -----------------------
# 2) Choose optimizer and MAP settings
#    (explicitly configure Optax SGD + momentum)
# -----------------------
learning_rate = 5e-4
momentum = 0.9
sgd_momentum = optax.sgd(learning_rate=learning_rate, momentum=momentum)

map_estimator = MAPOptimizer(
	steps=500,
	optimizer=sgd_momentum,
)

# -----------------------
# 3) Trial placement strategy (stubbed)
# -----------------------
trial_placement = GridPlacement(grid_points=[(0.0, 0.0)])  # MVP placeholder

# -----------------------
# 4) Orchestrate an experiment session
# -----------------------
session = ExperimentSession(wppm_model, map_estimator, trial_placement)

# Initialize posterior (before any data)
posterior = session.initialize()

# Propose a batch of trials and collect responses (simulated/user-provided)
proposed_batch = session.next_batch(batch_size=5)
# responses = run_subject(proposed_batch) 
# session.data.add_batch(responses, proposed_batch)

# Update posterior after adding data
posterior = session.update()

# Predict threshold contour around a reference (placeholder unit circle in MVP)
reference_point = jnp.array([0.0, 0.0])
threshold_contour = posterior.predict_thresholds(
	reference=reference_point,
	criterion=0.667,
	directions=16,
)
```

### Alternative: Offline fit without the session
If you already have data and just want to fit and predict without the experiment orchestrator:

```python
from psyphy.data.dataset import ResponseData
from psyphy.model import WPPM, Prior, TwoAFC
from psyphy.inference.map_optimizer import MAPOptimizer
import optax
import jax.numpy as jnp

# Prepare data
# Create an empty container for trials (reference, probe, response)
data = ResponseData()

# Add one trial:
# - ref: reference stimulus (shape: (input_dim,))
# - probe: probe stimulus (same shape as ref)
# - resp: binary response in {0, 1}; TwoAFC log-likelihood treats 1 as "correct"
data.add_trial(ref=jnp.array([0.0, 0.0]), probe=jnp.array([0.1, 0.0]), resp=1)

# Add another trial (subject responded 0 = "incorrect")
data.add_trial(ref=jnp.array([0.0, 0.0]), probe=jnp.array([0.0, 0.1]), resp=0)

# Model
model = WPPM(input_dim=2, prior=Prior.default(2), task=TwoAFC())

# Optimizer config (SGD + momentum)
opt = optax.sgd(learning_rate=5e-4, momentum=0.9)
posterior = MAPOptimizer(steps=500, optimizer=opt).fit(model, data)

# Predictions
p = posterior.predict_prob((jnp.array([0.0, 0.0]), jnp.array([0.05, 0.05])))
contour = posterior.predict_thresholds(reference=jnp.array([0.0, 0.0]))
```
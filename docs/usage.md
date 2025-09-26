# Usage

## Quick start

```python
from psyphy import WPPM, Prior, OddityTask, MAPOptimizer

prior = Prior.default()
model = WPPM(prior=prior)
lik = OddityTask()
opt = MAPOptimizer(model, lik)
res = opt.fit(data=None)
print(res)
```

## CLI / Notebooks
- Use JupyterLab for interactive exploration: `jupyter lab`
- See examples you create under `examples/` as you build them out.

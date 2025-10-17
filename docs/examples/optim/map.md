# Build your own opimizer with jax

This example caters to the users who are interested in building their own optimizers with `jax`.

For this purpose, we expose how the MAP optimizer is implemented in `psyphy`.

You can run the toy example with this from scratch implementation yourself with the following script:
```bash
python docs/examples/mvp/offline_fit_mvp_with_map_optimizer.py
```

---

## Model
```python title="Model"
--8<-- "docs/examples/mvp/offline_fit_mvp.py:model"
```

---

## Training / Fitting
Here, we define the training loop that minimizes the model’s negative log posterior using stochastic gradient descent and momentum both with pshyphy and from scratch.


```python title="Fitting with psyphy"
--8<-- "docs/examples/mvp/offline_fit_mvp_with_map_optimizer.py:training"
```

### Implementing optimizers with `jax` --  exposing psyphy's MAP implementation in Jax
Below we illustrate the implementation of MAP, one of our optimizers implemented
in the `inference` module.
In particular, we will point out why and how we use Jax and [optax](https://optax.readthedocs.io/en/latest/).

**A note on JAX:**
The key feature here is JAX’s Just-In-Time (JIT) compilation, which transforms our Python function into a single, optimized computation graph that runs efficiently on CPU, GPU, or TPU.
To make this work, we represent parameters and optimizer states as PyTrees (nested dictionaries or tuples of arrays) — a core JAX data structure that supports efficient vectorization and differentiation.
This approach lets us scale optimization and inference routines from small CPU experiments to large GPU-accelerated Bayesian models with minimal code changes.

```python title="From scratch: training loop exposing psyphy's MAP implementation in Jax"
--8<-- "docs/examples/mvp/offline_fit_mvp.py:training"
```


<!--
Show code above but don't execute and include generated plots.

- How to format code blocks: https://squidfunk.github.io/mkdocs-material/reference/code-blocks/#usage
- Options for including code from a separate file: https://facelessuser.github.io/pymdown-extensions/extensions/snippets/#snippets-notation
- Options for executing code in blocks: https://pawamoy.github.io/markdown-exec/usage/#render-the-source-code-as-well
- Options for displaying plots: https://pawamoy.github.io/markdown-exec/gallery/#with-matplotlib
- Options for sharing variables between code blocks etc.: https://pawamoy.github.io/markdown-exec/usage
-->

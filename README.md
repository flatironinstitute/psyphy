
<div align="center">
    <picture>
      <source srcset="docs/images/psyphy_logo.png" media="(prefers-color-scheme: light)"/>
      <source srcset="docs/images/psyphy_logo.png"  media="(prefers-color-scheme: dark)"/>
      <img alt="psyphy logo" src="docs/images/psyphy_logo.png" width="200">
    </picture>
    <h3>Active-learning-driven adaptive experimentation in psychophysics</h3>
</div>


<h4 align="center">
  <a href="https://flatironinstitute.github.io/psyphy/#install/">Installation</a> |
  <a href="https://flatironinstitute.github.io/psyphy/reference/">Documentation</a> |
  <a href="https://flatironinstitute.github.io/psyphy/examples/">Examples</a> |
  <a href="https://flatironinstitute.github.io/psyphy/CONTRIBUTING/">Contributing</a>
</h4>

## Overview

`psyphy` is an open-source, JAX-based package for psychophysics research. It fits Bayesian
models of perceptual discrimination, currently the Wishart Process Psychophysical Model
(WPPM), on CPU or GPU. A fit gives either a single maximum a posteriori (MAP) estimate or
samples from the full posterior with the No-U-Turn Sampler (NUTS). The aim is to use that
posterior to choose the next trial in an adaptive experiment.

The examples so far are on human color perception, but the model applies to any stimulus
space where discrimination noise varies smoothly.

The package is under active development, and we welcome contributions.

## Install
`psyphy` supports Python 3.10 and later. We recommend installing `psyphy` under a virtual environment. Once you've created a virtual environment for `psyphy` and activated it, you can install `psyphy` using pip:

```bash
pip install psyphy
```

If you're a developer or want the latest features, install from GitHub:
```bash
git clone https://github.com/flatironinstitute/psyphy.git
cd psyphy
pip install -e .
```
Contributors can also use [uv](https://docs.astral.sh/uv/), which installs from the committed `uv.lock` for a reproducible environment instead of resolving the loose ranges above fresh each time; see [CONTRIBUTING.md](CONTRIBUTING.md) for the uv-based workflow.

### GPU support
The commands above install a **CPU-only** `jax`. For an NVIDIA/CUDA GPU (including the Flatiron cluster), install the `cuda` extra instead:
```bash
pip install "psyphy[cuda]"     # from PyPI
pip install -e ".[cuda]"       # from a clone
```
This pulls in JAX's own `cuda12` extra, which bundles a matching CUDA/cuDNN runtime as pip packages, so no system CUDA module needs to be loaded alongside it.

**Apple Silicon (M-series) GPUs are not supported.** JAX's Metal backend cannot do `float64`, which some `psyphy` examples require for published-precision comparisons (Apple/Google's own `jax-metal` package is also no longer maintained). On a Mac, `psyphy` runs on CPU.

## [Quickstart](https://flatironinstitute.github.io/psyphy/examples/wppm/quick_start/)
- Go [here](https://flatironinstitute.github.io/psyphy/examples/wppm/quick_start/) for a light-weight tutorial that demonstrates how to instantiate, evaluate and fit a model quickly. You should be able to run the underlying [script](https://github.com/flatironinstitute/psyphy/blob/main/docs/examples/wppm/quick_start.py) on your CPU.
- Go [here](https://flatironinstitute.github.io/psyphy/examples/wppm/full_wppm_fit_example/) for a more comprehensive example visualizing a spatially varying covariance field, also explaining the underlying math. The underlying [script](https://github.com/flatironinstitute/psyphy/blob/main/docs/examples/wppm/full_wppm_fit_example.py) for this tutorial requires a GPU.


## Contributing

For contributors, see  [`CONTRIBUTING.md`](CONTRIBUTING.md) for full doc guidelines and NumPy-style docstrings.

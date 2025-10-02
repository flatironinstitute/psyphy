![psyphy logo](images/psyphy_logo_draft.png)

<div align="center">
    <picture>
    <source srcset="docs/images/psyphy_logo_draft.png" media="(prefers-color-scheme: light)"/>
    <source srcset="docs/images/psyphy_logo_draft.png"  media="(prefers-color-scheme: dark)"/>
    <!-- <img align="center" src="docs/assets/logo/logo_text_black.svg" alt="Inferno" width="400" style="padding-right: 10px; padding left: 10px;"/> -->
    </picture>
    <h3>Psychophysical Modeling and Adaptive Trial Placement</h3>
</div>



<h4 align="center">
  <a href="https://hmd101.github.io/psyphy/reference/">Documentation</a> | 
  <a href="https://hmd101.github.io/psyphy/examples/mvp/offline_fit_mvp/">Examples</a> | 
  <a href="https://hmd101.github.io/psyphy/CONTRIBUTING/">Contributing</a>
</h4>

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
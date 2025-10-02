# MVP Offline Fit

This example fits the MVP WPPM model to synthetic 2D data and saves two plots:

- Thresholds around the reference (ground-truth, init, fitted)
- Learning curve (negative log posterior vs steps)

Generated plots (from running `docs/examples/mvp/offline_fit_mvp.py`):

![Thresholds](plots/offline_fit_mvp_lr2e-2_steps1000_thresholds.png)

![Learning curve](plots/offline_fit_mvp_lr2e-2_steps1000_learning_curve.png)

Run the underlying script:

```bash
python docs/examples/mvp/offline_fit_mvp.py
```

The script writes plots into `docs/examples/mvp/plots/` and this page embeds them.

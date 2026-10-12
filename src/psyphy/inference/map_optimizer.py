"""
map_optimizer.py
----------------

MAP (Maximum A Posteriori) optimizer using Optax.

MVP implementation:
- Uses gradient ascent on log posterior.
- Defaults to SGD with momentum, but any Optax optimizer can be passed in.

Connections
-----------
- Calls WPPM.log_posterior_from_data(params, data) as the objective.
- Returns a Posterior object wrapping the MAP estimate.
"""

from __future__ import annotations

import contextlib
import warnings
from typing import Literal

import jax
import optax

from psyphy.inference.base import InferenceEngine
from psyphy.posterior.posterior import MAPPosterior


def _global_norm(tree) -> jax.Array:
    """L2 norm over all leaves of a gradient PyTree."""
    leaves = jax.tree_util.tree_leaves(tree)
    return jax.numpy.sqrt(sum(jax.numpy.sum(jax.numpy.square(x)) for x in leaves))


class MAPOptimizer(InferenceEngine):
    """
    MAP (Maximum A Posteriori) optimizer.

    Parameters
    ----------
    steps : int, default=500
        Number of optimization steps.
    optimizer : optax.GradientTransformation, optional
        Optax optimizer to use. Default: SGD with momentum.

    Notes
    -----
    - Loss function = negative log posterior.
    - Gradients computed with jax.grad.
    - By default the objective is scaled to a *per-trial* quantity
      (``reduction="mean"``); see the ``reduction`` parameter.
    """

    def __init__(
        self,
        steps: int = 500,
        learning_rate: float = 5e-5,
        momentum: float = 0.9,
        optimizer: optax.GradientTransformation | None = None,
        *,
        reduction: Literal["mean", "sum"] = "mean",
        track_history: bool = True,
        log_every: int = 1,
        progress_every: int = 10,
        show_progress: bool = False,
        max_grad_norm: float | None = None,
    ):
        """Create a MAP optimizer.

        Parameters
        ----------
        steps : int
            Number of optimization steps.
        optimizer : optax.GradientTransformation | None
            Optax optimizer to use.
        learning_rate : float, optional
            Learning rate for the default optimizer (SGD with momentum).
        momentum : float, optional
            Momentum for the default optimizer (SGD with momentum).
        reduction : {"mean", "sum"}, optional
            How to scale the objective before differentiating.

            - ``"mean"`` (default): divide the negative log posterior by the
              number of trials N, giving a *per-trial* objective.
              (Hong et al. 2025, elife, used this and no gradient clipping)
            - ``"sum"``: use the negative log posterior as-is.

            The two objectives differ by the positive constant N, so they have
            the **same minimizer**, and with no gradient clipping they trace an
            identical path under ``lr_sum = lr_mean / N``. The choice matters
            for two practical reasons:

            1. **Learning-rate portability.** Under ``"sum"``, gradient
               magnitude grows with N, so ``learning_rate`` must be retuned
               whenever the dataset size changes. Under ``"mean"`` the gradient
               is an average of per-trial gradients, so a working learning rate
               transfers across dataset sizes.
            2. **Gradient clipping.** ``max_grad_norm`` is a fixed threshold.
               Under ``"sum"`` the raw gradient norm scales with N, so the clip
               saturates on essentially every step for realistic N — which
               discards gradient *magnitude* and silently turns SGD into
               normalized fixed-step descent.

               To disable it entirely (e.g. to match a reference implementation
               that does no clipping) pass ``max_grad_norm=None``.

            Note that the *model* is unaffected: ``WPPM.log_posterior_from_data``
            still returns the true (summed) log posterior. Scaling is
            step-size conditioning and lives here, in the optimizer, so that
            density-based consumers (e.g. a Laplace approximation taking the
            Hessian at the mode) keep seeing the unnormalized log posterior.

            Because the recorded loss is per-trial under ``"mean"``, learning
            curves are not comparable to those produced with ``"sum"``: the
            values differ by a factor of N.
        track_history : bool, optional
            When True, record loss history during fitting for plotting.
        log_every : int, optional
            Record every N steps (also records the last step).
        progress_every : int, optional
            Update the progress-bar loss display every N steps (and the last step)
            when show_progress=True.
            This is kept separate from log_every so you can record loss at high
            frequency for plotting (e.g. log_every=1) without forcing a device->host
            sync for the progress UI every step.
        show_progress : bool, optional
            When True, display a tqdm progress bar during fitting.
            This is a UI feature: `tqdm` ships in the ``progress`` extra
            (``pip install 'psyphy[progress]'``), not as a core dependency, and
            if it is not installed fitting proceeds without a progress bar.
        max_grad_norm : float | None, optional
            If set, clip gradients by global norm to this value before applying
            optimizer updates: ``g <- g * min(1, max_grad_norm / ||g||)``.

            **Off by default.** Clipping is a guard rail against rare gradient
            spikes, and only behaves as one when the threshold sits *above* the
            typical gradient norm so that it binds on a small fraction of steps.
            A threshold below the typical norm binds on every step, which fixes
            the step length at ``learning_rate * max_grad_norm`` regardless of
            gradient magnitude -- that is normalized gradient descent, a
            different algorithm, and it makes ``learning_rate`` non-transferable
            (only the product of the two matters).

            The typical norm is model- and data-dependent, so there is no safe
            universal default; for the WPPM it is O(10^2)-O(10^3). Choose a
            threshold from the observed distribution rather than a convention
            borrowed from deep-learning recipes, and check
            :attr:`clip_rate` after :meth:`fit` to confirm which regime you are
            in. Note this is orthogonal to ``reduction``: under ``"sum"`` the
            norm additionally scales with the trial count N, but a mis-set
            threshold saturates under ``"mean"`` too.

        Attributes
        ----------
        clip_rate : float | None
            Fraction of steps on which gradient clipping bound, populated after
            :meth:`fit`. ``None`` when ``max_grad_norm`` is ``None``. A value
            near 1.0 means clipping is not acting as a guard rail but as a
            change of optimizer.
        """
        self.steps = steps
        if reduction not in ("mean", "sum"):
            raise ValueError(f'reduction must be "mean" or "sum", got {reduction!r}.')
        self.reduction = reduction
        base_optimizer = optimizer or optax.sgd(
            learning_rate=learning_rate, momentum=momentum
        )
        if max_grad_norm is None:
            self.optimizer = base_optimizer
        else:
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(float(max_grad_norm)),
                base_optimizer,
            )

        self.track_history = track_history
        self.log_every = max(1, int(log_every))
        self.progress_every = max(1, int(progress_every))
        self.show_progress = bool(show_progress)
        self.max_grad_norm = max_grad_norm
        # Exposed after fit() when tracking is enabled
        self.loss_steps: list[int] = []
        self.loss_history: list[float] = []
        # Exposed after fit(); None when clipping is off. See `max_grad_norm`.
        self.clip_rate: float | None = None
        self.n_clipped_steps: int = 0

    def fit(
        self,
        model,
        data,
        init_params: dict | None = None,
        seed: int | None = None,
    ) -> MAPPosterior:
        """
        Fit model parameters with MAP optimization.

        Parameters
        ----------
        model : WPPM
            Model instance.
        data : ResponseData
            Observed trials.
        init_params : dict | None, optional
            Initial parameter PyTree to start optimization from. If provided,
            this takes precedence over the seed.
        seed : int | None, optional
            PRNG seed used to draw initial parameters from the model's prior
            when init_params is not provided, and as the base key for the MC
            likelihood random stream during optimization. If None, defaults to 0.

        Returns
        -------
        MAPPosterior
            Posterior wrapper around MAP params and model.
        """

        rng_seed = 0 if seed is None else int(seed)
        # Master key: split into init key and optimization key stream.
        master_key = jax.random.PRNGKey(rng_seed)
        init_key, opt_key = jax.random.split(master_key)

        # Initialize parameters
        params = init_params if init_params is not None else model.init_params(init_key)
        opt_state = self.optimizer.init(params)

        # Objective scale. Resolved here, outside the jitted step, so it is a
        # Python float baked in at trace time rather than a traced value.
        # `model.log_posterior_from_data` always returns the true (summed) log
        # posterior; "mean" turns it into a per-trial quantity for the gradient
        # step. See the `reduction` docstring for why this lives here and not
        # in the model.
        scale = 1.0
        if self.reduction == "mean":
            n_trials = int(jax.numpy.asarray(data.responses).shape[0])
            if n_trials == 0:
                raise ValueError('reduction="mean" requires at least one trial.')
            scale = 1.0 / n_trials

        # key is now an explicit argument so each JIT-compiled call receives a
        # distinct random key — a fresh MC noise realization per gradient step.
        clip_at = self.max_grad_norm

        @jax.jit
        def step(params, opt_state, key):
            loss, grads = jax.value_and_grad(
                lambda p: -scale * model.log_posterior_from_data(p, data, key=key)
            )(params)
            # Measured before the update, so it is the norm the clip actually saw.
            # Accumulated on device and read once after the loop, to avoid adding
            # a host sync per step.
            clipped = (
                jax.numpy.zeros((), dtype=jax.numpy.int32)
                if clip_at is None
                else (_global_norm(grads) > clip_at).astype(jax.numpy.int32)
            )
            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, clipped

        # clear any previous history
        if self.track_history:
            self.loss_steps.clear()
            self.loss_history.clear()
        self.clip_rate = None
        self.n_clipped_steps = 0
        n_clipped = jax.numpy.zeros((), dtype=jax.numpy.int32)
        steps_run = 0

        # Optional progress bar.
        #
        # Why we *manually* advance the bar:
        # - When JAX runs on GPU, the first `step(...)` call can spend a long time in
        #   compilation, and tqdm may not visibly advance if the underlying iterator
        #   doesn't get a chance to redraw.
        # - By keeping a normal `range(self.steps)` loop and calling `pbar.update(1)`
        #   ourselves, we ensure the bar advances exactly once per iteration.
        #
        # Performance note: *displaying the loss* requires transferring `loss` from
        # device -> host, which can add sync overhead. We therefore only attach a
        # loss postfix every `progress_every` steps.
        pbar = None
        if self.show_progress:
            try:
                from tqdm.auto import tqdm

                pbar = tqdm(total=self.steps, desc="MAP fit", leave=False)
            except Exception:
                # Soft dependency: tqdm not available (or terminal unsuitable).
                pbar = None

        for i in range(self.steps):
            # Split a fresh subkey for each step so the MC likelihood sees a
            # different noise realization on every gradient evaluation.
            opt_key, subkey = jax.random.split(opt_key)
            params, opt_state, loss, clipped = step(params, opt_state, subkey)
            n_clipped = n_clipped + clipped
            steps_run = i + 1

            # Non-finite guard: if loss becomes NaN/Inf, optimization has diverged.
            # Stop early so downstream plots don’t look “truncated” due to NaNs.
            if not bool(jax.numpy.isfinite(loss)):
                if self.track_history:
                    try:
                        self.loss_steps.append(i)
                        self.loss_history.append(float(loss))
                    except Exception:
                        pass
                print(
                    f"[MAPOptimizer] Non-finite loss at step {i}: {loss}. "
                    "Stopping early."
                )
                if pbar is not None:
                    with contextlib.suppress(Exception):
                        pbar.update(1)
                break

            if self.track_history and (
                (i % self.log_every == 0) or (i == self.steps - 1)
            ):
                # Pull scalar to host and record
                try:
                    self.loss_steps.append(i)
                    self.loss_history.append(float(loss))
                except Exception:
                    #  do not break fitting if logging fails
                    pass

            #  progress bar loss display (avoid host sync every step)
            if pbar is not None and (
                (i % self.progress_every == 0) or (i == self.steps - 1)
            ):
                with contextlib.suppress(Exception):
                    pbar.set_postfix(loss=float(loss))

            if pbar is not None:
                with contextlib.suppress(Exception):
                    pbar.update(1)
                    # Encourage a redraw occasionally in environments with buffered/stale
                    # TTY updates.
                    if (i % self.progress_every == 0) or (i == self.steps - 1):
                        pbar.refresh()

        if pbar is not None:
            with contextlib.suppress(Exception):
                pbar.close()

        # Clipping regime. Read once, after the loop, so the counter costs no
        # per-step host sync. A rate near 1.0 means the threshold sits below the
        # model's typical gradient norm, so every step is rescaled to
        # `max_grad_norm` and the effective algorithm is no longer plain SGD.
        if self.max_grad_norm is not None and steps_run > 0:
            self.n_clipped_steps = int(n_clipped)
            self.clip_rate = self.n_clipped_steps / steps_run
            if self.clip_rate > 0.5:
                warnings.warn(
                    f"Gradient clipping bound on {self.clip_rate:.0%} of steps "
                    f"({self.n_clipped_steps}/{steps_run}) at "
                    f"max_grad_norm={self.max_grad_norm}. At this rate clipping "
                    "is not a guard rail: the step length is fixed at "
                    "learning_rate * max_grad_norm regardless of gradient "
                    "magnitude, so `learning_rate` no longer means what it would "
                    "without clipping. Raise max_grad_norm above the typical "
                    "gradient norm, or pass max_grad_norm=None.",
                    stacklevel=2,
                )

        return MAPPosterior(params=params, model=model)

    # Optional helper
    def get_history(self) -> tuple[list[int], list[float]]:
        """Return (steps, losses) recorded during the last fit when tracking was enabled."""
        return self.loss_steps, self.loss_history

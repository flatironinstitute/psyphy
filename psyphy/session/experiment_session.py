"""
experiment_session.py
---------------------

ExperimentSession orchestrates the adaptive experiment loop.

Responsibilities
----------------
1. Store trial data (ResponseData).
2. Manage inference engine (MAP, Langevin, Laplace).
3. Keep track of the current posterior.
4. Delegate adaptive placement to a TrialPlacement strategy.

MVP implementation:
- Data container starts empty and can be appended to.
- initialize() fits a posterior once before trials.
- update() refits posterior with accumulated data.
- next_batch() proposes new trials from the placement strategy.

Full WPPM mode:
- Will support batch vs online updates.
- Integrate with lab software for live trial execution.
- Save/load checkpoints for long experiments.
"""

from psyphy.data.dataset import ResponseData


class ExperimentSession:
    """
    High-level experiment orchestrator.

    Parameters
    ----------
    model : WPPM
        (Psychophysical) model instance.
    inference : InferenceEngine
        Inference engine (MAP, Langevin, etc.).
    placement : TrialPlacement
        Adaptive trial placement strategy.
    init_placement : TrialPlacement, optional
        Initial placement strategy (e.g., Sobol exploration).

    Attributes
    ----------
    data : ResponseData
        Stores all collected trials.
    posterior : Posterior or None
        Current posterior estimate (None before initialization).
    """

    def __init__(self, model, inference, placement, init_placement=None):
        self.model = model
        self.inference = inference
        self.placement = placement
        self.init_placement = init_placement

        # Data store starts empty
        self.data = ResponseData()

        # Posterior will be set after initialize() or update()
        self.posterior = None

    # ------------------------------------------------------------------
    # FITTING INTERFACE
    # ------------------------------------------------------------------
    def initialize(self):
        """
        Fit an initial posterior before any adaptive placement.

        Returns
        -------
        Posterior
            Posterior object wrapping fitted parameters.

        Notes
        -----
        MVP:
            Posterior is fitted to empty data (prior only).
        Full WPPM mode:
            Could use pilot data or pre-collected trials along grid etc.
        """
        self.posterior = self.inference.fit(self.model, self.data)
        return self.posterior

    def update(self):
        """
        Refit posterior with accumulated data.

        Returns
        -------
        Posterior
            Updated posterior.

        Notes
        -----
        MVP:
            Re-optimizes from scratch using all data.
        Full WPPM mode:
            Could support warm-start or online parameter updates.
        """
        self.posterior = self.inference.fit(self.model, self.data)
        return self.posterior

    # ------------------------------------------------------------------
    # PLACEMENT INTERFACE
    # ------------------------------------------------------------------
    def next_batch(self, batch_size: int):
        """
        Propose the next batch of trials.

        Parameters
        ----------
        batch_size : int
            Number of trials to propose.

        Returns
        -------
        TrialBatch
            Batch of proposed (reference, probe) stimuli.

        Notes
        -----
        MVP:
            Always calls placement.propose() on current posterior.
        Full WPPM mode:
            Could support hybrid placement (init strategy -> adaptive strategy).
        """
        if self.posterior is None:
            raise RuntimeError("Posterior not initialized. Call initialize() first.")
        return self.placement.propose(self.posterior, batch_size)

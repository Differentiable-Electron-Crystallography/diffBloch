"""The preprocess pipeline: fit the experiment's nuisances + numerics, emit the invariant ``Plan``.

Stage 11 builds this as a composable ``Plan -> Plan`` pipeline (a scikit-learn-style sequence of
transformers) producing the geometry the differentiable refinement is conditioned on:
``preprocess -> Plan -> refine``. ``refine`` is the terminal estimator (``Plan -> Result``) and
never re-enters here. The dependency points ``preprocess -> engine``; the engine never imports
preprocess.

This first slice lays the spine: the :class:`~diffBloch.preprocess.plan.Plan` value object and the
:mod:`diffBloch.preprocess.pipeline` combinators (``pipeline`` sequencing + ``iterate_until``
fixpoint). The real steps (``converge_numerics`` / ``fit_orientation`` / ``fit_thickness``) and
``from_experiment`` construction land in later slices.
"""

from diffBloch.preprocess.pipeline import (
    ConvergenceCheck,
    PlanStep,
    identity,
    iterate_until,
    pipeline,
)
from diffBloch.preprocess.plan import Plan

__all__ = [
    "ConvergenceCheck",
    "Plan",
    "PlanStep",
    "identity",
    "iterate_until",
    "pipeline",
]

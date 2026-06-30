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

from diffBloch.preprocess.beams import klar_beam_mask, select_beams
from diffBloch.preprocess.experiment import (
    ExperimentSetup,
    PlanSplit,
    RefinementSetup,
    from_experiment,
)
from diffBloch.preprocess.orientation import (
    busing_levy_matrix,
    goniometer_rotation,
    orientation_basis,
    orientation_matrices,
    u_matrix,
)
from diffBloch.preprocess.pipeline import (
    ConvergenceCheck,
    PlanStep,
    identity,
    iterate_until,
    pipeline,
)
from diffBloch.preprocess.plan import Plan
from diffBloch.preprocess.scoring import build_engine, score_orientations

__all__ = [
    "ConvergenceCheck",
    "ExperimentSetup",
    "Plan",
    "PlanSplit",
    "PlanStep",
    "RefinementSetup",
    "build_engine",
    "busing_levy_matrix",
    "from_experiment",
    "goniometer_rotation",
    "identity",
    "iterate_until",
    "klar_beam_mask",
    "orientation_basis",
    "orientation_matrices",
    "pipeline",
    "score_orientations",
    "select_beams",
    "u_matrix",
]

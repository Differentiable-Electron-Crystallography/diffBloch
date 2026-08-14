"""The preprocess pipeline: fit the experiment's nuisances + numerics, emit the invariant ``Plan``.

This is a composable ``Plan -> Plan`` pipeline (a scikit-learn-style sequence of
transformers) producing the geometry the differentiable refinement is conditioned on:
``preprocess -> Plan -> refine``. ``refine`` is the terminal estimator (``Plan -> Result``) and
never re-enters here. The dependency points ``preprocess -> engine``; the engine never imports
preprocess.

The core pieces are the :class:`~diffBloch.preprocess.plan.Plan` value object, the
:mod:`diffBloch.preprocess.pipeline` combinators (``pipeline`` sequencing + ``iterate_until``
fixpoint), the steps (``converge_numerics`` / ``optimize_orientation`` / ``optimize_thickness`` and the rest),
and ``from_experiment`` construction from typed records.
"""

from diffBloch.preprocess.driver import converge_numerics
from diffBloch.preprocess.experiment import (
    DatasetSetup,
    ExperimentSetup,
    PlanSplit,
    RefinementSetup,
    from_experiment,
    resolve_dataset_mosaicity,
    setup_datasets,
    validation_mask,
)
from diffBloch.preprocess.inference import (
    InferenceResult,
    RotationInference,
    run_inference,
)
from diffBloch.preprocess.orientation import (
    busing_levy_matrix,
    goniometer_rotation,
    hexagonal_tilt,
    orientation_basis,
    orientation_matrices,
    u_matrix,
)
from diffBloch.preprocess.pipeline import (
    OPAQUE,
    ConvergenceCheck,
    Fork,
    PlanStep,
    StatefulPlanStep,
    StateInitializer,
    Step,
    StepRecord,
    as_step,
    fork,
    identity,
    iterate_until,
    pipeline,
    resolve_recipe,
    stateful_pipeline,
    stateful_plan_step,
    step_records,
)
from diffBloch.preprocess.plan import (
    CandidatePlan,
    Plan,
    require_built_plans,
    require_candidate_plans,
    require_orientation_plans,
)
from diffBloch.preprocess.pool import pool
from diffBloch.preprocess.scoring import build_engine, score_orientations
from diffBloch.preprocess.serialize import read_plan, write_plan
from diffBloch.preprocess.steps.beams import (
    build_orientation_plans,
    klar_beam_mask,
    select_beams,
)
from diffBloch.preprocess.steps.convergence import (
    converge_beams,
    converge_sampling,
    converge_scalar,
    simulation_converged,
    simulation_rfactor,
)
from diffBloch.preprocess.steps.coupling import couple_beams
from diffBloch.preprocess.steps.coverage import (
    cover_beams,
    maximize_scalar,
    plan_coverage,
)
from diffBloch.preprocess.steps.frames import select_finite_loss_frames, select_frames
from diffBloch.preprocess.steps.optimize_orientation import optimize_orientation
from diffBloch.preprocess.steps.optimize_thickness import optimize_thickness
from diffBloch.preprocess.steps.report_coupling import report_coupling
from diffBloch.preprocess.steps.rocking_curve import integrate_rocking_curve
from diffBloch.specs import (
    BeamSelection,
    ConvergenceTest,
    ConvergenceTolerance,
    FrameSelection,
    RockingCurve,
    ThicknessGrid,
)

__all__ = [
    "BeamSelection",
    "CandidatePlan",
    "ConvergenceCheck",
    "ConvergenceTest",
    "ConvergenceTolerance",
    "DatasetSetup",
    "ExperimentSetup",
    "Fork",
    "FrameSelection",
    "InferenceResult",
    "OPAQUE",
    "Plan",
    "PlanSplit",
    "PlanStep",
    "RefinementSetup",
    "RockingCurve",
    "StateInitializer",
    "StatefulPlanStep",
    "RotationInference",
    "Step",
    "StepRecord",
    "ThicknessGrid",
    "as_step",
    "build_engine",
    "build_orientation_plans",
    "busing_levy_matrix",
    "converge_beams",
    "converge_numerics",
    "converge_sampling",
    "converge_scalar",
    "cover_beams",
    "optimize_orientation",
    "optimize_thickness",
    "fork",
    "from_experiment",
    "goniometer_rotation",
    "hexagonal_tilt",
    "identity",
    "integrate_rocking_curve",
    "iterate_until",
    "klar_beam_mask",
    "maximize_scalar",
    "couple_beams",
    "orientation_basis",
    "orientation_matrices",
    "pipeline",
    "plan_coverage",
    "pool",
    "report_coupling",
    "read_plan",
    "require_built_plans",
    "require_candidate_plans",
    "require_orientation_plans",
    "resolve_dataset_mosaicity",
    "resolve_recipe",
    "run_inference",
    "score_orientations",
    "select_beams",
    "select_finite_loss_frames",
    "select_frames",
    "setup_datasets",
    "simulation_converged",
    "simulation_rfactor",
    "stateful_pipeline",
    "stateful_plan_step",
    "step_records",
    "u_matrix",
    "validation_mask",
    "write_plan",
]

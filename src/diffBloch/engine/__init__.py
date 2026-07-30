"""The refinement engine: forward composition spine + the imperative refinement loop.

- :mod:`diffBloch.engine.plan` -- refinement-invariant geometry plans (the shared scattering grid,
  per-orientation bundles).
- :mod:`diffBloch.engine.forward` -- the pure, differentiable forward
  (``objective_value``/``simulate``) plus problem execution.
- :mod:`diffBloch.engine.refine` -- the quarantined ``torch.optim`` loop.

The dependency points one way (refine -> forward -> core); ``core/`` stays free of ``torch.optim``.
"""

from diffBloch.engine.components import (
    ApparentThicknessNN,
    PerOrientationThickness,
    QuadraticThicknessProfile,
    ThicknessBounds,
)
from diffBloch.engine.constraints import (
    ConstraintTransform,
    HydrogenRiding,
    perceive_hydrogen_riding,
    with_hydrogen_riding,
)
from diffBloch.engine.forward import (
    ForwardContext,
    LossFn,
    ModelComponent,
    ModelRefinementResult,
    RefinementEngine,
    RefinementModel,
    RefinementProblem,
    StructureComponent,
    build_refinement_model,
    build_refinement_problem,
    run_refinement_model,
)
from diffBloch.engine.losses import (
    l1_loss,
    mse_loss,
    rbragg_loss,
    w_rbragg_loss,
    weighted_mse_loss,
    wr2_loss,
)
from diffBloch.engine.penalties import BondLengthPenalty, perceive_bond_length_penalty
from diffBloch.engine.plan import (
    CoupledOrientationPlan,
    OrientationPlan,
    OrientationPlanLike,
    StructureFactorGrid,
    mean_plan_thickness,
)
from diffBloch.engine.refine import (
    AtomSelection,
    ObjectiveComponent,
    ObjectiveValue,
    OptimizerName,
    PenaltyTerm,
    RefinementResult,
    TrainableSpec,
    run_refinement,
)

__all__ = [
    "ApparentThicknessNN",
    "AtomSelection",
    "BondLengthPenalty",
    "ConstraintTransform",
    "ForwardContext",
    "HydrogenRiding",
    "LossFn",
    "mean_plan_thickness",
    "ModelComponent",
    "ObjectiveComponent",
    "ObjectiveValue",
    "OptimizerName",
    "OrientationPlan",
    "OrientationPlanLike",
    "PerOrientationThickness",
    "QuadraticThicknessProfile",
    "StructureComponent",
    "RefinementEngine",
    "RefinementModel",
    "ModelRefinementResult",
    "RefinementProblem",
    "RefinementResult",
    "PenaltyTerm",
    "StructureFactorGrid",
    "CoupledOrientationPlan",
    "ThicknessBounds",
    "TrainableSpec",
    "build_refinement_model",
    "build_refinement_problem",
    "l1_loss",
    "mse_loss",
    "perceive_bond_length_penalty",
    "perceive_hydrogen_riding",
    "rbragg_loss",
    "run_refinement",
    "run_refinement_model",
    "wr2_loss",
    "w_rbragg_loss",
    "weighted_mse_loss",
    "with_hydrogen_riding",
]

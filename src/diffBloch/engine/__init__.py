"""The refinement engine: forward composition spine + the imperative refinement loop.

- :mod:`diffBloch.engine.plan` -- refinement-invariant geometry plans (the shared scattering grid,
  per-orientation bundles).
- :mod:`diffBloch.engine.forward` -- the pure, differentiable forward (``objective``/``simulate``).
- :mod:`diffBloch.engine.refine` -- the quarantined ``torch.optim`` loop the ``refine`` method runs.

The dependency points one way (refine -> forward -> core); ``core/`` stays free of ``torch.optim``.
"""

from diffBloch.engine.forward import LossFn, RefinementEngine
from diffBloch.engine.losses import (
    l1_loss,
    mse_loss,
    rbragg_loss,
    scaled_w_rbragg_loss,
    w_rbragg_loss,
    weighted_mse_loss,
)
from diffBloch.engine.plan import (
    OrientationPlan,
    OrientationPlanLike,
    ScatteringGrid,
    SegmentedOrientationPlan,
)
from diffBloch.engine.refine import (
    ObjectiveComponent,
    ObjectiveValue,
    OptimizerName,
    RefinementResult,
    run_refinement,
)

__all__ = [
    "LossFn",
    "ObjectiveComponent",
    "ObjectiveValue",
    "OptimizerName",
    "OrientationPlan",
    "OrientationPlanLike",
    "RefinementEngine",
    "RefinementResult",
    "ScatteringGrid",
    "SegmentedOrientationPlan",
    "l1_loss",
    "mse_loss",
    "rbragg_loss",
    "run_refinement",
    "scaled_w_rbragg_loss",
    "w_rbragg_loss",
    "weighted_mse_loss",
]

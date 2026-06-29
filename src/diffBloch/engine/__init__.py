"""The refinement engine: forward composition spine + the imperative refinement loop.

- :mod:`diffBloch.engine.plan` -- refinement-invariant geometry plans (the shared scattering grid,
  per-orientation bundles).
- :mod:`diffBloch.engine.forward` -- the pure, differentiable forward (``objective``/``simulate``).
- :mod:`diffBloch.engine.refine` -- the quarantined ``torch.optim`` loop the ``refine`` method runs.

The dependency points one way (refine -> forward -> core); ``core/`` stays free of ``torch.optim``.
"""

from diffBloch.engine.forward import LossFn, RefinementEngine
from diffBloch.engine.plan import OrientationPlan, ScatteringGrid
from diffBloch.engine.refine import OptimizerName, RefinementResult, run_refinement

__all__ = [
    "LossFn",
    "OptimizerName",
    "OrientationPlan",
    "RefinementEngine",
    "RefinementResult",
    "ScatteringGrid",
    "run_refinement",
]

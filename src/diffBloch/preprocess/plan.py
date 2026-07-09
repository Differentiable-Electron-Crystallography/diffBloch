"""The ``Plan``: the invariant geometry the differentiable refinement is conditioned on.

``Plan`` is the *spine* of the preprocess pipeline -- one immutable value bundling the shared
:class:`~diffBloch.engine.plan.ScatteringGrid` and the per-rotation
:class:`~diffBloch.engine.plan.OrientationPlan`\\ s. Each ``Plan -> Plan`` step returns a sharpened
copy (via :func:`dataclasses.replace`); ``refine`` consumes the final ``Plan``. The dependency
points ``preprocess -> engine`` (this module imports the engine's geometry plans), never the
reverse -- the engine stays unaware of ``Plan`` and remains a pure consumer of grid + orientations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray
from torch import Tensor

from diffBloch.core.products import PatternBatch
from diffBloch.engine.plan import OrientationPlan, OrientationPlanLike, ScatteringGrid

if TYPE_CHECKING:
    from diffBloch.preprocess.pipeline import StepRecord

__all__ = [
    "CandidatePlan",
    "Plan",
    "require_built_plans",
    "require_candidate_plans",
    "require_orientation_plans",
]


@dataclass(frozen=True)
class CandidatePlan:
    """The pre-build *candidate* phase for one rotation: source only, no built geometry.

    ``from_experiment`` lays down a :class:`CandidatePlan` per rotation (the difference-safe
    candidate ``beam_hkl`` + orientation source), ``select_beams`` prunes ``beam_hkl`` (cheap,
    source-only), and :func:`~diffBloch.preprocess.steps.beams.build_orientation_plans` then
    builds it into an :class:`~diffBloch.engine.plan.OrientationPlan` -- the *one* place the
    structure-factor gather is built, over the already-pruned beam set. A ``CandidatePlan`` has no
    ``beam_plans``, so it is unsolvable by construction (the engine consumes only built
    ``OrientationPlan``\\ s); building the expensive gather over the full candidate pool is thereby
    avoided entirely.
    """

    orientation: NDArray[np.float64]
    energy: float
    u0: float
    thickness: NDArray[np.float64]
    beam_hkl: NDArray[np.int64]
    pattern: PatternBatch

    @classmethod
    def seed(
        cls,
        beam_hkl: NDArray[np.int64],
        pattern: PatternBatch,
        *,
        energy: float,
        thickness: Tensor | NDArray[np.float64] | Sequence[float],
        u0: float = 0.0,
        orientation: Tensor | NDArray[np.float64] | None = None,
    ) -> CandidatePlan:
        """Assemble a candidate seed for one orientation (plain-numpy source; builds no gather)."""
        rotation = np.eye(3) if orientation is None else np.asarray(orientation, dtype=np.float64)
        return cls(
            orientation=rotation,
            energy=float(energy),
            u0=float(u0),
            thickness=np.atleast_1d(np.asarray(thickness, dtype=np.float64)),
            beam_hkl=np.asarray(beam_hkl, dtype=np.int64),
            pattern=pattern,
        )


@dataclass(frozen=True)
class Plan:
    """The shared scattering ``grid`` plus the per-rotation ``orientations`` (the refinement spine).

    ``grid`` fixes the ``Fgb`` support and metric; ``orientations`` is one
    :class:`~diffBloch.engine.plan.OrientationPlan` per rotation (each already coupled to ``grid``
    at build time). Immutable: preprocess steps return :func:`dataclasses.replace` copies rather
    than mutating in place.

    ``provenance`` is the ordered tuple of :class:`~diffBloch.preprocess.pipeline.StepRecord`\\ s
    that produced this plan -- :func:`~diffBloch.preprocess.pipeline.pipeline` appends one per step
    as it runs. A freshly built plan (from ``from_experiment``) has empty provenance; the recipe
    identity a checkpoint locks against is this tuple. Steps do not touch it (their ``replace``
    preserves it); the combinator owns it.
    """

    grid: ScatteringGrid
    orientations: tuple[CandidatePlan | OrientationPlanLike, ...]
    provenance: tuple[StepRecord, ...] = field(default_factory=tuple)


def require_orientation_plans(plan: Plan) -> tuple[OrientationPlan, ...]:
    """Narrow a plan's orientations to plain :class:`OrientationPlan`\\ s (reject segmented ones).

    The tilt-independent-only plan-shaping steps (``select_beams``, ``integrate_rocking_curve``,
    ``mosaicity``) transform the :class:`OrientationPlan`, which carries one shared beam set.
    ``couple_beams`` replaces each orientation with a
    :class:`~diffBloch.engine.plan.SegmentedOrientationPlan` (a per-tilt-chunk beam set) that those
    steps cannot consume, so they must all precede ``couple_beams`` in a pipeline. This helper
    enforces that ordering with a clear error and narrows the element type for the caller.

    The fitting steps (``fit_orientation``, ``fit_thickness``) are deliberately *not* narrowed: they
    are plan-agnostic (they rebuild via :meth:`OrientationPlan.with_orientation` /
    ``replace(thickness=...)``, both defined on the segmented plan too), so they iterate
    ``plan.orientations`` directly and run either before or after ``couple_beams``.
    """
    narrowed: list[OrientationPlan] = []
    for op in plan.orientations:
        if not isinstance(op, OrientationPlan):
            raise TypeError(
                "this step transforms tilt-independent OrientationPlans, but this plan holds a "
                "SegmentedOrientationPlan; couple_beams produces those and no other Plan -> Plan "
                "step can consume them, so couple_beams must be the final step in the pipeline"
            )
        narrowed.append(op)
    return tuple(narrowed)


def require_built_plans(plan: Plan) -> tuple[OrientationPlanLike, ...]:
    """Parse the orientation *phase* to built (``OrientationPlan`` / ``SegmentedOrientationPlan``).

    ``Plan.orientations`` is a phase-union (:class:`CandidatePlan` before the build, built plans
    after) rather than a phase-indexed ``Plan[P]``, as the recipe is a runtime ``pipeline`` list
    of homogeneous ``Plan -> Plan`` steps: the phase cannot ride in the type across that list (nor
    across ``read_plan``, which reconstructs a ``Plan`` from ``.npz`` bytes). This is the parse that
    re-establishes the phase at that erased boundary -- *parse, don't validate*: it returns the
    narrowed type once, and the terminals (inference, the fits, ``couple_beams``, checkpoint
    serialize) then hold built geometry without re-checking. See
    ``design/decisions/candidate-vs-built-plan.md`` for why the phase-indexed alternative does not
    survive contact with Python (no phase-changing composition; ``disallow_any_generics``).

    A :class:`CandidatePlan` has no ``beam_plans`` and is unsolvable, so this raises unless
    ``build_orientation_plans`` has run (the default recipe runs it right after ``select_beams``).
    Returns the plan's own ``orientations`` tuple (identity preserved) once narrowed.
    """
    for op in plan.orientations:
        if isinstance(op, CandidatePlan):
            raise TypeError(
                "this operation needs a built plan (OrientationPlan / SegmentedOrientationPlan), "
                "but the plan holds a CandidatePlan; run build_orientation_plans after select_beams"
            )
    return cast("tuple[OrientationPlanLike, ...]", plan.orientations)


def require_candidate_plans(plan: Plan) -> tuple[CandidatePlan, ...]:
    """Parse the orientation *phase* to the pre-build candidate (:class:`CandidatePlan`).

    The candidate-phase counterpart of :func:`require_built_plans` (which documents why the phase is
    parsed at a runtime boundary rather than carried in the type). ``select_beams`` and
    ``build_orientation_plans`` operate on the candidate phase ``from_experiment`` lays down; this
    raises with a clear error if the plan is already built (holds
    :class:`~diffBloch.engine.plan.OrientationPlan`\\ s), since ``build_orientation_plans`` is the
    single step that builds them and runs once, right after ``select_beams``.
    """
    narrowed: list[CandidatePlan] = []
    for op in plan.orientations:
        if not isinstance(op, CandidatePlan):
            raise TypeError(
                "this step operates on the pre-build candidate phase, but this plan is already "
                "built (holds OrientationPlans); build_orientation_plans builds them once, right "
                "after select_beams"
            )
        narrowed.append(op)
    return tuple(narrowed)

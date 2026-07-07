"""The ``Plan``: the invariant geometry the differentiable refinement is conditioned on.

``Plan`` is the *spine* of the preprocess pipeline -- one immutable value bundling the shared
:class:`~diffBloch.engine.plan.ScatteringGrid` and the per-rotation
:class:`~diffBloch.engine.plan.OrientationPlan`\\ s. Each ``Plan -> Plan`` step returns a sharpened
copy (via :func:`dataclasses.replace`); ``refine`` consumes the final ``Plan``. The dependency
points ``preprocess -> engine`` (this module imports the engine's geometry plans), never the
reverse -- the engine stays unaware of ``Plan`` and remains a pure consumer of grid + orientations.
"""

from __future__ import annotations

from dataclasses import dataclass

from diffBloch.engine.plan import OrientationPlan, OrientationPlanLike, ScatteringGrid

__all__ = ["Plan", "require_orientation_plans"]


@dataclass(frozen=True)
class Plan:
    """The shared scattering ``grid`` plus the per-rotation ``orientations`` (the refinement spine).

    ``grid`` fixes the ``Fgb`` support and metric; ``orientations`` is one
    :class:`~diffBloch.engine.plan.OrientationPlan` per rotation (each already coupled to ``grid``
    at build time). Immutable: preprocess steps return :func:`dataclasses.replace` copies rather
    than mutating in place.
    """

    grid: ScatteringGrid
    orientations: tuple[OrientationPlanLike, ...]


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

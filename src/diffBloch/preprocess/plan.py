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

from diffBloch.engine.plan import OrientationPlan, ScatteringGrid

__all__ = ["Plan"]


@dataclass(frozen=True)
class Plan:
    """The shared scattering ``grid`` plus the per-rotation ``orientations`` (the refinement spine).

    ``grid`` fixes the ``Fgb`` support and metric; ``orientations`` is one
    :class:`~diffBloch.engine.plan.OrientationPlan` per rotation (each already coupled to ``grid``
    at build time). Immutable: preprocess steps return :func:`dataclasses.replace` copies rather
    than mutating in place.
    """

    grid: ScatteringGrid
    orientations: tuple[OrientationPlan, ...]

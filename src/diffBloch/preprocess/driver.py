"""The preprocess convergence driver: the ``runState`` runner for the coupled beam-knob sweeps.

The individual convergence levers (``converge_beams`` / ``converge_pool`` / ``converge_sampling``)
and the coverage levers (``cover_beams`` / ``cover_pool``) are pure ``Plan -> Plan`` steps, each
self-contained. But the *coupled* sweep -- pool and window tuned together, and coverage handing its
settled knobs to self-stability (the ``both`` operation) -- needs state the ``Plan`` deliberately
does not carry: the live scalars ``g_max_refine`` / ``integration_semiangle``. This module is the
**driver** that owns that state.

In State-monad terms (see ``design/decisions/plan-composition-shapes.md``):
:class:`ConvergenceState`
is the state ``s``; a *phase* (:func:`run_coverage_phase` here; the self-stability phase lands next)
is a ``(Plan, ConvergenceState) -> (Plan, ConvergenceState)`` computation; the driver plays
``runState``, threading ``s`` between phases and, at its outer boundary, returning just the
converged ``Plan`` (``evalState``). "Driver" is the runner's role name, not a monad.

Obstruction 1 (``stage11-cross-lever-fixpoint.md``): each candidate must re-select the window from
the **un-pruned pool**, not from a previous lever's pruned ``Plan``. The driver re-derives that pool
as ``seed_beam_hkl(grid, g_max_refine)`` -- so the pool is *derived* from the grid + the live
``g_max_refine`` scalar, never stored.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from diffBloch.engine.plan import OrientationPlan
from diffBloch.preprocess.experiment import seed_beam_hkl
from diffBloch.preprocess.plan import Plan
from diffBloch.preprocess.steps.beams import select_beams
from diffBloch.preprocess.steps.coverage import maximize_scalar, plan_coverage
from diffBloch.specs import BeamSelection

__all__ = [
    "ConvergenceState",
    "run_coverage_phase",
]


@dataclass(frozen=True)
class ConvergenceState:
    """The driver's coordinate-descent state -- the live beam scalars (the State monad's ``s``).

    ``g_max_refine`` is the candidate-pool radius; ``integration_semiangle`` is the Klar window. The
    un-pruned pool is *not* a field: it is re-derived as ``seed_beam_hkl(grid, g_max_refine)`` from
    whichever ``Plan`` the driver is transforming, so there is no stored copy to desync. The
    self-stability phase will extend this with the rocking-curve tilt count when it lands.
    """

    g_max_refine: float
    integration_semiangle: float


def run_coverage_phase(
    plan: Plan,
    state: ConvergenceState,
    selection: BeamSelection,
    *,
    pool_step: float,
    window_step: float,
    max_iterations: int = 100,
) -> tuple[Plan, ConvergenceState]:
    """Grow the pool then the window to the minimum that maximises coverage; thread the scalars.

    The coverage phase (``State ConvergenceState Plan``): a single ordered pass -- pool
    (``g_max_refine``) then window (``integration_semiangle``) -- each swept by
    :func:`maximize_scalar`
    to the smallest value that still strictly increases :func:`plan_coverage` (matched observed
    reflections). Faithful to the private ``_run_initial_minimum_param_sweep``, minus the tilt knob:
    2.0 coverage is pure geometric membership, so the tilt count cannot move it (see
    ``DIVERGENCE.md``);
    tilts are tuned in the self-stability phase instead.

    Each candidate is built from the **un-pruned pool** re-derived at the current ``g_max_refine``
    (obstruction 1), so widening the window can recover beams a narrower pool clipped. ``selection``
    supplies the fixed ``rsg`` / ``dsg`` / ``geometry``; its ``integration_semiangle`` is ignored --
    the live window comes from ``state``. Returns the coverage-maximising ``Plan`` and the settled
    :class:`ConvergenceState` (for the ``both`` handoff to self-stability). ``pool_step`` /
    ``window_step`` must be positive.
    """
    if pool_step <= 0.0:
        raise ValueError("pool_step must be positive")
    if window_step <= 0.0:
        raise ValueError("window_step must be positive")

    g_max_refine = maximize_scalar(
        lambda value: value,
        lambda value: plan_coverage(
            _windowed_pool(plan, value, state.integration_semiangle, selection)
        ),
        start=state.g_max_refine,
        step=pool_step,
        max_iterations=max_iterations,
    )
    integration_semiangle = maximize_scalar(
        lambda value: value,
        lambda value: plan_coverage(_windowed_pool(plan, g_max_refine, value, selection)),
        start=state.integration_semiangle,
        step=window_step,
        max_iterations=max_iterations,
    )
    settled = ConvergenceState(
        g_max_refine=g_max_refine,
        integration_semiangle=integration_semiangle,
    )
    return _windowed_pool(plan, g_max_refine, integration_semiangle, selection), settled


def _windowed_pool(
    plan: Plan,
    g_max_refine: float,
    integration_semiangle: float,
    selection: BeamSelection,
) -> Plan:
    """Re-seed every orientation from the grid at ``g_max_refine``, then apply the Klar window.

    The driver's build step: ``select_beams(window) applied to seed(g_max_refine)``. Re-seeding from
    the shared grid (not a previous pruned ``Plan``) is what honours obstruction 1. Guards the
    ``Fgb`` difference support exactly like
    :func:`~diffBloch.preprocess.steps.convergence.converge_pool`.
    """
    if 2.0 * g_max_refine > plan.grid.g_max:
        raise ValueError(
            f"g_max_refine={g_max_refine:.4g} exceeds the grid's beam-difference support "
            f"(g_max={plan.grid.g_max:.4g}); dependent grid resizing is not implemented"
        )
    beam_hkl = seed_beam_hkl(plan.grid, g_max_refine=g_max_refine)
    reseeded = tuple(
        OrientationPlan.build(
            plan.grid,
            beam_hkl,
            op.pattern,
            energy=op.energy,
            thickness=op.thickness,
            u0=op.u0,
            orientation=op.orientation,
        )
        for op in plan.orientations
    )
    windowed = replace(selection, integration_semiangle=integration_semiangle)
    return select_beams(windowed)(replace(plan, orientations=reseeded))

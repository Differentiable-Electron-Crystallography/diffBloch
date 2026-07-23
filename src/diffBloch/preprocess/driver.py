"""The preprocess convergence driver: threads state through the coupled beam-knob sweeps.

The individual convergence levers (``converge_beams`` / ``converge_pool`` / ``converge_sampling``)
and the coverage levers (``cover_beams`` / ``cover_pool``) are pure ``Plan -> Plan`` steps, each
self-contained. But the *coupled* sweep -- pool and window tuned together, and coverage handing its
settled knobs to self-stability (the ``both`` operation) -- needs state the ``Plan`` deliberately
does not carry: the live scalars ``g_max_refine`` / ``integration_semiangle`` / ``rocking_curve_sampling``.
This module is the **driver** that owns that state.

:class:`ConvergenceState` holds that state; a *phase* (:func:`run_coverage_phase`,
:func:`run_stability_phase`) is a ``(Plan, ConvergenceState) -> (Plan, ConvergenceState)``
computation; the driver threads the state between phases and, at its outer boundary, returns just
the converged ``Plan``.

The cross-lever fixpoint: each candidate must re-select the window from
the **un-pruned pool**, not from a previous lever's pruned ``Plan``. The driver re-derives that pool
as ``seed_beam_hkl(grid, g_max_refine)`` -- so the pool is *derived* from the grid + the live
``g_max_refine`` scalar, never stored.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from diffBloch.core.solver import SolverMethod
from diffBloch.preprocess.experiment import RefinementSetup
from diffBloch.preprocess.pipeline import PlanStep
from diffBloch.preprocess.plan import Plan
from diffBloch.preprocess.steps.beams import reseed_pool
from diffBloch.preprocess.steps.convergence import converge_scalar, simulation_rfactor
from diffBloch.preprocess.steps.coverage import maximize_scalar, plan_coverage
from diffBloch.preprocess.steps.rocking_curve import integrate_rocking_curve
from diffBloch.specs import BeamSelection, ConvergenceTest, ConvergenceTolerance, RockingCurve

__all__ = [
    "ConvergenceState",
    "converge_numerics",
    "run_coverage_phase",
    "run_stability_phase",
]


@dataclass(frozen=True)
class ConvergenceState:
    """The driver's coordinate-descent state -- the live beam scalars threaded between phases.

    ``g_max_refine`` is the candidate-pool radius; ``integration_semiangle`` is the Klar window;
    ``rocking_curve_sampling`` is the rocking-curve tilt count. The un-pruned pool is *not* a field: it is
    re-derived as ``seed_beam_hkl(grid, g_max_refine)`` from whichever ``Plan`` the driver is
    transforming, so there is no stored copy to desync. The coverage phase tunes the first two (the
    beam SET is tilt-independent under pure geometric membership); the
    self-stability phase tunes all three.
    """

    g_max_refine: float
    integration_semiangle: float
    rocking_curve_sampling: int


def converge_numerics(
    test: ConvergenceTest,
    selection: BeamSelection,
    rocking: RockingCurve,
    refinement: RefinementSetup,
    tolerance: ConvergenceTolerance,
    *,
    method: SolverMethod = "matrix_exp",
) -> PlanStep:
    """Return a ``Plan -> Plan`` step running the selected convergence operation (the driver entry).

    The driver's outer boundary: it seeds the initial :class:`ConvergenceState` (pool
    from ``test.start_g_max_refine``, window from ``selection.integration.semiangle``, tilt from
    ``rocking.sampling``), runs the ``test.operation`` phase(s), and returns just the converged
    ``Plan`` -- the settled scalars are dropped, so downstream steps never see driver state and this
    nests as one *optional* ordinary step in the preprocess pipeline (convergence stays entirely
    opt-in by construction). ``both`` runs :func:`run_coverage_phase` first and seeds
    :func:`run_stability_phase` from its settled state; the single-phase operations run one and
    discard the state. Dispatches on ``convergence_test.operation``.
    """

    def run(plan: Plan) -> Plan:
        start = ConvergenceState(
            g_max_refine=test.start_g_max_refine,
            integration_semiangle=selection.integration.semiangle,
            rocking_curve_sampling=rocking.sampling,
        )
        if test.operation == "coverage":
            converged, _ = run_coverage_phase(
                plan,
                start,
                selection,
                g_max_refine_step=test.g_max_refine_step,
                integration_semiangle_step=test.integration_semiangle_step,
                max_iterations=tolerance.max_iterations,
            )
            return converged
        if test.operation == "self_stability":
            converged, _ = run_stability_phase(
                plan,
                start,
                selection,
                rocking,
                refinement,
                tolerance,
                g_max_refine_step=test.g_max_refine_step,
                integration_semiangle_step=test.integration_semiangle_step,
                rocking_curve_sampling_step=test.rocking_curve_sampling_step,
                num_passes=test.num_passes,
                method=method,
            )
            return converged
        # "both": coverage's settled scalars seed self-stability.
        covered, settled = run_coverage_phase(
            plan,
            start,
            selection,
            g_max_refine_step=test.g_max_refine_step,
            integration_semiangle_step=test.integration_semiangle_step,
            max_iterations=tolerance.max_iterations,
        )
        converged, _ = run_stability_phase(
            covered,
            settled,
            selection,
            rocking,
            refinement,
            tolerance,
            g_max_refine_step=test.g_max_refine_step,
            integration_semiangle_step=test.integration_semiangle_step,
            rocking_curve_sampling_step=test.rocking_curve_sampling_step,
            num_passes=test.num_passes,
            method=method,
        )
        return converged

    return run


def run_coverage_phase(
    plan: Plan,
    state: ConvergenceState,
    selection: BeamSelection,
    *,
    g_max_refine_step: float,
    integration_semiangle_step: float,
    max_iterations: int = 100,
) -> tuple[Plan, ConvergenceState]:
    """Grow the pool then the window to the minimum that maximises coverage; thread the scalars.

    The coverage phase: a single ordered pass -- pool
    (``g_max_refine``) then window (``integration_semiangle``) -- each swept by
    :func:`maximize_scalar`
    to the smallest value that still strictly increases :func:`plan_coverage` (matched observed
    reflections). Coverage is pure geometric membership, so the tilt count cannot move it;
    tilts are tuned in the self-stability phase instead.

    Each candidate is built from the **un-pruned pool** re-derived at the current ``g_max_refine``
    (the cross-lever fixpoint), so widening the window can recover beams a narrower pool clipped. ``selection``
    supplies the fixed ``rsg`` / ``dsg`` / ``geometry``; its ``integration.semiangle`` is ignored --
    the live window comes from ``state``. Returns the coverage-maximising ``Plan`` and the settled
    :class:`ConvergenceState` (for the ``both`` handoff to self-stability). ``g_max_refine_step`` /
    ``integration_semiangle_step`` must be positive.
    """
    if g_max_refine_step <= 0.0:
        raise ValueError("g_max_refine_step must be positive")
    if integration_semiangle_step <= 0.0:
        raise ValueError("integration_semiangle_step must be positive")

    g_max_refine = maximize_scalar(
        lambda value: value,
        lambda value: plan_coverage(
            _windowed_pool(plan, value, state.integration_semiangle, selection)
        ),
        start=state.g_max_refine,
        step=g_max_refine_step,
        max_iterations=max_iterations,
    )
    integration_semiangle = maximize_scalar(
        lambda value: value,
        lambda value: plan_coverage(_windowed_pool(plan, g_max_refine, value, selection)),
        start=state.integration_semiangle,
        step=integration_semiangle_step,
        max_iterations=max_iterations,
    )
    settled = ConvergenceState(
        g_max_refine=g_max_refine,
        integration_semiangle=integration_semiangle,
        rocking_curve_sampling=state.rocking_curve_sampling,
    )
    return _windowed_pool(plan, g_max_refine, integration_semiangle, selection), settled


def run_stability_phase(
    plan: Plan,
    state: ConvergenceState,
    selection: BeamSelection,
    rocking: RockingCurve,
    refinement: RefinementSetup,
    tolerance: ConvergenceTolerance,
    *,
    g_max_refine_step: float,
    integration_semiangle_step: float,
    rocking_curve_sampling_step: float,
    num_passes: int = 2,
    method: SolverMethod = "matrix_exp",
) -> tuple[Plan, ConvergenceState]:
    """Grow all three knobs to self-stability over ``num_passes`` coordinated passes; thread them.

    The self-stability phase (``State ConvergenceState Plan``): a fixed ``num_passes`` coordinate
    sweep over the pool (``g_max_refine``), window (``integration_semiangle``) and tilt
    (``rocking_curve_sampling``) knobs, each driven by :func:`converge_scalar` to the first knob value whose
    *consecutive-simulation* R-factor drops below ``tolerance.r_factor_threshold`` (a numerical
    resolution study, not an accuracy fit -- so it needs ``refinement``, unlike the pure-geometry
    coverage phase). Pass 1 sweeps
    pool -> tilt -> window; pass 2+ sweeps tilt -> pool -> window (a per-pass
    order-swap, revisiting each knob after the others moved). Each settled scalar threads into the
    next sweep
    and the next pass via the running ``(g_max_refine, integration_semiangle, rocking_curve_sampling)``.

    Every candidate is a full simulation Plan rebuilt from the three scalars
    (:func:`_stability_build` -- pool + window via :func:`_windowed_pool`, then rocking-curve tilt
    integration), so the sweeps are a pure function of the knobs, never an incrementally-mutated
    Plan. ``selection`` supplies the
    fixed ``rsg`` / ``dsg`` / ``geometry`` and ``rocking`` the fixed tilt span + geometry; their
    ``integration.semiangle`` / ``sampling`` are ignored -- the live values come from ``state``.
    Returns the self-stable ``Plan`` and settled :class:`ConvergenceState`. ``g_max_refine_step`` /
    ``integration_semiangle_step`` / ``rocking_curve_sampling_step`` must be positive and ``num_passes`` at least 1.
    """
    if g_max_refine_step <= 0.0:
        raise ValueError("g_max_refine_step must be positive")
    if integration_semiangle_step <= 0.0:
        raise ValueError("integration_semiangle_step must be positive")
    if rocking_curve_sampling_step <= 0.0:
        raise ValueError("rocking_curve_sampling_step must be positive")
    if num_passes < 1:
        raise ValueError("num_passes must be at least 1")

    measure = simulation_rfactor(refinement, method=method)
    steps = {
        "pool": g_max_refine_step,
        "window": integration_semiangle_step,
        "tilt": rocking_curve_sampling_step,
    }

    def sweep(
        g_max_refine: float, integration_semiangle: float, tilt: float, *, knob: str
    ) -> float:
        def build(value: float) -> Plan:
            return _stability_build(
                plan,
                value if knob == "pool" else g_max_refine,
                value if knob == "window" else integration_semiangle,
                value if knob == "tilt" else tilt,
                selection,
                rocking,
            )

        start = {"pool": g_max_refine, "window": integration_semiangle, "tilt": tilt}[knob]
        return converge_scalar(
            lambda value: value,
            lambda previous, candidate: measure(build(previous), build(candidate)),
            tolerance,
            start=start,
            step=steps[knob],
        )

    g_max_refine = state.g_max_refine
    integration_semiangle = state.integration_semiangle
    tilt = float(state.rocking_curve_sampling)
    for pass_idx in range(1, num_passes + 1):
        order = ("pool", "tilt", "window") if pass_idx == 1 else ("tilt", "pool", "window")
        for knob in order:
            settled_value = sweep(g_max_refine, integration_semiangle, tilt, knob=knob)
            if knob == "pool":
                g_max_refine = settled_value
            elif knob == "window":
                integration_semiangle = settled_value
            else:
                tilt = settled_value

    rocking_curve_sampling = int(round(tilt))
    settled = ConvergenceState(
        g_max_refine=g_max_refine,
        integration_semiangle=integration_semiangle,
        rocking_curve_sampling=rocking_curve_sampling,
    )
    final = _stability_build(
        plan, g_max_refine, integration_semiangle, rocking_curve_sampling, selection, rocking
    )
    return final, settled


def _stability_build(
    plan: Plan,
    g_max_refine: float,
    integration_semiangle: float,
    rocking_curve_sampling: float,
    selection: BeamSelection,
    rocking: RockingCurve,
) -> Plan:
    """The full simulation Plan as a pure function of the three self-stability knobs.

    Pool + window via :func:`_windowed_pool` (which fixes the beam SET), then rocking-curve tilt
    integration at ``rocking_curve_sampling`` on that set. Order matters: the pool/window decide *which*
    beams, the tilt count integrates the rocking curve over them. Each candidate is a fresh build
    from ``g_max`` / ``sg_max`` / ``tilt_steps``, never an incrementally-mutated Plan.
    """
    pooled = _windowed_pool(plan, g_max_refine, integration_semiangle, selection)
    tilts = replace(rocking, sampling=int(round(rocking_curve_sampling)))
    return integrate_rocking_curve(tilts)(pooled)


def _windowed_pool(
    plan: Plan,
    g_max_refine: float,
    integration_semiangle: float,
    selection: BeamSelection,
) -> Plan:
    """Re-seed the pool at ``g_max_refine`` and apply the window ``integration_semiangle``.

    The driver's build step, over :func:`~diffBloch.preprocess.steps.beams.reseed_pool` (the shared
    reseed-and-window builder, which also carries the ``Fgb`` difference-support guard). Unlike the
    standalone pool levers, the driver *varies the window too*, so it overrides
    ``selection.integration.semiangle`` with the swept value before delegating.
    """
    windowed = replace(
        selection, integration=replace(selection.integration, semiangle=integration_semiangle)
    )
    return reseed_pool(plan, windowed, g_max_refine=g_max_refine)

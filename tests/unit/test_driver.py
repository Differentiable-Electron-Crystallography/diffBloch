"""Slice 6: the convergence driver's coverage + self-stability phases.

``run_coverage_phase`` grows the pool then the window to the coverage-maximising minimum (pure
geometry); ``run_stability_phase`` grows all three knobs (pool / window / tilt) to consecutive-
simulation self-stability over a fixed ``num_passes`` coordinate sweep with the private's per-pass
order-swap. Both are ``(Plan, ConvergenceState) -> (Plan, ConvergenceState)`` computations,
exercised
on the fast synthetic cube from ``test_convergence``: for coverage a narrow start window drives the
*window* sweep (the pool is starved), a wider start window drives the *pool* sweep; for stability
the settled scalars and their monotone growth across passes are pinned from an empirical probe.
"""

from __future__ import annotations

import pytest
from tests.unit.test_convergence import _beam_count, _seed_system

from diffBloch.preprocess.driver import (
    ConvergenceState,
    _windowed_pool,
    run_coverage_phase,
    run_stability_phase,
)
from diffBloch.preprocess.steps.coverage import plan_coverage
from diffBloch.specs import BeamSelection, ConvergenceTolerance, RockingCurve

_SELECTION = BeamSelection(integration_semiangle=1.0)  # rsg/dsg/geometry fixed; angle from state


def _coverage(plan, g_max_refine: float, integration_semiangle: float) -> int:
    return plan_coverage(_windowed_pool(plan, g_max_refine, integration_semiangle, _SELECTION))


def test_coverage_phase_window_sweep_grows_the_window_when_the_pool_is_narrow() -> None:
    _, seed = _seed_system()
    # Narrow start window: growing the pool adds nothing (the window clips it), so only the window
    # sweep moves -- 0.68 -> 0.88 -- and the pool is left at its start.
    start = ConvergenceState(g_max_refine=0.5, integration_semiangle=0.68, tilt_sampling=1)
    plan, settled = run_coverage_phase(
        seed, start, _SELECTION, pool_step=0.1, window_step=0.2, max_iterations=30
    )
    assert settled.g_max_refine == 0.5
    assert settled.integration_semiangle == pytest.approx(0.88)
    assert settled.tilt_sampling == 1  # coverage is pure geometry -- tilt passes through untouched
    assert plan_coverage(plan) == 13
    assert _coverage(seed, 0.5, 0.68) == 9  # grew coverage 9 -> 13


def test_coverage_phase_pool_sweep_grows_the_pool_when_the_window_is_wide() -> None:
    _, seed = _seed_system()
    # Wider start window: the pool is no longer starved, so the pool sweep moves 0.5 -> 0.9.
    start = ConvergenceState(g_max_refine=0.5, integration_semiangle=1.2, tilt_sampling=1)
    plan, settled = run_coverage_phase(
        seed, start, _SELECTION, pool_step=0.1, window_step=0.2, max_iterations=30
    )
    assert settled.g_max_refine == pytest.approx(0.9)
    assert settled.integration_semiangle == 1.2
    assert plan_coverage(plan) == 39
    assert _coverage(seed, 0.5, 1.2) == 17  # grew coverage 17 -> 39


def test_coverage_phase_returns_the_plan_at_the_settled_scalars() -> None:
    _, seed = _seed_system()
    start = ConvergenceState(g_max_refine=0.5, integration_semiangle=1.2, tilt_sampling=1)
    plan, settled = run_coverage_phase(
        seed, start, _SELECTION, pool_step=0.1, window_step=0.2, max_iterations=30
    )
    # The returned Plan is exactly the windowed pool at the settled state (the evalState value).
    assert plan_coverage(plan) == _coverage(
        seed, settled.g_max_refine, settled.integration_semiangle
    )


def test_coverage_phase_rejects_non_positive_steps() -> None:
    _, seed = _seed_system()
    start = ConvergenceState(g_max_refine=0.5, integration_semiangle=1.0, tilt_sampling=1)
    with pytest.raises(ValueError, match="pool_step must be positive"):
        run_coverage_phase(seed, start, _SELECTION, pool_step=0.0, window_step=0.2)
    with pytest.raises(ValueError, match="window_step must be positive"):
        run_coverage_phase(seed, start, _SELECTION, pool_step=0.1, window_step=0.0)


# --- run_stability_phase: the fixed num_passes coordinate sweep to consecutive-sim stability ---

_ROCKING = RockingCurve(semiangle=0.5, sampling=1)  # tilt span/geometry fixed; count from state
_TOLERANCE = ConvergenceTolerance(r_factor_threshold=0.05, max_iterations=20)
_START = ConvergenceState(g_max_refine=0.5, integration_semiangle=0.68, tilt_sampling=1)


def _stability(refinement, seed, **kwargs):
    return run_stability_phase(
        seed,
        _START,
        _SELECTION,
        _ROCKING,
        refinement,
        _TOLERANCE,
        pool_step=kwargs.get("pool_step", 0.1),
        window_step=kwargs.get("window_step", 0.2),
        tilt_step=kwargs.get("tilt_step", 2),
        num_passes=kwargs.get("num_passes", 2),
    )


def test_stability_phase_one_pass_settles_all_three_knobs() -> None:
    refinement, seed = _seed_system()
    # One pass (pool -> tilt -> window): each knob grows to its first below-threshold
    # consecutive-sim step. Values pinned from an empirical probe of the synthetic cube.
    plan, settled = _stability(refinement, seed, num_passes=1)
    assert settled.g_max_refine == pytest.approx(0.6)
    assert settled.integration_semiangle == pytest.approx(1.08)
    assert settled.tilt_sampling == 7  # tilt IS tuned here (unlike coverage), 1 -> 7
    assert _beam_count(plan) == 17
    assert len(plan.orientations[0].tilts) == 7  # returned Plan is built at the settled state


def test_stability_phase_second_pass_revisits_and_grows_the_knobs() -> None:
    refinement, seed = _seed_system()
    # A second pass (tilt -> pool -> window) revisits each knob after the others moved: every scalar
    # is >= its one-pass value and at least one strictly grows (the order-swap does work).
    _, one = _stability(refinement, seed, num_passes=1)
    _, two = _stability(refinement, seed, num_passes=2)
    assert two.g_max_refine == pytest.approx(0.7)
    assert two.integration_semiangle == pytest.approx(1.28)
    assert two.tilt_sampling == 9
    assert two.g_max_refine >= one.g_max_refine
    assert two.integration_semiangle >= one.integration_semiangle
    assert two.tilt_sampling >= one.tilt_sampling


def test_stability_phase_rejects_bad_steps_and_passes() -> None:
    refinement, seed = _seed_system()
    with pytest.raises(ValueError, match="pool_step must be positive"):
        _stability(refinement, seed, pool_step=0.0)
    with pytest.raises(ValueError, match="window_step must be positive"):
        _stability(refinement, seed, window_step=0.0)
    with pytest.raises(ValueError, match="tilt_step must be positive"):
        _stability(refinement, seed, tilt_step=0.0)
    with pytest.raises(ValueError, match="num_passes must be at least 1"):
        _stability(refinement, seed, num_passes=0)

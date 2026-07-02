"""Slice 6 (commit 1): the convergence driver's coverage phase.

``run_coverage_phase`` is the ``(Plan, ConvergenceState) -> (Plan, ConvergenceState)`` computation
that grows the pool then the window to the coverage-maximising minimum, threading the settled
scalars for the ``both`` handoff. Exercised on the fast synthetic cube from ``test_convergence``:
a narrow start window drives the *window* sweep (the pool is starved), a wider start window drives
the *pool* sweep -- so the two coordinate directions are pinned independently.
"""

from __future__ import annotations

import pytest
from tests.unit.test_convergence import _seed_system

from diffBloch.preprocess.coverage import plan_coverage
from diffBloch.preprocess.driver import ConvergenceState, _windowed_pool, run_coverage_phase
from diffBloch.specs import BeamSelection

_SELECTION = BeamSelection(integration_semiangle=1.0)  # rsg/dsg/geometry fixed; angle from state


def _coverage(plan, g_max_refine: float, integration_semiangle: float) -> int:
    return plan_coverage(_windowed_pool(plan, g_max_refine, integration_semiangle, _SELECTION))


def test_coverage_phase_window_sweep_grows_the_window_when_the_pool_is_narrow() -> None:
    _, seed = _seed_system()
    # Narrow start window: growing the pool adds nothing (the window clips it), so only the window
    # sweep moves -- 0.68 -> 0.88 -- and the pool is left at its start.
    start = ConvergenceState(g_max_refine=0.5, integration_semiangle=0.68)
    plan, settled = run_coverage_phase(
        seed, start, _SELECTION, pool_step=0.1, window_step=0.2, max_iterations=30
    )
    assert settled.g_max_refine == 0.5
    assert settled.integration_semiangle == pytest.approx(0.88)
    assert plan_coverage(plan) == 13
    assert _coverage(seed, 0.5, 0.68) == 9  # grew coverage 9 -> 13


def test_coverage_phase_pool_sweep_grows_the_pool_when_the_window_is_wide() -> None:
    _, seed = _seed_system()
    # Wider start window: the pool is no longer starved, so the pool sweep moves 0.5 -> 0.9.
    start = ConvergenceState(g_max_refine=0.5, integration_semiangle=1.2)
    plan, settled = run_coverage_phase(
        seed, start, _SELECTION, pool_step=0.1, window_step=0.2, max_iterations=30
    )
    assert settled.g_max_refine == pytest.approx(0.9)
    assert settled.integration_semiangle == 1.2
    assert plan_coverage(plan) == 39
    assert _coverage(seed, 0.5, 1.2) == 17  # grew coverage 17 -> 39


def test_coverage_phase_returns_the_plan_at_the_settled_scalars() -> None:
    _, seed = _seed_system()
    start = ConvergenceState(g_max_refine=0.5, integration_semiangle=1.2)
    plan, settled = run_coverage_phase(
        seed, start, _SELECTION, pool_step=0.1, window_step=0.2, max_iterations=30
    )
    # The returned Plan is exactly the windowed pool at the settled state (the evalState value).
    assert plan_coverage(plan) == _coverage(
        seed, settled.g_max_refine, settled.integration_semiangle
    )


def test_coverage_phase_rejects_non_positive_steps() -> None:
    _, seed = _seed_system()
    start = ConvergenceState(g_max_refine=0.5, integration_semiangle=1.0)
    with pytest.raises(ValueError, match="pool_step must be positive"):
        run_coverage_phase(seed, start, _SELECTION, pool_step=0.0, window_step=0.2)
    with pytest.raises(ValueError, match="window_step must be positive"):
        run_coverage_phase(seed, start, _SELECTION, pool_step=0.1, window_step=0.0)

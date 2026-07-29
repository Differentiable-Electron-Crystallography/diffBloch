"""Simulation convergence over g_max, sg_max, and rocking-curve tilt steps."""

from __future__ import annotations

import pytest
from tests.unit.synthetic import built_seed_system

from diffBloch.engine.plan import CoupledOrientationPlan
from diffBloch.preprocess.driver import ConvergenceState, converge_numerics, run_convergence
from diffBloch.specs import (
    ConvergenceTest,
    ConvergenceTolerance,
    IntegrationGeometry,
    RockingCurve,
    SegmentedUnionCoupling,
)

_ROCKING = RockingCurve(sampling=3, integration=IntegrationGeometry(semiangle=0.5))
_SIMULATION = SegmentedUnionCoupling(
    fixed_n_segments=2,
    g_max=0.5,
    sg_max=0.01,
)
_TEST = ConvergenceTest(
    g_max_step=0.1,
    sg_max_step=0.005,
    tilt_steps_step=2,
    num_passes=1,
)
_TOLERANCE = ConvergenceTolerance(r_factor_threshold=2.0, max_iterations=2)


def test_convergence_sweeps_g_max_sg_max_and_tilt_steps() -> None:
    refinement, seed = built_seed_system()
    plan, settled = run_convergence(
        seed,
        ConvergenceState(g_max=0.5, sg_max=0.01, tilt_steps=3),
        _TEST,
        _ROCKING,
        _SIMULATION,
        refinement,
        _TOLERANCE,
    )

    assert settled == ConvergenceState(g_max=pytest.approx(0.5), sg_max=0.01, tilt_steps=3)
    assert isinstance(plan.orientations[0], CoupledOrientationPlan)
    assert len(plan.orientations[0].tilts) == 3


def test_converge_numerics_returns_a_pure_plan_step() -> None:
    refinement, seed = built_seed_system()
    converged = converge_numerics(
        _TEST,
        _ROCKING,
        _SIMULATION,
        refinement,
        _TOLERANCE,
    )(seed)

    assert isinstance(converged.orientations[0], CoupledOrientationPlan)
    assert len(converged.orientations[0].tilts) == 3
    assert not isinstance(seed.orientations[0], CoupledOrientationPlan)

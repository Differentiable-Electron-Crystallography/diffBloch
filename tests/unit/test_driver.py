"""Simulation convergence over g_max, sg_max, and rocking-curve tilt steps."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tests.unit.synthetic import built_seed_system

from diffBloch.engine.plan import CoupledOrientationPlan
from diffBloch.preprocess.driver import ConvergenceState, convergence_test_numerics, run_convergence
from diffBloch.specs import (
    ConvergenceTest,
    ConvergenceTolerance,
    IntegrationGeometry,
    RockingCurve,
    UnionCoupling,
)

_ROCKING = RockingCurve(sampling=3, integration=IntegrationGeometry(semiangle=0.5))
_SIMULATION = UnionCoupling(
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


def test_convergence_test_numerics_returns_a_pure_plan_step() -> None:
    refinement, seed = built_seed_system()
    converged = convergence_test_numerics(
        _TEST,
        _ROCKING,
        _SIMULATION,
        refinement,
        _TOLERANCE,
    )(seed)

    assert isinstance(converged.orientations[0], CoupledOrientationPlan)
    assert len(converged.orientations[0].tilts) == 3
    assert not isinstance(seed.orientations[0], CoupledOrientationPlan)


def test_comparison_set_is_observed_above_one_sigma_within_g_max() -> None:
    from diffBloch.core.products import PatternBatch
    from diffBloch.preprocess.driver import _observed_hkl

    pattern = PatternBatch(
        hkl=torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [3, 0, 0]], dtype=torch.int64),
        intensities=torch.tensor([9.0, 9.0, 0.5, 9.0], dtype=torch.float64),
        sigmas=torch.ones(4, dtype=torch.float64),
    )
    kept = _observed_hkl(pattern, np.eye(3), g_max=2.0)
    assert kept.tolist() == [[1, 0, 0]]
    with pytest.raises(ValueError, match="I > sigma"):
        _observed_hkl(pattern, np.eye(3), g_max=0.5)

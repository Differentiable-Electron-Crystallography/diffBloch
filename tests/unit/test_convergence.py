"""Slice 11: the convergence machinery -- ``simulation_converged``, ``converge_scalar``,
``converge_beams``.

Uses the same fast synthetic silicon system as ``test_fit_thickness`` (no heavy fixture sim). The
check compares two *simulations*, so the orientations' observed patterns are irrelevant
placeholders;
what matters is that two Plans with different beam sets produce a non-zero R-factor that the
tolerance threshold gates, and that identical Plans read as converged.

``converge_scalar`` (the parameter-agnostic driver) is tested purely with a scripted R-factor
sequence and an identity ``build`` -- no simulation -- so the skip-null / patience / cap logic is
pinned in isolation. ``converge_beams`` is then exercised end-to-end: widening the Klar window over
a richer seed until the pattern saturates.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from diffBloch.core.products import PatternBatch
from diffBloch.core.symmetry import build_asu_expansion_plan
from diffBloch.engine import OrientationPlan, ScatteringGrid
from diffBloch.params import ConstraintSpec, RefinableParams
from diffBloch.preprocess import (
    RefinementSetup,
    converge_beams,
    converge_scalar,
    select_beams,
    simulation_converged,
)
from diffBloch.preprocess.plan import Plan
from diffBloch.specs import BeamSelection, ConvergenceTolerance

_ENERGY = 200e3
_CELL = np.eye(3, dtype=np.float64) * 5.0
_FULL_BEAMS = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.int64)
_PRUNED_BEAMS = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64)  # one fewer coupled beam
_THICKNESS = 300.0


def _silicon() -> tuple[ScatteringGrid, object, ConstraintSpec, torch.Tensor]:
    grid = ScatteringGrid.from_cell(_CELL, g_max=0.45)
    asu_plan = build_asu_expansion_plan(np.zeros((1, 3)), np.eye(3)[None], np.zeros((1, 3)))
    spec = ConstraintSpec(
        fixed_positions=torch.zeros((1, 3), dtype=torch.float64),
        refinable_position_mask=torch.ones((1, 3), dtype=torch.float64),
        occupancies=torch.ones(1, dtype=torch.float64),
        reciprocal_basis=grid.reciprocal_basis,
    )
    numbers = torch.tensor([14], dtype=torch.int64)
    return grid, asu_plan, spec, numbers


def _params() -> RefinableParams:
    return RefinableParams(
        asu_positions=torch.zeros((1, 3), dtype=torch.float64),
        uij_raw=torch.eye(3, dtype=torch.float64)[None] * 0.1,
    )


def _refinement(asu_plan: object, spec: ConstraintSpec, numbers: torch.Tensor) -> RefinementSetup:
    return RefinementSetup(
        asu_plan=asu_plan,  # type: ignore[arg-type]
        spec=spec,
        params=_params(),
        numbers=numbers,
    )


def _orientation(grid: ScatteringGrid, beam_hkl: np.ndarray) -> OrientationPlan:
    pattern = PatternBatch(
        hkl=torch.tensor(beam_hkl, dtype=torch.int64),
        intensities=torch.zeros(len(beam_hkl), dtype=torch.float64),
        sigmas=torch.ones(len(beam_hkl), dtype=torch.float64),
    )
    return OrientationPlan.build(grid, beam_hkl, pattern, energy=_ENERGY, thickness=(_THICKNESS,))


def test_identical_plans_read_as_converged() -> None:
    grid, asu_plan, spec, numbers = _silicon()
    refinement = _refinement(asu_plan, spec, numbers)
    plan = Plan(grid=grid, orientations=(_orientation(grid, _FULL_BEAMS),))

    check = simulation_converged(refinement, ConvergenceTolerance(r_factor_threshold=0.005))

    # Comparing a Plan against itself: R-factor is ~0, well under any positive threshold.
    assert check(plan, plan) is True


def test_changed_beam_set_is_gated_by_the_threshold() -> None:
    grid, asu_plan, spec, numbers = _silicon()
    refinement = _refinement(asu_plan, spec, numbers)
    previous = Plan(grid=grid, orientations=(_orientation(grid, _FULL_BEAMS),))
    current = Plan(grid=grid, orientations=(_orientation(grid, _PRUNED_BEAMS),))

    # Dropping a coupled beam changes the dynamical intensities on the shared reflections, so the
    # consecutive-simulation R-factor is non-zero: a tight threshold rejects it, a loose one
    # accepts.
    tight = simulation_converged(refinement, ConvergenceTolerance(r_factor_threshold=1e-6))
    loose = simulation_converged(refinement, ConvergenceTolerance(r_factor_threshold=10.0))
    assert tight(previous, current) is False
    assert loose(previous, current) is True


def test_mismatched_orientation_count_is_rejected() -> None:
    grid, asu_plan, spec, numbers = _silicon()
    refinement = _refinement(asu_plan, spec, numbers)
    one = Plan(grid=grid, orientations=(_orientation(grid, _FULL_BEAMS),))
    two = Plan(grid=grid, orientations=(_orientation(grid, _FULL_BEAMS),) * 2)

    check = simulation_converged(refinement, ConvergenceTolerance())
    with pytest.raises(ValueError, match="share their orientations"):
        check(one, two)


# --- converge_scalar: the parameter-agnostic driver, tested with a scripted R-factor sequence ---


def _scripted_measure(r_values: list[float]) -> object:
    """A ``measure(previous, current)`` that ignores its Plans and returns the next scripted R."""
    sequence = iter(r_values)

    def measure(previous: float, current: float) -> float:
        return next(sequence)

    return measure


def test_converge_scalar_reaches_patience_over_settled_steps() -> None:
    # Two consecutive below-threshold steps (patience=2) declare convergence; the returned value is
    # the knob at the settling step (1.0 + 3 clicks of 0.1).
    tolerance = ConvergenceTolerance(r_factor_threshold=0.005, patience=2, max_iterations=10)
    measure = _scripted_measure([0.02, 0.003, 0.002])
    result = converge_scalar(lambda v: v, measure, tolerance, start=1.0, step=0.1)
    assert result == pytest.approx(1.3)


def test_converge_scalar_skips_null_steps_before_settling() -> None:
    # Null steps at the start (R=0, the discrete output has not changed yet) must NOT trigger
    # convergence -- the sweep keeps growing until a genuine change settles below threshold. This is
    # the private's plateau bug, corrected.
    tolerance = ConvergenceTolerance(r_factor_threshold=0.005, patience=2, max_iterations=10)
    measure = _scripted_measure([0.0, 0.0, 0.02, 0.004, 0.003])
    result = converge_scalar(lambda v: v, measure, tolerance, start=1.0, step=0.1)
    assert result == pytest.approx(1.5)


def test_converge_scalar_treats_saturation_nulls_as_converged() -> None:
    # Once a changed step has settled below threshold, a following null step (the knob saturating,
    # e.g. the beam pool exhausted) confirms convergence rather than being skipped.
    tolerance = ConvergenceTolerance(r_factor_threshold=0.005, patience=2, max_iterations=10)
    measure = _scripted_measure([0.02, 0.003, 0.0])
    result = converge_scalar(lambda v: v, measure, tolerance, start=1.0, step=0.1)
    assert result == pytest.approx(1.2)


def test_converge_scalar_raises_when_it_never_settles() -> None:
    tolerance = ConvergenceTolerance(r_factor_threshold=0.005, patience=2, max_iterations=3)
    measure = _scripted_measure([0.02, 0.02, 0.02])
    with pytest.raises(RuntimeError, match="did not converge within 3 steps"):
        converge_scalar(lambda v: v, measure, tolerance, start=1.0, step=0.1)


# --- converge_beams: the beam-window adapter, exercised end-to-end on a richer synthetic seed ---

# A seed rich enough that widening the Klar window admits beams progressively then saturates: an
# hkl cube whose outer beams are only kept at wider integration angles.
_SEED_BEAMS = np.array(
    [
        [hkl_h, hkl_k, hkl_l]
        for hkl_h in range(-3, 4)
        for hkl_k in range(-3, 4)
        for hkl_l in range(-3, 4)
    ],
    dtype=np.int64,
)


def _seed_system() -> tuple[RefinementSetup, Plan]:
    grid = ScatteringGrid.from_cell(_CELL, g_max=2.2)
    asu_plan = build_asu_expansion_plan(np.zeros((1, 3)), np.eye(3)[None], np.zeros((1, 3)))
    spec = ConstraintSpec(
        fixed_positions=torch.zeros((1, 3), dtype=torch.float64),
        refinable_position_mask=torch.ones((1, 3), dtype=torch.float64),
        occupancies=torch.ones(1, dtype=torch.float64),
        reciprocal_basis=grid.reciprocal_basis,
    )
    refinement = RefinementSetup(
        asu_plan=asu_plan,  # type: ignore[arg-type]
        spec=spec,
        params=_params(),
        numbers=torch.tensor([14], dtype=torch.int64),
    )
    pattern = PatternBatch(
        hkl=torch.tensor(_SEED_BEAMS, dtype=torch.int64),
        intensities=torch.zeros(len(_SEED_BEAMS), dtype=torch.float64),
        sigmas=torch.ones(len(_SEED_BEAMS), dtype=torch.float64),
    )
    op = OrientationPlan.build(grid, _SEED_BEAMS, pattern, energy=_ENERGY, thickness=(_THICKNESS,))
    return refinement, Plan(grid=grid, orientations=(op,))


def _beam_count(plan: Plan) -> int:
    return len(plan.orientations[0].beam_hkl)


def test_converge_beams_widens_the_window_until_the_pattern_saturates() -> None:
    refinement, seed = _seed_system()
    # Step wide enough to cross the intermediate count plateaus (the seed admits beams in bands as
    # the window widens: 15 -> 39 -> 43); the sweep then saturates and the trailing nulls settle it
    # on the fully-selected beam set.
    step = converge_beams(
        BeamSelection(integration_semiangle=0.68),
        refinement,
        ConvergenceTolerance(r_factor_threshold=0.05, patience=2, max_iterations=20),
        step=0.6,
    )
    converged = step(seed)

    # It widened past the starting selection, all the way to the fully-reachable (saturated) set.
    started = select_beams(BeamSelection(integration_semiangle=0.68))(seed)
    saturated = select_beams(BeamSelection(integration_semiangle=5.0))(seed)
    assert _beam_count(started) < _beam_count(converged) == _beam_count(saturated)


def test_converge_beams_raises_when_no_change_settles_below_threshold() -> None:
    refinement, seed = _seed_system()
    # Tight threshold: the near-Ewald change stays above it, so no step settles; the trailing nulls
    # are skipped and the cap raises rather than declaring false convergence.
    step = converge_beams(
        BeamSelection(integration_semiangle=0.68),
        refinement,
        ConvergenceTolerance(r_factor_threshold=0.005, patience=2, max_iterations=6),
        step=0.06,
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        step(seed)


def test_converge_beams_rejects_non_positive_step() -> None:
    refinement, seed = _seed_system()
    with pytest.raises(ValueError, match="step must be positive"):
        converge_beams(BeamSelection(), refinement, ConvergenceTolerance(), step=0.0)

"""Slice 11: the convergence machinery -- ``simulation_converged``, ``converge_scalar``,
``converge_beams``, ``converge_sampling``.

Uses the same fast synthetic silicon system as ``test_fit_thickness`` (no heavy fixture sim). The
check compares two *simulations*, so the orientations' observed patterns are irrelevant
placeholders;
what matters is that two Plans with different beam sets produce a non-zero R-factor that the
tolerance threshold gates, and that identical Plans read as converged.

``converge_scalar`` (the parameter-agnostic driver) is tested purely with a scripted R-factor
sequence and an identity ``build`` -- no simulation -- so the first-below-threshold / cap logic is
pinned in isolation. ``converge_beams`` is then exercised end-to-end: widening the Klar window over
a richer seed until the pattern saturates. ``converge_sampling`` refines the rocking-curve tilt
count until the integrated pattern settles.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tests.unit.synthetic import (
    CELL,
    ENERGY,
    THICKNESS,
    beam_count,
    make_constraint_spec,
    seed_system,
    silicon_params,
)

from diffBloch.core.products import PatternBatch
from diffBloch.core.symmetry import build_asu_expansion_plan
from diffBloch.engine import OrientationPlan, StructureFactorGrid
from diffBloch.params import ConstraintSpec
from diffBloch.preprocess import (
    RefinementSetup,
    build_orientation_plans,
    converge_beams,
    converge_sampling,
    converge_scalar,
    integrate_rocking_curve,
    select_beams,
    simulation_converged,
)
from diffBloch.preprocess.plan import Plan
from diffBloch.specs import BeamSelection, ConvergenceTolerance, IntegrationGeometry, RockingCurve

_FULL_BEAMS = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.int64)
_PRUNED_BEAMS = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64)  # one fewer coupled beam


def _silicon() -> tuple[StructureFactorGrid, object, ConstraintSpec, torch.Tensor]:
    grid = StructureFactorGrid.from_cell(CELL, g_max=0.45)
    asu_plan = build_asu_expansion_plan(np.zeros((1, 3)), np.eye(3)[None], np.zeros((1, 3)))
    spec = make_constraint_spec(reciprocal_basis=grid.reciprocal_basis)
    numbers = torch.tensor([14], dtype=torch.int64)
    return grid, asu_plan, spec, numbers


def _refinement(asu_plan: object, spec: ConstraintSpec, numbers: torch.Tensor) -> RefinementSetup:
    return RefinementSetup(
        asu_plan=asu_plan,  # type: ignore[arg-type]
        spec=spec,
        params=silicon_params(),
        numbers=numbers,
    )


def _orientation(grid: StructureFactorGrid, beam_hkl: np.ndarray) -> OrientationPlan:
    pattern = PatternBatch(
        hkl=torch.tensor(beam_hkl, dtype=torch.int64),
        intensities=torch.zeros(len(beam_hkl), dtype=torch.float64),
        sigmas=torch.ones(len(beam_hkl), dtype=torch.float64),
    )
    return OrientationPlan.build(grid, beam_hkl, pattern, energy=ENERGY, thickness=(THICKNESS,))


def test_identical_plans_read_as_converged() -> None:
    grid, asu_plan, spec, numbers = _silicon()
    refinement = _refinement(asu_plan, spec, numbers)
    plan = Plan(structure_factor_grid=grid, orientations=(_orientation(grid, _FULL_BEAMS),))

    check = simulation_converged(refinement, ConvergenceTolerance(r_factor_threshold=0.005))

    # Comparing a Plan against itself: R-factor is ~0, well under any positive threshold.
    assert check(plan, plan) is True


def test_changed_beam_set_is_gated_by_the_threshold() -> None:
    grid, asu_plan, spec, numbers = _silicon()
    refinement = _refinement(asu_plan, spec, numbers)
    previous = Plan(structure_factor_grid=grid, orientations=(_orientation(grid, _FULL_BEAMS),))
    current = Plan(structure_factor_grid=grid, orientations=(_orientation(grid, _PRUNED_BEAMS),))

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
    one = Plan(structure_factor_grid=grid, orientations=(_orientation(grid, _FULL_BEAMS),))
    two = Plan(structure_factor_grid=grid, orientations=(_orientation(grid, _FULL_BEAMS),) * 2)

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


def test_converge_scalar_returns_first_below_threshold_step() -> None:
    # The sweep stops the first time the consecutive-build R-factor drops below threshold. Here
    # that is the second click (1.0 + 2 * 0.1).
    tolerance = ConvergenceTolerance(r_factor_threshold=0.005, max_iterations=10)
    measure = _scripted_measure([0.02, 0.003, 0.002])
    result = converge_scalar(lambda v: v, measure, tolerance, start=1.0, step=0.1)
    assert result == pytest.approx(1.2)


def test_converge_scalar_treats_an_unchanged_build_as_converged() -> None:
    # No skip-null handling: an unchanged build gives R = 0, which is below threshold, so the very
    # first click declares convergence -- the discrete-plateau sensitivity is managed by choosing a
    # coarse enough step, not by second-guessing the stop.
    tolerance = ConvergenceTolerance(r_factor_threshold=0.005, max_iterations=10)
    measure = _scripted_measure([0.0])
    result = converge_scalar(lambda v: v, measure, tolerance, start=1.0, step=0.1)
    assert result == pytest.approx(1.1)


def test_converge_scalar_raises_when_it_never_settles() -> None:
    tolerance = ConvergenceTolerance(r_factor_threshold=0.005, max_iterations=3)
    measure = _scripted_measure([0.02, 0.02, 0.02])
    with pytest.raises(RuntimeError, match="did not converge within 3 steps"):
        converge_scalar(lambda v: v, measure, tolerance, start=1.0, step=0.1)


# --- converge_beams: the beam-window adapter, exercised end-to-end on a richer synthetic seed ---
# The shared seed system lives in tests.unit.synthetic: a seed Plan rich enough (full hkl cube)
# that widening a beam knob admits beams progressively then saturates.


def test_converge_beams_widens_the_window_until_the_pattern_saturates() -> None:
    refinement, seed = seed_system()
    # Step wide enough to cross the intermediate count plateaus (the seed admits beams in bands as
    # the window widens: 15 -> 39 -> 43); the sweep then saturates and the trailing nulls settle it
    # on the fully-selected beam set.
    selection = BeamSelection(rsg=0.9, integration=IntegrationGeometry(semiangle=0.68))
    step = converge_beams(
        selection,
        refinement,
        ConvergenceTolerance(r_factor_threshold=0.05, max_iterations=20),
        step=0.6,
    )
    converged = step(seed)

    # It widened past the starting selection, all the way to the fully-reachable (saturated) set.
    started = select_beams(selection)(seed)
    saturated = select_beams(
        BeamSelection(rsg=selection.rsg, integration=IntegrationGeometry(semiangle=5.0))
    )(seed)
    assert beam_count(started) < beam_count(converged) == beam_count(saturated)


def test_converge_beams_fine_step_stops_early_at_an_intermediate_plateau() -> None:
    refinement, seed = seed_system()
    # Faithful first-dip stop is step-sensitive: too fine a step lets the consecutive-sim R-factor
    # dip below threshold on an intermediate count plateau, stopping short of the saturated set.
    # This is the documented discrete-plateau sensitivity (managed by step choice, not patience).
    selection = BeamSelection(rsg=0.9, integration=IntegrationGeometry(semiangle=0.68))
    step = converge_beams(
        selection,
        refinement,
        ConvergenceTolerance(r_factor_threshold=0.005, max_iterations=20),
        step=0.06,
    )
    converged = step(seed)

    started = select_beams(selection)(seed)
    saturated = select_beams(
        BeamSelection(rsg=selection.rsg, integration=IntegrationGeometry(semiangle=5.0))
    )(seed)
    assert beam_count(started) < beam_count(converged) < beam_count(saturated)


def test_converge_beams_rejects_non_positive_step() -> None:
    refinement, seed = seed_system()
    with pytest.raises(ValueError, match="step must be positive"):
        converge_beams(BeamSelection(), refinement, ConvergenceTolerance(), step=0.0)


# --- converge_sampling: the rocking-curve tilt-count (rocking_curve_sampling) lever ---


def test_converge_sampling_refines_tilts_until_the_integral_settles() -> None:
    refinement, seed = seed_system()
    seed = build_orientation_plans()(seed)  # rocking-curve sampling refines a built plan
    # Refine the tilt count: the summed |psi|^2 approaches the continuous rotation-frame integral,
    # so the consecutive-simulation change shrinks monotonically and settles below threshold. Each
    # orientation carries one beam plan per tilt, so len(beam_plans) is the converged tilt count.
    step = converge_sampling(
        RockingCurve(sampling=1, integration=IntegrationGeometry(semiangle=0.5)),
        refinement,
        ConvergenceTolerance(r_factor_threshold=0.01, max_iterations=30),
        step=2.0,
    )
    converged = step(seed)

    tilt_count = len(converged.orientations[0].beam_plans)
    assert tilt_count == 9  # deterministic: first consecutive-sim R below 0.01 is at 9 tilts
    # started from a single static solve (sampling == 1); convergence genuinely refined the grid
    assert (
        len(
            integrate_rocking_curve(
                RockingCurve(sampling=1, integration=IntegrationGeometry(semiangle=0.5))
            )(seed)
            .orientations[0]
            .beam_plans
        )
        == 1
    )


def test_converge_sampling_rejects_non_positive_step() -> None:
    refinement, seed = seed_system()
    with pytest.raises(ValueError, match="step must be positive"):
        converge_sampling(RockingCurve(), refinement, ConvergenceTolerance(), step=0.0)

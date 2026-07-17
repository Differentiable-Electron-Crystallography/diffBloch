"""Stateless refinement forward (``diffBloch.engine``): params -> simulated diffraction -> loss."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tests.unit.synthetic import make_constraint_spec

from diffBloch.core.losses import mse
from diffBloch.core.products import BlochSolution, PatternBatch
from diffBloch.core.symmetry import build_asu_expansion_plan
from diffBloch.engine import (
    LossFn,
    ObjectiveComponent,
    ObjectiveValue,
    OrientationPlan,
    RefinementEngine,
    RefinementProblem,
    ScatteringGrid,
    mse_loss,
    run_refinement_problem,
)
from diffBloch.observability import RecordingLogger, RefinementCompleted, RefinementStep
from diffBloch.params import RefinableParams

_ENERGY = 200e3
_CELL = np.eye(3, dtype=np.float64) * 5.0  # 5 A cubic -> reciprocal basis (1/5) I
_BEAM_HKL = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.int64)


def _engine(
    loss: LossFn = mse_loss,
    pattern: PatternBatch | None = None,
    tilts: np.ndarray | None = None,
) -> RefinementEngine:
    grid = ScatteringGrid.from_cell(_CELL, g_max=0.45)  # spans the beam differences (h up to +-2)
    asu_plan = build_asu_expansion_plan(
        np.zeros((1, 3)),
        np.eye(3)[None],
        np.zeros((1, 3)),  # P1: single atom, identity symop
    )
    if pattern is None:
        pattern = PatternBatch(
            hkl=torch.tensor(_BEAM_HKL, dtype=torch.int64),
            intensities=torch.tensor([0.9, 0.05, 0.05], dtype=torch.float64),
            sigmas=torch.full((3,), 0.01, dtype=torch.float64),
        )
    orientation = OrientationPlan.build(
        grid, _BEAM_HKL, pattern, energy=_ENERGY, thickness=(300.0,), tilts=tilts
    )  # 300 A: dynamical regime (I_diff ~0.1)
    spec = make_constraint_spec(reciprocal_basis=grid.reciprocal_basis)
    return RefinementEngine(
        spec=spec,
        asu_plan=asu_plan,
        numbers=torch.tensor([14], dtype=torch.int64),  # silicon
        grid=grid,
        orientations=(orientation,),
        loss=loss,
    )


def _params(
    *, requires_grad: bool = False, u_iso_scale: float = 0.1, occupancy_logit: float | None = None
) -> RefinableParams:
    uij = torch.eye(3, dtype=torch.float64)[None] * u_iso_scale
    fields = {
        "asu_positions": torch.zeros((1, 3), dtype=torch.float64, requires_grad=requires_grad),
        "uij_raw": uij.requires_grad_(requires_grad),
    }
    if occupancy_logit is not None:
        occ = torch.full((1,), occupancy_logit, dtype=torch.float64)
        fields["occupancy_raw"] = occ.requires_grad_(requires_grad)
    return RefinableParams(**fields)


def _observed_pattern(true_params: RefinableParams, *, sigma: float = 0.01) -> PatternBatch:
    """Self-consistent observations: the intensities the engine produces at ``true_params``."""
    dummy = PatternBatch(
        hkl=torch.tensor(_BEAM_HKL, dtype=torch.int64),
        intensities=torch.zeros(3, dtype=torch.float64),
        sigmas=torch.ones(3, dtype=torch.float64),
    )
    (solution,) = _engine(pattern=dummy).simulate(true_params)
    return PatternBatch(
        hkl=solution.beam_hkl,
        intensities=solution.intensities[0].detach(),
        sigmas=torch.full((3,), sigma, dtype=torch.float64),
    )


def test_simulate_returns_a_solution_per_orientation() -> None:
    engine = _engine()
    solutions = engine.simulate(_params())

    assert len(solutions) == len(engine.orientations)
    solution = solutions[0]
    assert isinstance(solution, BlochSolution)
    assert solution.intensities.shape == (1, _BEAM_HKL.shape[0])  # (T=1, N=3)
    # matrix_exp is unitary on this Hermitian system -> incident flux conserved
    assert torch.allclose(solution.intensities.sum(dim=1), torch.ones(1, dtype=torch.float64))


def test_simulate_sums_intensities_over_rocking_curve_tilts() -> None:
    from diffBloch.preprocess.orientation import rocking_curve_tilts

    params = _params()
    tilts = rocking_curve_tilts(0.5, 3)  # 3 tilts about x spanning +/- 0.5 deg (middle = angle 0)
    (integrated,) = _engine(tilts=tilts).simulate(params)
    # _solve sums |psi|^2 over the tilts: integrated intensity == the sum of the per-tilt solves.
    per_tilt = [_engine(tilts=tilts[i : i + 1]).simulate(params)[0].intensities for i in range(3)]
    assert torch.allclose(integrated.intensities, per_tilt[0] + per_tilt[1] + per_tilt[2])
    # incoherent: the stored amplitude is the real effective sqrt(total intensity).
    assert torch.allclose(integrated.amplitudes.abs().square(), integrated.intensities)
    # a genuine rocking curve differs from a single static solve.
    (static,) = _engine().simulate(params)
    assert not torch.allclose(integrated.intensities, static.intensities)


def test_objective_returns_scalar() -> None:
    loss = _engine().objective(_params())
    assert loss.shape == ()
    assert torch.isfinite(loss) and loss >= 0.0


def test_objective_value_names_the_diffraction_component() -> None:
    objective = _engine().objective_value(_params())
    assert objective.total.shape == ()
    assert set(objective.components) == {"diffraction"}
    diffraction = objective.components["diffraction"]
    assert diffraction.weight == 1.0
    assert torch.equal(diffraction.raw, diffraction.contribution)
    assert torch.equal(objective.total, sum(c.contribution for c in objective.components.values()))
    assert torch.equal(objective.total, _engine().objective(_params()))


def test_refinable_thickness_drives_the_forward_model() -> None:
    # The "thickness" refine target maps to thickness_raw; the forward model must consume it (not
    # silently use each orientation's frozen thickness). orientation seed = 300 A.
    engine = _engine()
    base = _params()
    base_params = RefinableParams(asu_positions=base.asu_positions, uij_raw=base.uij_raw)
    raw_500 = torch.log(torch.expm1(torch.tensor([500.0], dtype=torch.float64)))  # softplus^-1
    raw_300 = torch.log(torch.expm1(torch.tensor([300.0], dtype=torch.float64)))

    (no_thickness,) = engine.simulate(base_params)
    (thick_500,) = engine.simulate(
        RefinableParams(
            asu_positions=base.asu_positions, uij_raw=base.uij_raw, thickness_raw=raw_500
        )
    )
    (thick_300,) = engine.simulate(
        RefinableParams(
            asu_positions=base.asu_positions, uij_raw=base.uij_raw, thickness_raw=raw_300
        )
    )

    # Refinable thickness is honoured (carried onto the solution) and changes the intensities...
    assert torch.allclose(thick_500.thicknesses, torch.tensor([500.0], dtype=torch.float64))
    assert not torch.allclose(thick_500.intensities, no_thickness.intensities)
    # ...and equals the orientation's frozen thickness exactly at 300 A (proving the wiring).
    assert torch.allclose(thick_300.intensities, no_thickness.intensities)


def test_refinement_problem_is_pure_data() -> None:
    params = _params()
    problem = RefinementProblem(initial=params)

    assert problem.initial is params
    assert not hasattr(problem, "engine")
    assert not hasattr(problem, "refine")


def test_refinement_problem_can_run_current_refinement_loop_with_engine() -> None:
    observed = _observed_pattern(_params(occupancy_logit=2.2))
    engine = _engine(loss=mse_loss, pattern=observed)
    problem = RefinementProblem(initial=_params(occupancy_logit=0.0))

    result = run_refinement_problem(
        engine, problem, steps=6, targets=("occupancy",), optimizer="adam", lr=0.2
    )

    assert result.losses.shape == (6,)
    assert result.losses[-1] < result.losses[0]


def test_objective_value_computes_weighted_total_from_components() -> None:
    value = ObjectiveValue(
        {
            "diffraction": ObjectiveComponent(torch.tensor(2.0, dtype=torch.float64)),
            "bond": ObjectiveComponent(torch.tensor(3.0, dtype=torch.float64), weight=0.25),
        }
    )

    assert torch.equal(value.total, torch.tensor(2.75, dtype=torch.float64))
    with pytest.raises(TypeError):
        value.components["angle"] = ObjectiveComponent(torch.tensor(1.0))  # type: ignore[index]


def test_objective_is_differentiable_through_the_whole_chain() -> None:
    engine = _engine()
    params = _params(requires_grad=True)

    engine.objective(params).backward()

    # gradient flows back through align -> intensity -> propagate -> A -> Fgb -> expand -> constrain
    for grad in (params.asu_positions.grad, params.uij_raw.grad):
        assert grad is not None
        assert torch.isfinite(grad).all()
    # the 000-atom structure-factor depends on the ADP; positions at a fixed special site need not
    assert params.uij_raw.grad.abs().sum() > 0


def test_objective_co_locates_invariants_on_the_param_device() -> None:
    # On CPU this is a no-op, but it pins the contract: engine-owned invariants (numbers, grid_hkl,
    # reciprocal_basis, beam_hkl) and each orientation's thickness are moved to the params device at
    # the use site, so a simulated solution lands on the same device as the parameter-derived
    # tensors.
    engine = _engine()
    params = _params()
    (solution,) = engine.simulate(params)
    assert solution.intensities.device == params.asu_positions.device
    assert solution.beam_hkl.device == params.asu_positions.device


def test_objective_rejects_engine_without_orientations() -> None:
    engine = _engine()
    empty = RefinementEngine(
        spec=engine.spec,
        asu_plan=engine.asu_plan,
        numbers=engine.numbers,
        grid=engine.grid,
        orientations=(),
        loss=engine.loss,
    )
    with pytest.raises(ValueError, match="no orientations"):
        empty.objective(_params())
    with pytest.raises(ValueError, match="no orientations"):
        empty.simulate(_params())


def test_objective_rejects_non_scalar_loss() -> None:
    # A loss term that forgets to reduce to a scalar is caught at the engine, not later in backward.
    engine = _engine(loss=lambda aligned: mse(aligned.calculated, aligned.observed))
    with pytest.raises(ValueError, match="loss must return a scalar"):
        engine.objective(_params())


def test_scattering_grid_from_cell_spans_difference_support() -> None:
    grid = ScatteringGrid.from_cell(_CELL, g_max=0.45)
    # building beam plans validates the grid covers hkl_j - hkl_i; too-small g_max must raise.
    pattern = PatternBatch(
        hkl=torch.tensor(_BEAM_HKL, dtype=torch.int64),
        intensities=torch.zeros(3, dtype=torch.float64),
        sigmas=torch.ones(3, dtype=torch.float64),
    )
    OrientationPlan.build(grid, _BEAM_HKL, pattern, energy=_ENERGY, thickness=(300.0,))  # ok

    tiny = ScatteringGrid.from_cell(_CELL, g_max=0.15)  # |g|<=0.15 -> only h=0, no differences
    with pytest.raises(ValueError, match="difference support|gpts is too small"):
        OrientationPlan.build(tiny, _BEAM_HKL, pattern, energy=_ENERGY, thickness=(300.0,))


@pytest.mark.parametrize("optimizer", ["adam", "lbfgs"])
def test_refine_reduces_loss_toward_self_consistent_target(optimizer: str) -> None:
    # Observations are the engine's own output at occupancy ~0.9 (logit 2.2); start from 0.5.
    # Occupancy scales every F linearly -> strong, monotonic leverage on the diffracted intensity.
    true_params = _params(occupancy_logit=2.2)
    engine = _engine(loss=mse_loss, pattern=_observed_pattern(true_params))
    start = _params(occupancy_logit=0.0)

    result = engine.refine(start, steps=20, targets=("occupancy",), optimizer=optimizer, lr=0.2)

    assert result.losses.shape == (20,)
    assert result.losses[-1] < result.losses[0]  # the loop made progress
    assert result.best_loss <= float(result.losses[0])


def test_refine_does_not_mutate_caller_params() -> None:
    engine = _engine(loss=mse_loss)
    start = _params(u_iso_scale=0.05)
    before = start.uij_raw.detach().clone()

    engine.refine(start, steps=5, targets=("adp",), optimizer="adam", lr=0.05)

    # functional contract: the caller's tensors are untouched and gradient-free
    assert torch.equal(start.uij_raw, before)
    assert not start.uij_raw.requires_grad


def test_refine_best_params_track_the_lowest_recorded_loss() -> None:
    observed = _observed_pattern(_params(occupancy_logit=2.2))
    engine = _engine(loss=mse_loss, pattern=observed)
    result = engine.refine(
        _params(occupancy_logit=0.0), steps=12, targets=("occupancy",), optimizer="adam", lr=0.2
    )

    assert 0 <= result.best_step < 12
    assert result.best_loss == float(result.losses.min())
    assert result.best_params.occupancy_raw.shape == (1,)
    assert not result.best_params.occupancy_raw.requires_grad


def test_refine_emits_a_step_stream_and_a_completion_event() -> None:
    engine = _engine(loss=mse_loss, pattern=_observed_pattern(_params(occupancy_logit=2.2)))
    recorder = RecordingLogger()

    result = engine.refine(
        _params(occupancy_logit=0.0),
        steps=6,
        targets=("occupancy",),
        optimizer="adam",
        lr=0.2,
        logger=recorder,
    )

    steps = [e for e in recorder.events if isinstance(e, RefinementStep)]
    (completed,) = [e for e in recorder.events if isinstance(e, RefinementCompleted)]
    assert [e.iteration for e in steps] == [0, 1, 2, 3, 4, 5]  # one event per step, in order
    assert [e.loss for e in steps] == [float(x) for x in result.losses]  # the reported curve
    assert completed.n_steps == 6
    assert completed.best_step == result.best_step
    assert completed.best_loss == result.best_loss


def test_refine_only_selected_targets_change() -> None:
    # With only "adp" selected, positions must be carried through as an untouched constant.
    engine = _engine(loss=mse_loss, pattern=_observed_pattern(_params(u_iso_scale=0.15)))
    start = RefinableParams(
        asu_positions=torch.full((1, 3), 0.1, dtype=torch.float64),
        uij_raw=torch.eye(3, dtype=torch.float64)[None] * 0.05,
    )
    result = engine.refine(start, steps=8, targets=("adp",), optimizer="adam", lr=0.05)

    assert torch.equal(result.params.asu_positions, start.asu_positions)
    assert not torch.equal(result.params.uij_raw, start.uij_raw)


def test_refine_rejects_unknown_target() -> None:
    with pytest.raises(ValueError, match="unknown refinement target"):
        _engine().refine(_params(), steps=1, targets=("spin",))


def test_refine_rejects_target_with_no_parameter() -> None:
    # occupancy lives on the spec here, not as a refinable occupancy_raw -> nothing to optimize.
    with pytest.raises(ValueError, match="no matching parameter"):
        _engine().refine(_params(), steps=1, targets=("occupancy",))


def test_refine_rejects_non_positive_steps() -> None:
    with pytest.raises(ValueError, match="steps must be"):
        _engine().refine(_params(), steps=0, targets=("adp",))


def test_refine_rejects_unknown_optimizer() -> None:
    with pytest.raises(ValueError, match="optimizer must be"):
        _engine().refine(_params(), steps=1, targets=("adp",), optimizer="sgd")

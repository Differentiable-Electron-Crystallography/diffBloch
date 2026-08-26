"""Stateless refinement forward (``diffBloch.engine``): params -> simulated diffraction -> loss."""

from __future__ import annotations

import dataclasses
from typing import cast

import numpy as np
import pytest
import torch
from tests.unit.synthetic import make_constraint_spec

from diffBloch.core.losses import mse
from diffBloch.core.products import BlochSolution, PatternBatch
from diffBloch.core.symmetry import build_asu_expansion_plan
from diffBloch.engine import (
    AtomSelection,
    LossFn,
    ObjectiveComponent,
    ObjectiveValue,
    OptimizerName,
    OrientationPlan,
    RefinementEngine,
    RefinementProblem,
    RefinementResult,
    StructureFactorGrid,
    TrainableSpec,
    build_refinement_model,
    mse_loss,
    run_refinement_model,
    wr2_loss,
)
from diffBloch.observability import (
    NULL_LOGGER,
    Logger,
    ObjectiveManifest,
    ObjectiveTerm,
    RecordingLogger,
    RefinementCompleted,
    RefinementOrientationStep,
    RefinementStep,
)
from diffBloch.params import PhysicalState, RefinableParams

_ENERGY = 200e3
_CELL = np.eye(3, dtype=np.float64) * 5.0  # 5 A cubic -> reciprocal basis (1/5) I
_BEAM_HKL = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.int64)


def _engine(
    loss: LossFn = mse_loss,
    pattern: PatternBatch | None = None,
    tilts: np.ndarray | None = None,
    asu_positions: np.ndarray | None = None,
    numbers: torch.Tensor | None = None,
) -> RefinementEngine:
    grid = StructureFactorGrid.from_cell(
        _CELL, g_max=0.45
    )  # spans the beam differences (h up to +-2)
    if asu_positions is None:
        asu_positions = np.zeros((1, 3))
    asu_plan = build_asu_expansion_plan(
        asu_positions,
        np.eye(3)[None],
        np.zeros((1, 3)),  # P1: identity symop
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
    spec = make_constraint_spec(reciprocal_basis=grid.reciprocal_basis, n_atoms=len(asu_positions))
    return RefinementEngine(
        spec=spec,
        asu_plan=asu_plan,
        numbers=torch.tensor([14], dtype=torch.int64) if numbers is None else numbers,
        grid=grid,
        orientations=(orientation,),
        loss=loss,
    )


def _params(
    *,
    requires_grad: bool = False,
    u_iso_scale: float = 0.1,
    occupancy_logit: float | None = None,
    asu_positions: torch.Tensor | None = None,
) -> RefinableParams:
    if asu_positions is None:
        asu_positions = torch.zeros((1, 3), dtype=torch.float64)
    uij = torch.eye(3, dtype=torch.float64).expand(asu_positions.shape[0], 3, 3) * u_iso_scale
    fields = {
        "asu_positions": asu_positions.detach().clone().requires_grad_(requires_grad),
        "uij_raw": uij.requires_grad_(requires_grad),
    }
    if occupancy_logit is not None:
        occ = torch.full((1,), occupancy_logit, dtype=torch.float64)
        fields["occupancy_raw"] = occ.requires_grad_(requires_grad)
    return RefinableParams(**fields)


def _with_pattern(engine: RefinementEngine, pattern: PatternBatch) -> RefinementEngine:
    """Return ``engine`` with the same context but a replacement observed pattern."""
    orientation = OrientationPlan.build(
        engine.grid,
        _BEAM_HKL,
        pattern,
        energy=_ENERGY,
        thickness=(300.0,),
    )
    return dataclasses.replace(engine, orientations=(orientation,))


def _observed_pattern(
    true_params: RefinableParams,
    *,
    sigma: float = 0.01,
    engine: RefinementEngine | None = None,
) -> PatternBatch:
    """Self-consistent observations: the intensities the engine produces at ``true_params``."""
    dummy = PatternBatch(
        hkl=torch.tensor(_BEAM_HKL, dtype=torch.int64),
        intensities=torch.zeros(3, dtype=torch.float64),
        sigmas=torch.ones(3, dtype=torch.float64),
    )
    engine = _engine(pattern=dummy) if engine is None else _with_pattern(engine, dummy)
    (solution,) = engine.simulate(true_params)
    return PatternBatch(
        hkl=solution.beam_hkl,
        intensities=solution.intensities[0].detach(),
        sigmas=torch.full((3,), sigma, dtype=torch.float64),
    )


def _refine(
    engine: RefinementEngine,
    initial: RefinableParams,
    *,
    steps: int,
    trainable: TrainableSpec | None = None,
    optimizer: OptimizerName | str = "lbfgs",
    lr: float = 1e-3,
    logger: Logger = NULL_LOGGER,
) -> RefinementResult:
    """Test helper for the explicit problem + executor API."""
    return run_refinement_model(
        engine,
        build_refinement_model(initial=initial),
        RefinementProblem(),
        trainable=trainable or TrainableSpec.positions_and_adp(),
        steps=steps,
        optimizer=cast(OptimizerName, optimizer),
        lr=lr,
        logger=logger,
    )


def test_simulate_returns_a_solution_per_orientation() -> None:
    engine = _engine()
    solutions = engine.simulate(_params())

    assert len(solutions) == len(engine.orientations)
    solution = solutions[0]
    assert isinstance(solution, BlochSolution)
    assert solution.intensities.shape == (1, _BEAM_HKL.shape[0])  # (T=1, N=3)
    # matrix_exp is unitary on this Hermitian system -> incident flux conserved
    assert torch.allclose(
        solution.intensities.sum(dim=1), torch.ones(1, dtype=solution.intensities.dtype), atol=1e-5
    )


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


def test_objective_value_returns_scalar_total() -> None:
    loss = _engine().objective_value(_params()).total
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


def test_atom_selection_rejects_invalid_runtime_mode() -> None:
    with pytest.raises(ValueError, match="atom selection mode"):
        AtomSelection("bogus")  # type: ignore[arg-type]


def test_atom_selection_rejects_unknown_element_symbol() -> None:
    with pytest.raises(ValueError, match="unknown element symbol"):
        AtomSelection.exclude_elements("not-an-element")


def test_refinement_problem_is_pure_data() -> None:
    params = _params()
    model = build_refinement_model(initial=params)
    problem = RefinementProblem()

    assert model.structure.initial is params
    assert problem.penalties == ()
    assert not hasattr(problem, "engine")
    assert not hasattr(problem, "refine")


def test_refinement_problem_can_run_current_refinement_loop_with_engine() -> None:
    observed = _observed_pattern(_params(occupancy_logit=2.2))
    engine = _engine(loss=mse_loss, pattern=observed)
    model = build_refinement_model(initial=_params(occupancy_logit=0.0))
    problem = RefinementProblem()

    result = run_refinement_model(
        engine,
        model,
        problem,
        trainable=TrainableSpec(occupancy=AtomSelection.all()),
        steps=6,
        optimizer="adam",
        lr=0.2,
    )

    assert result.losses.shape == (6,)
    assert result.losses[-1] < result.losses[0]


def test_run_refinement_model_verbose_reports_per_rotation_steps() -> None:
    """``verbose`` ("verbose refinement") adds one per-rotation event per step, off by default."""
    engine = _engine(
        loss=wr2_loss
    )  # wr2 loss so RefinementStep.wr2 is populated to compare against
    model = build_refinement_model(initial=_params())
    logger = RecordingLogger()

    run_refinement_model(
        engine,
        model,
        RefinementProblem(),
        trainable=TrainableSpec.positions_and_adp(),
        steps=3,
        optimizer="adam",
        lr=1e-3,
        logger=logger,
        verbose=True,
    )

    epoch_events = [e for e in logger.events if isinstance(e, RefinementStep)]
    orientation_events = [e for e in logger.events if isinstance(e, RefinementOrientationStep)]
    assert len(epoch_events) == 3
    # one engine orientation -> one RefinementOrientationStep per step, matching the epoch mean.
    assert len(orientation_events) == 3
    for epoch, orientation_event in zip(epoch_events, orientation_events, strict=True):
        assert orientation_event.iteration == epoch.iteration
        assert orientation_event.rotation_index == 0
        assert orientation_event.wr2 == pytest.approx(epoch.wr2)
        assert orientation_event.r_obs == pytest.approx(epoch.r_obs)
        assert orientation_event.diff_loss == pytest.approx(epoch.diff_loss)


def test_run_refinement_model_declares_the_objective_before_the_first_step() -> None:
    """The manifest opens the stream and is returned on the result, empty penalties included."""

    @dataclasses.dataclass(frozen=True)
    class _Penalty:
        name: str = "bond_length"
        weight: float = 2.5

        def value(self, state: PhysicalState) -> torch.Tensor:
            return state.positions.new_zeros(())

    engine = _engine(loss=mse_loss)
    model = build_refinement_model(initial=_params())
    logger = RecordingLogger()

    result = run_refinement_model(
        engine,
        model,
        RefinementProblem(penalties=(_Penalty(),)),
        trainable=TrainableSpec.positions_and_adp(),
        steps=2,
        optimizer="adam",
        lr=1e-3,
        logger=logger,
    )

    (manifest,) = [e for e in logger.events if isinstance(e, ObjectiveManifest)]
    assert logger.events.index(manifest) == 0  # declared before any compute is reported
    assert manifest.penalties == (ObjectiveTerm(name="bond_length", weight=2.5),)
    assert manifest.constraints == () and manifest.components == ()
    assert result.objective_manifest == manifest


def test_run_refinement_model_default_omits_per_rotation_steps() -> None:
    engine = _engine(loss=mse_loss)
    model = build_refinement_model(initial=_params())
    logger = RecordingLogger()

    run_refinement_model(
        engine,
        model,
        RefinementProblem(),
        trainable=TrainableSpec.positions_and_adp(),
        steps=2,
        optimizer="adam",
        lr=1e-3,
        logger=logger,
    )

    assert not [e for e in logger.events if isinstance(e, RefinementOrientationStep)]


@dataclasses.dataclass(frozen=True)
class _DummyPenalty:
    name: str = "dummy_penalty"
    weight: float = 0.25

    def value(self, state: PhysicalState) -> torch.Tensor:
        return state.positions[:, 0].sum() + 2.0


def test_objective_value_composes_penalty_components() -> None:
    engine = _engine()
    params = _params(asu_positions=torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float64))

    baseline = engine.objective_value(params)
    penalized = engine.objective_value(params, penalties=(_DummyPenalty(),))
    component = penalized.components["dummy_penalty"]

    assert set(penalized.components) == {"diffraction", "dummy_penalty"}
    assert torch.equal(component.raw, torch.tensor(2.5, dtype=torch.float64))
    assert component.weight == 0.25
    assert torch.equal(
        penalized.total,
        baseline.total + component.contribution,
    )


def test_objective_value_penalty_contributes_gradient() -> None:
    engine = _engine()
    params = _params(requires_grad=True)

    engine.objective_value(params, penalties=(_DummyPenalty(),)).total.backward()

    assert params.asu_positions.grad is not None
    assert params.asu_positions.grad[0, 0] != 0.0


def test_objective_value_rejects_duplicate_component_name() -> None:
    engine = _engine()
    with pytest.raises(ValueError, match="duplicate objective component"):
        engine.objective_value(_params(), penalties=(_DummyPenalty(name="diffraction"),))


def test_refinement_problem_records_penalties() -> None:
    penalty = _DummyPenalty()
    problem = RefinementProblem(penalties=(penalty,))

    assert problem.penalties == (penalty,)


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

    engine.objective_value(params).total.backward()

    # gradient flows back through align -> intensity -> propagate -> A -> Fgb -> expand -> constrain
    for grad in (params.asu_positions.grad, params.uij_raw.grad):
        assert grad is not None
        assert torch.isfinite(grad).all()
    # the 000-atom structure-factor depends on the ADP; positions at a fixed special site need not
    assert params.uij_raw.grad.abs().sum() > 0


def test_objective_co_locates_invariants_on_the_param_device() -> None:
    # On CPU this is a no-op, but it pins the contract: engine-owned invariants (numbers, structure_factor_hkl,
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
        empty.objective_value(_params())
    with pytest.raises(ValueError, match="no orientations"):
        empty.simulate(_params())


def test_objective_rejects_non_scalar_loss() -> None:
    # A loss term that forgets to reduce to a scalar is caught at the engine, not later in backward.
    engine = _engine(loss=lambda aligned: mse(aligned.calculated, aligned.observed))
    with pytest.raises(ValueError, match="loss must return a scalar"):
        engine.objective_value(_params())


def test_scattering_grid_from_cell_spans_difference_support() -> None:
    grid = StructureFactorGrid.from_cell(_CELL, g_max=0.45)
    # building beam plans validates the grid covers hkl_j - hkl_i; too-small g_max must raise.
    pattern = PatternBatch(
        hkl=torch.tensor(_BEAM_HKL, dtype=torch.int64),
        intensities=torch.zeros(3, dtype=torch.float64),
        sigmas=torch.ones(3, dtype=torch.float64),
    )
    OrientationPlan.build(grid, _BEAM_HKL, pattern, energy=_ENERGY, thickness=(300.0,))  # ok

    tiny = StructureFactorGrid.from_cell(_CELL, g_max=0.15)  # |g|<=0.15 -> only h=0, no differences
    with pytest.raises(ValueError, match="difference support|gpts is too small"):
        OrientationPlan.build(tiny, _BEAM_HKL, pattern, energy=_ENERGY, thickness=(300.0,))


@pytest.mark.parametrize("optimizer", ["adam", "lbfgs"])
def test_refine_reduces_loss_toward_self_consistent_target(optimizer: str) -> None:
    # Observations are the engine's own output at occupancy ~0.9 (logit 2.2); start from 0.5.
    # Occupancy scales every F linearly -> strong, monotonic leverage on the diffracted intensity.
    true_params = _params(occupancy_logit=2.2)
    engine = _engine(loss=mse_loss, pattern=_observed_pattern(true_params))
    start = _params(occupancy_logit=0.0)

    result = _refine(
        engine,
        start,
        steps=20,
        trainable=TrainableSpec(occupancy=AtomSelection.all()),
        optimizer=optimizer,
        lr=0.2,
    )

    assert result.losses.shape == (20,)
    assert result.losses[-1] < result.losses[0]  # the loop made progress
    assert result.best_loss <= float(result.losses[0])


def test_refine_does_not_mutate_caller_params() -> None:
    engine = _engine(loss=mse_loss)
    start = _params(u_iso_scale=0.05)
    before = start.uij_raw.detach().clone()

    _refine(
        engine,
        start,
        steps=5,
        trainable=TrainableSpec(adp=AtomSelection.all()),
        optimizer="adam",
        lr=0.05,
    )

    # functional contract: the caller's tensors are untouched and gradient-free
    assert torch.equal(start.uij_raw, before)
    assert not start.uij_raw.requires_grad


def test_refine_best_params_track_the_lowest_recorded_loss() -> None:
    observed = _observed_pattern(_params(occupancy_logit=2.2))
    engine = _engine(loss=mse_loss, pattern=observed)
    result = _refine(
        engine,
        _params(occupancy_logit=0.0),
        steps=12,
        trainable=TrainableSpec(occupancy=AtomSelection.all()),
        optimizer="adam",
        lr=0.2,
    )

    assert result.selection_losses is None
    assert 0 <= result.best_step < 12
    assert result.best_loss == float(result.losses.min())
    assert result.best_params.occupancy_raw.shape == (1,)
    assert not result.best_params.occupancy_raw.requires_grad


def test_refine_best_params_can_track_a_selection_engine() -> None:
    train_engine = _engine(loss=mse_loss, pattern=_observed_pattern(_params(occupancy_logit=2.2)))
    selection_engine = _engine(
        loss=mse_loss,
        pattern=_observed_pattern(_params(occupancy_logit=0.0)),
    )
    initial = _params(occupancy_logit=0.0)
    recorder = RecordingLogger()

    result = run_refinement_model(
        train_engine,
        build_refinement_model(initial=initial),
        RefinementProblem(),
        trainable=TrainableSpec(occupancy=AtomSelection.all()),
        steps=8,
        optimizer="adam",
        lr=0.2,
        selection_engine=selection_engine,
        logger=recorder,
    )

    assert result.selection_losses is not None
    assert result.losses[-1] < result.losses[0]  # the training objective still improves
    assert result.best_step == int(torch.argmin(result.selection_losses))
    assert result.best_loss == float(result.selection_losses[result.best_step])
    assert result.best_step != int(torch.argmin(result.losses))
    assert torch.allclose(result.best_params.occupancy_raw, initial.occupancy_raw)

    # The completion event must name the objective that selected the epoch, and must not report a
    # held-out number under the training key -- the per-step stream stays the training objective.
    (completed,) = [e for e in recorder.events if isinstance(e, RefinementCompleted)]
    assert completed.selection == "validation"
    assert completed.measurements["best_validation_loss"] == result.best_loss
    assert "best_training_loss" not in completed.measurements
    steps_reported = [e.loss for e in recorder.events if isinstance(e, RefinementStep)]
    assert steps_reported == [float(x) for x in result.losses]


def test_refine_emits_a_step_stream_and_a_completion_event() -> None:
    engine = _engine(loss=mse_loss, pattern=_observed_pattern(_params(occupancy_logit=2.2)))
    recorder = RecordingLogger()

    result = _refine(
        engine,
        _params(occupancy_logit=0.0),
        steps=6,
        trainable=TrainableSpec(occupancy=AtomSelection.all()),
        optimizer="adam",
        lr=0.2,
        logger=recorder,
    )

    steps = [e for e in recorder.events if isinstance(e, RefinementStep)]
    (completed,) = [e for e in recorder.events if isinstance(e, RefinementCompleted)]
    assert [e.iteration for e in steps] == [0, 1, 2, 3, 4, 5]  # one event per step, in order
    assert [e.loss for e in steps] == [float(x) for x in result.losses]  # the reported curve
    first = steps[0]
    assert first.objective_total == first.loss
    assert first.components.keys() == {"diffraction"}
    # wr2/r_obs are always-computed reporting diagnostics, independent of the configured loss
    # (mse_loss here) -- both are real numbers and both appear in measurements.
    assert first.wr2 is not None
    assert first.r_obs is not None
    assert first.measurements == {
        "wr2": first.wr2,
        "r_obs": first.r_obs,
        "diff_loss": first.loss,
        # The sole composed term reports its raw value, weight, and weighted contribution.
        "diffraction/raw": first.loss,
        "diffraction/weight": 1.0,
        "diffraction/contribution": first.loss,
        # Each mean carries the denominator it was taken over.
        "n_rotations": 1.0,
        "n_wr2_evaluated": 1.0,
        "n_r_obs_evaluated": 1.0,
    }
    assert completed.n_steps == 6
    assert completed.best_step == result.best_step
    assert completed.best_loss == result.best_loss
    # No selection engine: the summary's best is the training objective, and says so.
    assert completed.selection == "training"
    assert "best_training_loss" in completed.measurements


def test_refine_lbfgs_step_diagnostics_match_reported_pre_update_loss() -> None:
    engine = _engine(loss=mse_loss, pattern=_observed_pattern(_params(occupancy_logit=2.2)))
    recorder = RecordingLogger()

    result = _refine(
        engine,
        _params(occupancy_logit=0.0),
        steps=1,
        trainable=TrainableSpec(occupancy=AtomSelection.all()),
        optimizer="lbfgs",
        lr=0.2,
        logger=recorder,
    )

    (step,) = [e for e in recorder.events if isinstance(e, RefinementStep)]
    assert step.loss == float(result.losses[0])
    assert step.objective_total == step.loss
    assert step.r_obs is not None  # always-computed reporting diagnostic, independent of loss
    assert step.measurements == {
        "wr2": step.wr2,
        "r_obs": step.r_obs,
        "diff_loss": step.loss,
        "diffraction/raw": step.loss,
        "diffraction/weight": 1.0,
        "diffraction/contribution": step.loss,
        "n_rotations": 1.0,
        "n_wr2_evaluated": 1.0,
        "n_r_obs_evaluated": 1.0,
    }


def test_refine_element_selection_freezes_excluded_position_rows() -> None:
    numbers = torch.tensor([6, 1], dtype=torch.int64)  # C, H
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]], dtype=torch.float64)
    true_positions = positions.clone()
    true_positions[0, 0] = 0.05
    engine = _engine(
        numbers=numbers,
        asu_positions=positions.numpy(),
        loss=mse_loss,
    )
    observed = _observed_pattern(
        _params(asu_positions=true_positions),
        engine=engine,
    )
    engine = _with_pattern(engine, observed)
    start = _params(asu_positions=positions)

    result = _refine(
        engine,
        start,
        steps=8,
        trainable=TrainableSpec(positions=AtomSelection.exclude_elements("H")),
        optimizer="adam",
        lr=0.02,
    )

    assert torch.equal(result.params.asu_positions[1], start.asu_positions[1])
    assert not torch.equal(result.params.asu_positions[0], start.asu_positions[0])


def test_refine_only_selected_targets_change() -> None:
    # With only "adp" selected, positions must be carried through as an untouched constant.
    engine = _engine(loss=mse_loss, pattern=_observed_pattern(_params(u_iso_scale=0.15)))
    start = RefinableParams(
        asu_positions=torch.full((1, 3), 0.1, dtype=torch.float64),
        uij_raw=torch.eye(3, dtype=torch.float64)[None] * 0.05,
    )
    result = _refine(
        engine,
        start,
        steps=8,
        trainable=TrainableSpec(adp=AtomSelection.all()),
        optimizer="adam",
        lr=0.05,
    )

    assert torch.equal(result.params.asu_positions, start.asu_positions)
    assert not torch.equal(result.params.uij_raw, start.uij_raw)


def test_refine_rejects_empty_trainable_spec() -> None:
    with pytest.raises(ValueError, match="at least one trainable parameter group"):
        _refine(_engine(), _params(), steps=1, trainable=TrainableSpec())


def test_refine_rejects_element_selection_matching_no_atoms() -> None:
    with pytest.raises(ValueError, match="matched no atoms"):
        _refine(
            _engine(),
            _params(),
            steps=1,
            trainable=TrainableSpec(positions=AtomSelection.exclude_elements("Si")),
        )


def test_refine_rejects_trainable_group_with_no_parameter() -> None:
    # occupancy lives on the spec here, not as a refinable occupancy_raw -> nothing to optimize.
    with pytest.raises(ValueError, match="no matching parameter"):
        _refine(
            _engine(),
            _params(),
            steps=1,
            trainable=TrainableSpec(occupancy=AtomSelection.all()),
        )


def test_refine_rejects_non_positive_steps() -> None:
    with pytest.raises(ValueError, match="steps must be"):
        _refine(_engine(), _params(), steps=0, trainable=TrainableSpec(adp=AtomSelection.all()))


def test_refine_rejects_unknown_optimizer() -> None:
    with pytest.raises(ValueError, match="optimizer must be"):
        _refine(
            _engine(),
            _params(),
            steps=1,
            trainable=TrainableSpec(adp=AtomSelection.all()),
            optimizer="sgd",
        )


def _rotation_observing(
    engine: RefinementEngine, hkl: list[list[int]], intensities: list[float]
) -> OrientationPlan:
    """One rotation of ``engine``'s system observing exactly ``hkl``.

    sigma is 0.01 throughout, so an intensity above 0.03 is "strong" under the I > 3 sigma split
    ``refinement_metrics`` applies.
    """
    pattern = PatternBatch(
        hkl=torch.tensor(hkl, dtype=torch.int64),
        intensities=torch.tensor(intensities, dtype=torch.float64),
        sigmas=torch.full((len(hkl),), 0.01, dtype=torch.float64),
    )
    return OrientationPlan.build(
        engine.grid, _BEAM_HKL, pattern, energy=_ENERGY, thickness=(300.0,)
    )


def test_refinement_metrics_counts_a_reflection_re_observed_across_rotations_once() -> None:
    """Counts are unique (h, k, l) triples, not raw PETS rows.

    Rotation electron diffraction frames overlap in angle, so the same reciprocal-lattice point is
    routinely re-observed in several consecutive rotations. Summing per-rotation row counts inflates
    every total by exactly those overlaps -- here 100 is seen in both rotations, so a row sum would
    report 3 matched where there are only 2 distinct reflections.
    """
    base = _engine()
    engine = dataclasses.replace(
        base,
        orientations=(
            _rotation_observing(base, [[0, 0, 0], [1, 0, 0]], [0.9, 0.05]),
            _rotation_observing(base, [[1, 0, 0]], [0.05]),
        ),
    )

    _, n_matched, _, _, _ = engine.refinement_metrics(build_refinement_model(initial=_params()))

    assert n_matched == 2  # {000, 100}; the raw row sum would be 3


def test_refinement_metrics_splits_strong_and_weak_over_the_same_unique_set() -> None:
    """strong + weak must equal matched, with each triple landing in exactly one bucket.

    A triple is strong if *any* of its matched measurements clears I > 3 sigma, so 100 -- weak in
    the first rotation and strong in the second -- is strong once, never counted in both buckets.
    """
    base = _engine()
    engine = dataclasses.replace(
        base,
        orientations=(
            _rotation_observing(base, [[0, 0, 0], [1, 0, 0]], [0.9, 0.01]),
            _rotation_observing(base, [[1, 0, 0], [-1, 0, 0]], [0.9, 0.01]),
        ),
    )

    _, n_matched, n_strong, n_weak, _ = engine.refinement_metrics(
        build_refinement_model(initial=_params())
    )

    assert n_matched == 3  # {000, 100, -100}
    assert n_strong == 2  # 000 always strong; 100 strong via the second rotation
    assert n_weak == 1  # -100 only ever weak
    assert n_strong + n_weak == n_matched


def test_refinement_metrics_counts_an_unmatched_reflection_once_not_per_rotation() -> None:
    """Observed-but-never-matched triples deduplicate the same way the matched ones do.

    200 is outside the solve beam set, so it never enters the alignment. Observed in both rotations,
    it is one unmatched reflection, not two.
    """
    base = _engine()
    engine = dataclasses.replace(
        base,
        orientations=(
            _rotation_observing(base, [[0, 0, 0], [2, 0, 0]], [0.9, 0.05]),
            _rotation_observing(base, [[0, 0, 0], [2, 0, 0]], [0.9, 0.05]),
        ),
    )

    _, n_matched, _, _, n_unmatched = engine.refinement_metrics(
        build_refinement_model(initial=_params())
    )

    assert n_matched == 1  # only 000 is in the beam set
    assert n_unmatched == 1  # 200, once -- not once per rotation

"""Slice 11 (5a): the wR2-scoring seam (``optimal_scale``, ``score_orientation``, ``build_engine``).

Pure scaling on hand-checkable intensities, then the per-orientation scaling-optimised wR2 on a fast
synthetic silicon system (mirroring ``test_engine``'s setup so no heavy fixture sim is needed).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from tests.unit.synthetic import make_constraint_spec

from diffBloch.core.losses import optimal_scale, w_rbragg
from diffBloch.core.products import PatternBatch
from diffBloch.core.symmetry import build_asu_expansion_plan
from diffBloch.engine import (
    OrientationPlan,
    RefinementEngine,
    ScatteringGrid,
    SegmentedOrientationPlan,
    w_rbragg_loss,
)
from diffBloch.params import ConstraintSpec, RefinableParams
from diffBloch.preprocess import (
    RefinementSetup,
    build_engine,
    run_inference,
    score_orientations,
)
from diffBloch.preprocess.orientation import rocking_curve_tilts
from diffBloch.preprocess.plan import Plan

_ENERGY = 200e3
_CELL = np.eye(3, dtype=np.float64) * 5.0
_BEAM_HKL = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.int64)


# --- optimal_scale (pure) -------------------------------------------------------------------------


def test_optimal_scale_recovers_a_known_constant_factor() -> None:
    observed = torch.tensor([0.9, 0.05, 0.05], dtype=torch.float64)
    sigmas = torch.full((3,), 0.01, dtype=torch.float64)

    # calculated = 2 x observed -> the absolute scale that recovers observed is 0.5, wR2 -> 0.
    scale, value = optimal_scale(2 * observed, observed, sigmas)
    assert abs(float(scale) - 0.5) < 0.02
    assert float(value) < 1e-6
    assert torch.allclose(2 * observed * scale, observed, atol=1e-3)


def test_optimal_scale_returns_the_grid_minimum() -> None:
    observed = torch.tensor([1.0, 0.4, 0.2, 0.1], dtype=torch.float64)
    sigmas = torch.full((4,), 0.02, dtype=torch.float64)
    calculated = torch.tensor([0.7, 0.5, 0.25, 0.05], dtype=torch.float64)

    scale, value = optimal_scale(calculated, observed, sigmas)
    # No other absolute scale on the search grid beats the returned one.
    ratio = float(observed.sum() / calculated.sum())
    for factor in np.linspace(0.02, 2.0, 100):
        other = w_rbragg(factor * ratio * calculated, observed, sigmas)
        assert float(value) <= float(other) + 1e-12


# --- score_orientation + seam (synthetic silicon) -------------------------------------------------


def _silicon() -> tuple[ScatteringGrid, object, ConstraintSpec, torch.Tensor]:
    grid = ScatteringGrid.from_cell(_CELL, g_max=0.45)
    asu_plan = build_asu_expansion_plan(np.zeros((1, 3)), np.eye(3)[None], np.zeros((1, 3)))
    spec = make_constraint_spec(reciprocal_basis=grid.reciprocal_basis)
    numbers = torch.tensor([14], dtype=torch.int64)
    return grid, asu_plan, spec, numbers


def _params() -> RefinableParams:
    return RefinableParams(
        asu_positions=torch.zeros((1, 3), dtype=torch.float64),
        uij_raw=torch.eye(3, dtype=torch.float64)[None] * 0.1,
    )


def _engine(
    grid: ScatteringGrid,
    asu_plan: object,
    spec: ConstraintSpec,
    numbers: torch.Tensor,
    orientation: OrientationPlan,
) -> RefinementEngine:
    return RefinementEngine(
        spec=spec,
        asu_plan=asu_plan,  # type: ignore[arg-type]
        numbers=numbers,
        grid=grid,
        orientations=(orientation,),
        loss=w_rbragg_loss,
    )


def _self_consistent_orientation(
    grid: ScatteringGrid, asu_plan: object, spec: ConstraintSpec, numbers: torch.Tensor
) -> tuple[OrientationPlan, torch.Tensor]:
    """An orientation whose observed pattern is what the engine simulates at ``_params()``."""
    dummy = PatternBatch(
        hkl=torch.tensor(_BEAM_HKL, dtype=torch.int64),
        intensities=torch.zeros(3, dtype=torch.float64),
        sigmas=torch.ones(3, dtype=torch.float64),
    )
    seed = OrientationPlan.build(grid, _BEAM_HKL, dummy, energy=_ENERGY, thickness=(300.0,))
    (solution,) = _engine(grid, asu_plan, spec, numbers, seed).simulate(_params())
    observed = PatternBatch(
        hkl=solution.beam_hkl,
        intensities=solution.intensities[0].detach(),
        sigmas=torch.full((3,), 0.01, dtype=torch.float64),
    )
    return (
        OrientationPlan.build(grid, _BEAM_HKL, observed, energy=_ENERGY, thickness=(300.0,)),
        solution.intensities[0],
    )


def test_score_orientation_vanishes_at_a_self_consistent_pattern() -> None:
    grid, asu_plan, spec, numbers = _silicon()
    orientation, _ = _self_consistent_orientation(grid, asu_plan, spec, numbers)
    engine = _engine(grid, asu_plan, spec, numbers, orientation)

    score = engine.score_orientation(orientation, engine.fgb(_params()))
    assert score.shape == ()
    assert torch.isfinite(score) and float(score) < 1e-4


def test_score_orientation_penalises_a_mismatched_pattern() -> None:
    grid, asu_plan, spec, numbers = _silicon()
    matched, intensities = _self_consistent_orientation(grid, asu_plan, spec, numbers)
    # Perturb the observed pattern: a worse fit must score strictly higher than the matched one.
    perturbed_pattern = PatternBatch(
        hkl=matched.pattern.hkl,
        intensities=(intensities.detach().flip(0) + 0.05),
        sigmas=torch.full((3,), 0.01, dtype=torch.float64),
    )
    perturbed = OrientationPlan.build(
        grid, _BEAM_HKL, perturbed_pattern, energy=_ENERGY, thickness=(300.0,)
    )
    engine = _engine(grid, asu_plan, spec, numbers, matched)
    fgb = engine.fgb(_params())

    assert float(engine.score_orientation(perturbed, fgb)) > float(
        engine.score_orientation(matched, fgb)
    )


def test_build_engine_wires_plan_geometry_and_structure_context() -> None:
    grid, asu_plan, spec, numbers = _silicon()
    orientation, _ = _self_consistent_orientation(grid, asu_plan, spec, numbers)
    plan = Plan(grid=grid, orientations=(orientation,))
    refinement = RefinementSetup(
        asu_plan=asu_plan,  # type: ignore[arg-type]
        spec=spec,
        params=_params(),
        numbers=numbers,
    )

    engine = build_engine(plan, refinement)
    assert engine.grid is plan.grid
    assert engine.orientations is plan.orientations
    assert engine.spec is refinement.spec
    assert engine.asu_plan is refinement.asu_plan
    assert engine.numbers is refinement.numbers
    assert engine.orientations[0].thickness.tolist() == [300.0]

    scores = score_orientations(plan, refinement)
    assert len(scores) == len(plan.orientations)
    assert all(torch.isfinite(s) and s.shape == () for s in scores)


def _silicon_plan() -> tuple[Plan, RefinementSetup]:
    grid, asu_plan, spec, numbers = _silicon()
    orientation, _ = _self_consistent_orientation(grid, asu_plan, spec, numbers)
    plan = Plan(grid=grid, orientations=(orientation,))
    refinement = RefinementSetup(
        asu_plan=asu_plan,  # type: ignore[arg-type]
        spec=spec,
        params=_params(),
        numbers=numbers,
    )
    return plan, refinement


def test_fp32_scores_end_to_end_through_build_engine() -> None:
    """The coarse knob works through the full build_engine -> simulate -> wR2 path, not just solver.

    Guards the feature's actual purpose: a ``precision="fp32"`` engine reaches the O(N^3) solve in
    complex64 (exit amplitudes complex64 -> intensities float32), and the cheap scoring tail still
    yields a finite score -- observed intensities/sigmas are float64, so type-promotion upcasts the
    reduction back to double (confining fp32 to the expensive solve). A stray complex128 constant
    mid-solve, or a promotion that errored instead, would trip this.
    """
    plan, refinement = _silicon_plan()
    engine = build_engine(plan, refinement, precision="fp32")
    assert engine.precision == "fp32"

    (solution,) = engine.simulate(refinement.params)
    assert solution.amplitudes.dtype == torch.complex64  # fp32 reached the solve
    assert solution.intensities.dtype == torch.float32

    score = engine.score_orientation(engine.orientations[0], engine.fgb(refinement.params))
    assert score.shape == () and torch.isfinite(score)


def test_terminal_stays_fp64_and_exposes_no_precision_knob() -> None:
    """The safety-critical invariant: the terminal Plan->Result never runs fp32.

    Two layers. (a) The default engine is fp64 -- the solve produces complex128 / float64, so the
    pinned result is exact. (b) Structural: ``run_inference`` exposes no ``precision`` parameter, so
    a coarse search cannot leak into the terminal by construction, not merely by callers omitting.
    """
    plan, refinement = _silicon_plan()
    engine = build_engine(plan, refinement)  # no precision -> fp64
    assert engine.precision == "fp64"

    (solution,) = engine.simulate(refinement.params)
    assert solution.amplitudes.dtype == torch.complex128
    assert solution.intensities.dtype == torch.float64


# --- device knob (place the forward solve on an accelerator; params.device is authoritative) ------


def test_refinable_params_to_moves_present_tensors_and_keeps_none() -> None:
    """``.to`` moves every present parameter tensor and leaves the un-refined (None) ones None."""
    moved = _params().to("cpu")  # _params has asu_positions + uij_raw; the rest are None
    assert moved.asu_positions.device == torch.device("cpu")
    assert moved.uij_raw is not None and moved.uij_raw.device == torch.device("cpu")
    assert moved.u_iso_raw is None and moved.thickness_raw is None and moved.occupancy_raw is None


def test_run_inference_device_cpu_is_a_no_op() -> None:
    """The device knob is a pure no-op on CPU: ``device='cpu'`` reproduces the default exactly."""
    plan, refinement = _silicon_plan()
    base = run_inference(plan, refinement)
    cpu = run_inference(plan, refinement, device="cpu")
    assert [r.r_obs for r in cpu.per_rotation] == [r.r_obs for r in base.per_rotation]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required (runs on the A100)")
def test_run_inference_cuda_matches_cpu_within_tolerance() -> None:
    """CPU<->GPU parity: on a CUDA box the on-device forward reproduces CPU R_obs to solver tol.

    Skipped locally (this Mac has no complex-capable GPU); the real assertion runs on the SSEC A100
    cluster, where it pins that placing params on ``"cuda"`` carries the whole eigensolve there and
    the terminal ``R_obs`` still matches the fp64 CPU reference.
    """
    plan, refinement = _silicon_plan()
    cpu = run_inference(plan, refinement, device="cpu")
    cuda = run_inference(plan, refinement, device="cuda")
    for c, g in zip(cpu.per_rotation, cuda.per_rotation, strict=True):
        assert abs(c.r_obs - g.r_obs) < 1e-6


# --- fp32 on the integrating solve paths (the coverage gap that shipped a broken segmented path) --


def _dummy_pattern() -> PatternBatch:
    """An observed pattern over ``_BEAM_HKL`` -- values irrelevant, only dtype/finiteness matter."""
    return PatternBatch(
        hkl=torch.tensor(_BEAM_HKL, dtype=torch.int64),
        intensities=torch.tensor([1.0, 0.5, 0.5], dtype=torch.float64),
        sigmas=torch.full((3,), 0.01, dtype=torch.float64),
    )


def _rocking_orientation() -> OrientationPlan:
    """A multi-tilt (rocking) plan -> the batched, non-segmented solve path."""
    grid, *_ = _silicon()
    tilts = rocking_curve_tilts(0.5, 3, geometry="continuous_rotation")  # (3, 3, 3)
    return OrientationPlan.build(
        grid, _BEAM_HKL, _dummy_pattern(), energy=_ENERGY, thickness=(300.0,), tilts=tilts
    )


def _segmented_orientation() -> SegmentedOrientationPlan:
    """A 2-segment / 2-tilt coupled plan -> the segmented solve path (the one that shipped bad)."""
    grid, *_ = _silicon()
    tilts = rocking_curve_tilts(0.5, 2, geometry="continuous_rotation")  # (2, 3, 3)
    return SegmentedOrientationPlan.build(
        grid,
        [(_BEAM_HKL, (0,)), (_BEAM_HKL, (1,))],  # two chunks, one tilt each, tiling 0..1
        _dummy_pattern(),
        energy=_ENERGY,
        thickness=(300.0,),
        u0=0.0,
        orientation=np.eye(3),
        tilts=tilts,
    )


def _engine_over(orientation: OrientationPlan | SegmentedOrientationPlan, *, precision: str):
    grid, asu_plan, spec, numbers = _silicon()
    return RefinementEngine(
        spec=spec,
        asu_plan=asu_plan,  # type: ignore[arg-type]
        numbers=numbers,
        grid=grid,
        orientations=(orientation,),
        loss=w_rbragg_loss,
        precision=precision,  # type: ignore[arg-type]
    )


def test_fp32_scores_through_the_segmented_coupled_path() -> None:
    """The test that would have caught the shipped bug: fp32 on the segmented (coupled) solve.

    ``_solve_segmented`` allocated its rocking-curve buffer from ``thicknesses.dtype`` (float64) and
    scattered float32 segment intensities into it -- a dtype mismatch that raised on the fp32
    coupled path (never exercised, as phase 1 tested only the static/single-system path). fp32 must
    now run the coupled solve in complex64 -> float32 intensities and still score finite.
    """
    orientation = _segmented_orientation()
    engine = _engine_over(orientation, precision="fp32")

    (solution,) = engine.simulate(_params())
    assert solution.intensities.dtype == torch.float32
    assert solution.amplitudes.dtype == torch.complex64  # matches the static/batched fp32 paths

    score = engine.score_orientation(orientation, engine.fgb(_params()))
    assert score.shape == () and torch.isfinite(score)


def test_fp32_scores_through_the_batched_tilt_path() -> None:
    """Confirm the rocking (batched, non-segmented) path really is fine at fp32, not just likely."""
    orientation = _rocking_orientation()
    engine = _engine_over(orientation, precision="fp32")

    (solution,) = engine.simulate(_params())
    assert solution.intensities.dtype == torch.float32
    assert solution.amplitudes.dtype == torch.complex64

    score = engine.score_orientation(orientation, engine.fgb(_params()))
    assert score.shape == () and torch.isfinite(score)


@pytest.mark.parametrize("build", [_rocking_orientation, _segmented_orientation])
def test_fp64_keeps_the_integrating_paths_in_double(build) -> None:
    """fp64 (the default) casts thicknesses to float64 -- a no-op -- so the solve stays complex128 /
    float64 on both integrating paths. (Byte-identity to the pre-fix output is locked by the
    existing default-precision forward tests in test_engine / test_plan_serialize.)"""
    orientation = build()
    (solution,) = _engine_over(orientation, precision="fp64").simulate(_params())
    assert solution.intensities.dtype == torch.float64
    assert solution.amplitudes.dtype == torch.complex128
    assert torch.isfinite(solution.intensities).all()

    assert "precision" not in inspect.signature(run_inference).parameters

"""Slice 11 (5b): ``optimize_orientation`` -- the local Nelder-Mead orientation refinement.

``hexagonal_tilt`` is checked as a proper rotation (and that right-multiplying it preserves a
non-orthonormal determinant -- the re-orthonormalisation trap); it remains a shared geometry
primitive even though the search itself no longer uses it. ``optimize_orientation`` is then exercised
end-to-end on the same fast synthetic silicon system as ``test_scoring``: an already
self-consistent orientation is left essentially unchanged. Recovery of a *specific* perturbed
orientation needs a unique minimum, which this trivial high-symmetry system does not have -- that
is deferred to the real quartz e2e.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch
from tests.unit.synthetic import make_constraint_spec

from diffBloch.app.loggers import EarlyAbortLogger, FitAbortedError
from diffBloch.core.products import PatternBatch
from diffBloch.core.symmetry import build_asu_expansion_plan
from diffBloch.engine import OrientationPlan, RefinementEngine, StructureFactorGrid, w_rbragg_loss
from diffBloch.params import ConstraintSpec, RefinableParams
from diffBloch.preprocess import RefinementSetup, hexagonal_tilt, optimize_orientation
from diffBloch.preprocess.orientation import rocking_curve_tilts
from diffBloch.preprocess.plan import Plan
from diffBloch.specs import (
    BeamSelection,
    NelderMeadSearch,
    ScoredHklSelection,
    TrialCoupling,
    UnionCoupling,
)

_ENERGY = 200e3
_CELL = np.eye(3, dtype=np.float64) * 5.0
_BEAM_HKL = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.int64)


# --- hexagonal_tilt (pure) ------------------------------------------------------------------------


def test_hexagonal_tilt_is_a_proper_rotation() -> None:
    tilt = hexagonal_tilt(120.0, 0.3)
    assert np.allclose(tilt @ tilt.T, np.eye(3))  # orthogonal
    assert abs(np.linalg.det(tilt) - 1.0) < 1e-12  # proper (det = +1)


def test_hexagonal_tilt_zero_azimuth_is_a_pure_x_rotation() -> None:
    theta = np.deg2rad(0.4)
    rx = np.array(
        [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]
    )
    assert np.allclose(hexagonal_tilt(0.0, 0.4), rx)


def test_hexagonal_tilt_right_multiply_preserves_a_non_orthonormal_determinant() -> None:
    m = np.diag([1.01, 1.0, 1.0])  # non-orthonormal (det != 1)
    assert abs(np.linalg.det(m @ hexagonal_tilt(60.0, 0.25)) - np.linalg.det(m)) < 1e-12


# --- optimize_orientation (synthetic silicon) ----------------------------------------------------------


def _silicon() -> tuple[StructureFactorGrid, object, ConstraintSpec, torch.Tensor]:
    grid = StructureFactorGrid.from_cell(_CELL, g_max=0.45)
    asu_plan = build_asu_expansion_plan(np.zeros((1, 3)), np.eye(3)[None], np.zeros((1, 3)))
    spec = make_constraint_spec(reciprocal_basis=grid.reciprocal_basis)
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


def _self_consistent(
    grid: StructureFactorGrid,
    asu_plan: object,
    spec: ConstraintSpec,
    numbers: torch.Tensor,
    orientation: np.ndarray,
    tilts: np.ndarray | None = None,
) -> OrientationPlan:
    """An OrientationPlan whose observed pattern is what the engine simulates at ``orientation``.

    When ``tilts`` is given the observed intensities are the engine's rocking-curve *integration*
    over that tilt set, so the returned plan is self-consistent *under integration*.
    """
    dummy = PatternBatch(
        hkl=torch.tensor(_BEAM_HKL, dtype=torch.int64),
        intensities=torch.zeros(3, dtype=torch.float64),
        sigmas=torch.ones(3, dtype=torch.float64),
    )
    seed = OrientationPlan.build(
        grid,
        _BEAM_HKL,
        dummy,
        energy=_ENERGY,
        thickness=(300.0,),
        orientation=orientation,
        tilts=tilts,
    )
    engine = RefinementEngine(
        spec=spec,
        asu_plan=asu_plan,  # type: ignore[arg-type]
        numbers=numbers,
        grid=grid,
        orientations=(seed,),
        loss=w_rbragg_loss,
    )
    (solution,) = engine.simulate(_params())
    observed = PatternBatch(
        hkl=solution.beam_hkl,
        intensities=solution.intensities[0].detach(),
        sigmas=torch.full((3,), 0.01, dtype=torch.float64),
    )
    return OrientationPlan.build(
        grid,
        _BEAM_HKL,
        observed,
        energy=_ENERGY,
        thickness=(300.0,),
        orientation=orientation,
        tilts=tilts,
    )


def test_fit_orientation_leaves_a_self_consistent_orientation_unchanged() -> None:
    grid, asu_plan, spec, numbers = _silicon()
    true_orientation = np.eye(3, dtype=np.float64)
    matched = _self_consistent(grid, asu_plan, spec, numbers, true_orientation)
    refinement = _refinement(asu_plan, spec, numbers)

    (refined,) = optimize_orientation(refinement, NelderMeadSearch())(
        Plan(structure_factor_grid=grid, orientations=(matched,))
    ).orientations

    # Already optimal: the search must not wander it away from the seed.
    assert np.linalg.norm(np.asarray(refined.orientation) - true_orientation) < 1e-2


def test_fit_orientation_stamps_the_dataset_label_onto_the_started_event() -> None:
    """dataset_label reaches OrientationOptimizationStarted, so a pooled multi-dataset console log
    can tell which dataset a "N rotation(s)" announcement belongs to."""
    from diffBloch.observability import OrientationOptimizationStarted, RecordingLogger

    grid, asu_plan, spec, numbers = _silicon()
    matched = _self_consistent(grid, asu_plan, spec, numbers, np.eye(3, dtype=np.float64))
    refinement = _refinement(asu_plan, spec, numbers)
    recorder = RecordingLogger()

    optimize_orientation(
        refinement, NelderMeadSearch(), logger=recorder, dataset_label="a.cif_pets"
    )(Plan(structure_factor_grid=grid, orientations=(matched,)))

    (started,) = [e for e in recorder.events if isinstance(e, OrientationOptimizationStarted)]
    assert started.dataset == "a.cif_pets"


def test_fit_orientation_threads_the_rocking_curve_tilts_through_the_search() -> None:
    # A Plan carrying a rocking-curve tilt set must be scored *under integration* at every trial --
    # the fit/eval consistency invariant. The trial builds thread op.tilts through unchanged, so an
    # already-integration-optimal orientation stays put AND the returned plan keeps its N tilts. A
    # dropped tilt set would score current-integrated vs trials-static (the latent bug this closes).
    grid, asu_plan, spec, numbers = _silicon()
    tilts = rocking_curve_tilts(0.5, 3, geometry="continuous_rotation")
    matched = _self_consistent(grid, asu_plan, spec, numbers, np.eye(3), tilts=tilts)
    assert len(matched.tilts) == 3  # the seed carries the integration geometry
    refinement = _refinement(asu_plan, spec, numbers)

    (refined,) = optimize_orientation(refinement, NelderMeadSearch())(
        Plan(structure_factor_grid=grid, orientations=(matched,))
    ).orientations

    assert len(refined.tilts) == 3  # trials preserved the tilt set (not dropped to a static N=1)
    assert np.linalg.norm(np.asarray(refined.orientation) - np.eye(3)) < 1e-2


# --- device knob (the coupled fit runs on the accelerator; params.device is authoritative) --------


def test_fit_orientation_device_cpu_is_a_no_op() -> None:
    """``device='cpu'`` reproduces the default fit exactly -- device plumbing inert on CPU."""
    grid, asu_plan, spec, numbers = _silicon()
    matched = _self_consistent(grid, asu_plan, spec, numbers, np.eye(3, dtype=np.float64))
    refinement = _refinement(asu_plan, spec, numbers)
    plan = Plan(structure_factor_grid=grid, orientations=(matched,))

    (base,) = optimize_orientation(refinement, NelderMeadSearch())(plan).orientations
    (cpu,) = optimize_orientation(refinement, NelderMeadSearch(), device="cpu")(plan).orientations
    assert torch.equal(cpu.orientation, base.orientation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required (runs on the A100)")
def test_fit_orientation_cuda_matches_cpu_within_tolerance() -> None:
    """CPU<->GPU parity: the on-device search reproduces the CPU-fitted orientation to solver tol.

    Skipped locally (this Mac has no complex-capable GPU); the real assertion runs on the SSEC A100,
    where it pins that ``device='cuda'`` carries the whole per-trial eigensolve onto the accelerator
    and the greedy search still converges to the same orientation the CPU fit finds. The ``< 1e-6``
    assumes a well-separated descent (it holds for this clean synthetic): the greedy
    first-improvement accept is a branch point, so a near-tie between two trial tilts could let the
    ~1e-11 cross-device score shift steer the path elsewhere -- a full-radius orientation
    difference, not 1e-11 (search-path divergence under FP perturbation, not a device bug), and
    harmless: reproducibility is anchored at the checkpoint boundary; only a fresh GPU-computed
    checkpoint would differ.
    """
    grid, asu_plan, spec, numbers = _silicon()
    matched = _self_consistent(grid, asu_plan, spec, numbers, np.eye(3, dtype=np.float64))
    refinement = _refinement(asu_plan, spec, numbers)
    plan = Plan(structure_factor_grid=grid, orientations=(matched,))

    (cpu,) = optimize_orientation(refinement, NelderMeadSearch(), device="cpu")(plan).orientations
    (cuda,) = optimize_orientation(refinement, NelderMeadSearch(), device="cuda")(plan).orientations
    assert np.linalg.norm(np.asarray(cpu.orientation) - np.asarray(cuda.orientation)) < 1e-6


# --- coupled coverage guard (wired into run; the unit-level invariant lives in test_coupling) -----


def test_fit_orientation_coupled_guard_rejects_a_grid_too_small_for_the_coupling() -> None:
    """A coupled fit whose grid cannot span the beam-difference support fails loudly at setup.

    The synthetic silicon grid is g_max=0.45; the default coupling cap is 2.05, so 2*2.05=4.1 far
    exceeds it. The guard must raise from ``run`` -- before the search -- rather than let the
    per-trial gathers silently gather zeros. Proves the coverage guard is actually wired in (the
    scalar invariant itself is tested in test_coupling).
    """
    grid, asu_plan, spec, numbers = _silicon()  # g_max = 0.45
    matched = _self_consistent(grid, asu_plan, spec, numbers, np.eye(3, dtype=np.float64))
    coupling = TrialCoupling(
        policy=UnionCoupling(), scored=ScoredHklSelection(klar=BeamSelection(), g_max=0.3)
    )
    with pytest.raises(ValueError, match="silently gather zeros|grid g_max"):
        optimize_orientation(
            _refinement(asu_plan, spec, numbers), NelderMeadSearch(), coupling=coupling
        )(Plan(structure_factor_grid=grid, orientations=(matched,)))


def test_fit_orientation_workers_matches_sequential() -> None:
    """``workers>1`` fans rotations over threads but returns byte-identical results (order kept).

    Rotations are independent and the engine/``F_gb`` are read-only shared context, so a threaded
    run must reproduce the sequential fit exactly -- the property that lets ``workers`` be
    execution-only (out of the recipe identity). Pinned on the fast synthetic system.
    """
    grid, asu_plan, spec, numbers = _silicon()
    matched = _self_consistent(grid, asu_plan, spec, numbers, np.eye(3, dtype=np.float64))
    refinement = _refinement(asu_plan, spec, numbers)
    plan = Plan(structure_factor_grid=grid, orientations=(matched,) * 5)

    sequential = optimize_orientation(refinement, NelderMeadSearch())(plan).orientations
    threaded = optimize_orientation(refinement, NelderMeadSearch(), workers=3)(plan).orientations
    for a, b in zip(sequential, threaded, strict=True):
        assert torch.equal(a.orientation, b.orientation)


def test_fit_orientation_workers_abort_cancels_pending_rotations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under ``workers > 1`` an early abort cancels the queued rotations, not drains them.

    Regression for the ThreadPoolExecutor drain: a ``with`` block's ``shutdown(wait=True)`` ran
    every submitted search before the ``FitAbortedError`` surfaced, defeating the point of stopping
    early. With ``cancel_futures`` the queued rotations' compute is saved (only the ``<= workers``
    already in flight finish). The per-rotation search is stubbed so the test is fast; we count how
    many actually run and assert it is below the total -- which the old draining code could not do.
    """
    import diffBloch.preprocess.steps.optimize_orientation as fo

    calls: list[int] = []

    def stub(  # type: ignore[no-untyped-def]
        engine,
        fgb,
        plan,
        op,
        *,
        search,
        coupling,
        validate=True,
    ):
        calls.append(1)
        time.sleep(0.03)  # a window in which the abort can cancel the still-queued rotations
        return fo._FitResult(
            plan=op,
            score=0.5,
            n_trials=1,
            n_passes=1,
            alpha=0.0,
            beta=0.0,
            omega=0.0,
            seed_score=0.5,
        )

    monkeypatch.setattr(fo, "_refine_one", stub)
    grid, asu_plan, spec, numbers = _silicon()
    matched = _self_consistent(grid, asu_plan, spec, numbers, np.eye(3, dtype=np.float64))
    plan = Plan(structure_factor_grid=grid, orientations=(matched,) * 40)
    refinement = _refinement(asu_plan, spec, numbers)
    logger = EarlyAbortLogger(wr2_ceiling=-1.0, patience=1)  # aborts on the first fit event

    with pytest.raises(FitAbortedError):
        optimize_orientation(refinement, NelderMeadSearch(), workers=4, logger=logger)(plan)
    assert len(calls) < 40  # queued rotations cancelled; the draining code ran all 40


def test_nelder_mead_search_rejects_a_nonpositive_iteration_cap() -> None:
    # Bound validation lives in NelderMeadSearch (parse, don't validate). This pins that
    # optimize_orientation's cap is one of those validated bounds.
    with pytest.raises(ValueError, match="max_iterations must be >= 1"):
        NelderMeadSearch(max_iterations=0)

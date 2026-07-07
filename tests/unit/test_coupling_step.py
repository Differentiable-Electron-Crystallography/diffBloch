"""Phase 2b: the ``couple_beams`` step + engine reassembly of a ``SegmentedOrientationPlan``.

The tilt-segment coupling *geometry* is proven in :mod:`test_coupling` (segments match the private
byte-for-byte) and the end-to-end forward parity in the ``test_coupling_parity`` e2e. Here we pin
the
step's wiring/guards and the engine's reassemble-then-reduce invariant on the fast synthetic silicon
(reusing ``test_inference``'s helpers): a segmentation whose chunks all carry the *full* beam set
and
merely partition the tilts must reduce to exactly the tilt-independent integrated solve -- the
reassembly adds beams per chunk, it does not change the physics when the beam set is unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tests.unit.test_inference import (
    _BEAM_HKL,
    _ENERGY,
    _METHOD,
    _refinement,
    _silicon,
    _simulated_intensities,
)

from diffBloch.core.products import MosaicSmoothed, PatternBatch, align
from diffBloch.engine import SegmentedOrientationPlan
from diffBloch.engine.plan import OrientationPlan
from diffBloch.preprocess import couple_beams, run_inference
from diffBloch.preprocess.orientation import rocking_curve_tilts
from diffBloch.preprocess.plan import Plan
from diffBloch.preprocess.scoring import build_engine
from diffBloch.specs import TiltIndependent, TiltSegmentUnion

_TILTS = rocking_curve_tilts(1.0, 4, geometry="continuous_rotation")  # (4, 3, 3)


def _pattern(intensities: torch.Tensor) -> PatternBatch:
    return PatternBatch(
        hkl=torch.tensor(_BEAM_HKL, dtype=torch.int64),
        intensities=intensities,
        sigmas=torch.full((len(_BEAM_HKL),), 0.01, dtype=torch.float64),
    )


def _rot_z(deg: float) -> np.ndarray:
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def test_with_orientation_reuses_the_gather_and_matches_a_fresh_build() -> None:
    """``OrientationPlan.with_orientation`` rebuilds at a new orientation, reusing the F-gather."""
    grid, asu_plan, spec, numbers = _silicon()
    refinement = _refinement(asu_plan, spec, numbers)
    pattern = _pattern(_simulated_intensities(grid, asu_plan, spec, numbers))
    seed = OrientationPlan.build(
        grid, _BEAM_HKL, pattern, energy=_ENERGY, thickness=(300.0,), tilts=_TILTS
    )
    target = _rot_z(3.0)

    rebuilt = seed.with_orientation(grid, target)
    fresh = OrientationPlan.build(
        grid,
        _BEAM_HKL,
        pattern,
        energy=_ENERGY,
        thickness=(300.0,),
        orientation=target,
        tilts=_TILTS,
    )

    assert all(bp.gather is seed.beam_plans[0].gather for bp in rebuilt.beam_plans)  # reused
    assert np.array_equal(rebuilt.orientation.numpy(), target)
    eng_rebuilt = build_engine(Plan(grid=grid, orientations=(rebuilt,)), refinement, method=_METHOD)
    eng_fresh = build_engine(Plan(grid=grid, orientations=(fresh,)), refinement, method=_METHOD)
    torch.testing.assert_close(
        eng_rebuilt.simulate(refinement.params)[0].intensities,
        eng_fresh.simulate(refinement.params)[0].intensities,
    )


def test_with_orientation_identity_reproduces_the_plan() -> None:
    grid, asu_plan, spec, numbers = _silicon()
    refinement = _refinement(asu_plan, spec, numbers)
    pattern = _pattern(_simulated_intensities(grid, asu_plan, spec, numbers))
    seed = OrientationPlan.build(
        grid,
        _BEAM_HKL,
        pattern,
        energy=_ENERGY,
        thickness=(300.0,),
        tilts=_TILTS,
        orientation=_rot_z(2.0),
    )
    same = seed.with_orientation(grid, seed.orientation)

    eng_seed = build_engine(Plan(grid=grid, orientations=(seed,)), refinement, method=_METHOD)
    eng_same = build_engine(Plan(grid=grid, orientations=(same,)), refinement, method=_METHOD)
    torch.testing.assert_close(
        eng_seed.simulate(refinement.params)[0].intensities,
        eng_same.simulate(refinement.params)[0].intensities,
    )


def test_segmented_with_orientation_reuses_gathers_and_preserves_scored_set() -> None:
    """The segmented rebuild verb: new orientation, same segments/gathers, pinned scored set."""
    grid, asu_plan, spec, numbers = _silicon()
    refinement = _refinement(asu_plan, spec, numbers)
    pattern = _pattern(_simulated_intensities(grid, asu_plan, spec, numbers))
    chunk_a = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
    chunk_b = np.array([[0, 0, 0], [-1, 0, 0]], dtype=np.int64)
    scored = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64)  # 100 in union, -100 excluded
    build_kwargs = dict(energy=_ENERGY, thickness=(300.0,), u0=0.0, tilts=_TILTS, scored_hkl=scored)
    seed = SegmentedOrientationPlan.build(
        grid, [(chunk_a, (0, 1)), (chunk_b, (2, 3))], pattern, orientation=np.eye(3), **build_kwargs
    )
    target = _rot_z(3.0)

    rebuilt = seed.with_orientation(grid, target)
    fresh = SegmentedOrientationPlan.build(
        grid, [(chunk_a, (0, 1)), (chunk_b, (2, 3))], pattern, orientation=target, **build_kwargs
    )

    for reb_seg, seed_seg in zip(rebuilt.segments, seed.segments, strict=True):
        assert reb_seg.plan.beam_plans[0].gather is seed_seg.plan.beam_plans[0].gather  # reused
    assert rebuilt.alignment.hkl.tolist() == seed.alignment.hkl.tolist()  # scored set idempotent
    eng_reb = build_engine(Plan(grid=grid, orientations=(rebuilt,)), refinement, method=_METHOD)
    eng_fresh = build_engine(Plan(grid=grid, orientations=(fresh,)), refinement, method=_METHOD)
    torch.testing.assert_close(
        eng_reb.simulate(refinement.params)[0].intensities,
        eng_fresh.simulate(refinement.params)[0].intensities,
    )


def test_couple_beams_tilt_independent_is_the_identity() -> None:
    grid, asu_plan, spec, numbers = _silicon()
    pattern = _pattern(_simulated_intensities(grid, asu_plan, spec, numbers))
    op = OrientationPlan.build(grid, _BEAM_HKL, pattern, energy=_ENERGY, thickness=(300.0,))
    plan = Plan(grid=grid, orientations=(op,))

    coupled = couple_beams(TiltIndependent())(plan)

    assert coupled.orientations[0] is op  # untouched: the shared beam set is kept as-is


def test_couple_beams_requires_a_rocking_curve_tilt_set() -> None:
    grid, asu_plan, spec, numbers = _silicon()
    pattern = _pattern(_simulated_intensities(grid, asu_plan, spec, numbers))
    op = OrientationPlan.build(grid, _BEAM_HKL, pattern, energy=_ENERGY, thickness=(300.0,))

    with pytest.raises(ValueError, match="requires a rocking-curve tilt set"):
        couple_beams(TiltSegmentUnion())(Plan(grid=grid, orientations=(op,)))


def test_segmented_build_unions_the_beams_and_preserves_the_reduction() -> None:
    grid, _asu_plan, _spec, _numbers = _silicon()
    pattern = _pattern(torch.zeros(len(_BEAM_HKL), dtype=torch.float64))
    # Two chunks with different (overlapping) beam sets; the union is sorted + deduplicated and 000
    # survives. Covers partition the 4 tilts.
    chunk_a = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
    chunk_b = np.array([[0, 0, 0], [-1, 0, 0]], dtype=np.int64)
    sop = SegmentedOrientationPlan.build(
        grid,
        [(chunk_a, (0, 1)), (chunk_b, (2, 3))],
        pattern,
        energy=_ENERGY,
        thickness=(300.0,),
        u0=0.0,
        orientation=np.eye(3),
        tilts=_TILTS,
        tilt_reduction=MosaicSmoothed(3),
    )

    union = sop.beam_hkl.tolist()
    assert union == sorted(union)  # np.unique ordering
    assert [0, 0, 0] in union and [1, 0, 0] in union and [-1, 0, 0] in union
    assert sop.tilt_reduction == MosaicSmoothed(3)  # carried through from the coupled orientation
    # union_index round-trips each chunk's beams to their union columns.
    for segment, chunk in zip(sop.segments, (chunk_a, chunk_b), strict=True):
        placed = sop.beam_hkl[segment.union_index].numpy()
        assert np.array_equal(placed, chunk)


def test_segmented_solve_equals_plain_integration_for_a_trivial_partition() -> None:
    """A segmentation that keeps the full beam set in every chunk must equal the plain solve."""
    grid, asu_plan, spec, numbers = _silicon()
    refinement = _refinement(asu_plan, spec, numbers)
    pattern = _pattern(_simulated_intensities(grid, asu_plan, spec, numbers))

    plain = OrientationPlan.build(
        grid, _BEAM_HKL, pattern, energy=_ENERGY, thickness=(300.0,), tilts=_TILTS
    )
    # Same full beam set in both chunks; covers partition the tilts. Reassembling identical beam
    # sets and summing over all tilts is exactly the tilt-independent integrated solve.
    segmented = SegmentedOrientationPlan.build(
        grid,
        [(_BEAM_HKL, (0, 1)), (_BEAM_HKL, (2, 3))],
        pattern,
        energy=_ENERGY,
        thickness=(300.0,),
        u0=0.0,
        orientation=np.eye(3),
        tilts=_TILTS,
    )

    engine = build_engine(Plan(grid=grid, orientations=(plain,)), refinement, method=_METHOD)
    plain_solution = engine.simulate(refinement.params)[0]
    seg_engine = build_engine(
        Plan(grid=grid, orientations=(segmented,)), refinement, method=_METHOD
    )
    seg_solution = seg_engine.simulate(refinement.params)[0]

    plain_aligned = align(plain_solution, plain.pattern, plain.alignment).calculated
    seg_aligned = align(seg_solution, segmented.pattern, segmented.alignment).calculated
    assert torch.allclose(plain_aligned, seg_aligned, atol=1e-10)


def test_run_inference_scores_a_segmented_plan() -> None:
    """The whole inference terminal works on a segmented plan (align/score are union-agnostic)."""
    grid, asu_plan, spec, numbers = _silicon()
    refinement = _refinement(asu_plan, spec, numbers)
    pattern = _pattern(_simulated_intensities(grid, asu_plan, spec, numbers))
    segmented = SegmentedOrientationPlan.build(
        grid,
        [(_BEAM_HKL, (0, 1)), (_BEAM_HKL, (2, 3))],
        pattern,
        energy=_ENERGY,
        thickness=(300.0,),
        u0=0.0,
        orientation=np.eye(3),
        tilts=_TILTS,
    )

    result = run_inference(Plan(grid=grid, orientations=(segmented,)), refinement, method=_METHOD)

    assert len(result.per_rotation) == 1
    assert result.per_rotation[0].r_obs < 1e-6  # self-consistent pattern


def test_scored_set_stays_pinned_when_the_solve_union_is_larger() -> None:
    """The regression guard for the 0.4825 drift: expanding the solve set must not widen scoring.

    A segmented plan whose union beam set is a strict superset of the scored set must score only
    ``scored_hkl ∩ union`` -- not every observed reflection that happens to land in the enlarged
    union.
    """
    grid, asu_plan, spec, numbers = _silicon()
    refinement = _refinement(asu_plan, spec, numbers)
    pattern = _pattern(_simulated_intensities(grid, asu_plan, spec, numbers))
    # Solve union covers the full beam set (3 reflections incl. 000); pin scoring to just 000.
    scored = np.array([[0, 0, 0]], dtype=np.int64)
    segmented = SegmentedOrientationPlan.build(
        grid,
        [(_BEAM_HKL, (0, 1)), (_BEAM_HKL, (2, 3))],
        pattern,
        energy=_ENERGY,
        thickness=(300.0,),
        u0=0.0,
        orientation=np.eye(3),
        tilts=_TILTS,
        scored_hkl=scored,
    )

    # union is the full 3-beam set, but the scored axis is only the pinned reflection.
    assert segmented.beam_hkl.shape[0] == len(_BEAM_HKL)
    assert segmented.alignment.hkl.tolist() == [[0, 0, 0]]
    engine = build_engine(Plan(grid=grid, orientations=(segmented,)), refinement, method=_METHOD)
    aligned = align(engine.simulate(refinement.params)[0], segmented.pattern, segmented.alignment)
    assert aligned.calculated.shape[-1] == 1  # scored on 1 reflection, not the 3-beam union

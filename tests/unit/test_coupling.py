"""Unit tests for :mod:`diffBloch.preprocess.coupling` against the vendored private segments.

The parity fixture (``tests/fixtures/quartz_anchor/parity_replay/``) carries the
``diffBloch_private`` reference's *exact* per-rotation coupling: the split boundaries and, per
segment, the union beam set (``seg{k}_hkl``) and covered tilt indices (``seg{k}_cover``), dumped
straight from ``BlochNet.forward``. These tests recompute that coupling from the
:class:`~diffBloch.specs.TiltSegmentUnion` policy alone and assert it matches -- proving the port of
the private's per-tilt excitation mask + boundary-union partition, independent of any solver.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from diffBloch.config import load_experiment
from diffBloch.engine import ScatteringGrid
from diffBloch.io import read_structure
from diffBloch.preprocess.coupling import Segment, tilt_segment_coupling
from diffBloch.specs import TiltSegmentUnion

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "quartz_anchor"
REPLAY_ROOT = FIXTURE_ROOT / "parity_replay"
ENERGY_EV = 200_000.0  # the private's exact beam energy (see the parity fixture README)
ROTATIONS = (13, 27, 60, 61, 64)


def _grid_and_tilts() -> tuple[ScatteringGrid, np.ndarray]:
    cfg, _ = load_experiment(FIXTURE_ROOT)
    structure = read_structure(FIXTURE_ROOT / cfg.inputs.structure)
    grid = ScatteringGrid.from_cell(structure.unit_cell, g_max=cfg.numerics.g_max)
    tilts = np.load(REPLAY_ROOT / "tilts.npz")["tilts"]
    return grid, tilts


def _compute(rotation: int) -> tuple[tuple[Segment, ...], np.lib.npyio.NpzFile]:
    grid, tilts = _grid_and_tilts()
    d = np.load(REPLAY_ROOT / f"rot_{rotation}.npz")
    segments = tilt_segment_coupling(
        TiltSegmentUnion(),
        np.asarray(grid.grid_hkl),
        cell=np.asarray(grid.cell),
        orientation=d["orientation"],
        tilts=tilts,
        energy=ENERGY_EV,
        u0=float(d["u0"]),
    )
    return segments, d


def _as_set(hkl: np.ndarray) -> set[tuple[int, int, int]]:
    return {(int(row[0]), int(row[1]), int(row[2])) for row in np.asarray(hkl)}


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_segment_count_and_covers_match_private(rotation: int) -> None:
    segments, d = _compute(rotation)
    assert len(segments) == int(d["n_segments"])
    for k, segment in enumerate(segments):
        assert list(segment.cover) == [int(t) for t in d[f"seg{k}_cover"]]
    # Covers tile every tilt exactly once, disjoint and contiguous.
    covered = [t for segment in segments for t in segment.cover]
    assert covered == list(range(42))


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_segment_beam_sets_match_private(rotation: int) -> None:
    segments, d = _compute(rotation)
    for k, segment in enumerate(segments):
        # Beam ordering is eigensolver-invariant, so the coupling is the beam *set* per segment.
        assert _as_set(segment.beam_hkl) == _as_set(d[f"seg{k}_hkl"]), f"segment {k} beam set"
    # (0,0,0) is always excited, so it is in every segment's union.
    for segment in segments:
        assert (0, 0, 0) in _as_set(segment.beam_hkl)


def test_split_boundaries_match_private() -> None:
    from diffBloch.preprocess.coupling import _split_boundaries

    d = np.load(REPLAY_ROOT / "rot_61.npz")
    boundaries = _split_boundaries(42, TiltSegmentUnion().n_splits)
    assert list(boundaries) == [int(b) for b in d["split_idx"]]


def test_policy_rejects_degenerate_cap() -> None:
    with pytest.raises(ValueError, match="cap"):
        TiltSegmentUnion(g_max=0.2, cap_margin=0.2)

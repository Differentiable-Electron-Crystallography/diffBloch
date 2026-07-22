"""Unit tests for :mod:`diffBloch.preprocess.coupling` against the vendored private segments.

The parity fixture (``tests/fixtures/quartz_anchor/parity_replay/``) carries the
``diffBloch_private`` reference's per-rotation coupling: the split boundaries and, per segment, the
union beam set (``seg{k}_hkl``) and covered tilt indices (``seg{k}_cover``), dumped straight from
``BlochNet.forward``. These fixtures are **pre-#154** (coupling cap ``|g| < g_max - 0.2 = 2.05``).
These tests recompute the coupling from the :class:`~diffBloch.specs.TiltSegmentUnion` policy alone:
the split boundaries and covers still match exactly, but the post-#154 policy (cap ``|g| < g_max``,
the ``- 0.2`` margin dropped) widens each segment's beam set, so the beam-set test asserts
**containment** (post-#154 ⊇ pre-#154, no beam dropped), not equality. Restoring equality needs a
regenerated post-#154 private replay -- a private-reference comparison tracked in KNOWN_ISSUES.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from diffBloch.config import load_experiment
from diffBloch.engine import ScatteringGrid
from diffBloch.io import read_structure
from diffBloch.preprocess.coupling import (
    Segment,
    tilt_segment_coupling,
)
from diffBloch.specs import TiltSegmentUnion, assert_grid_covers_coupling

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "quartz_anchor"
REPLAY_ROOT = FIXTURE_ROOT / "parity_replay"
ENERGY_EV = 200_000.0  # the private's exact beam energy (see the parity fixture README)
ROTATIONS = (13, 27, 60, 61, 64)


def _grid_and_tilts() -> tuple[ScatteringGrid, np.ndarray]:
    cfg, _ = load_experiment(FIXTURE_ROOT)
    structure = read_structure(FIXTURE_ROOT / cfg.inputs.structure)
    grid = ScatteringGrid.from_cell_for_solve_cutoff(
        structure.unit_cell, cfg.preprocess.coupling.g_max
    )
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
def test_segment_beam_sets_contain_private(rotation: int) -> None:
    # The replay goldens are the private's PRE-#154 segment beam sets, coupled at |g| < g_max - 0.2
    # (cap 2.05). Post-#154 drops that margin, so the coupling cap is the physical g_max (2.25) and
    # each segment admits MORE beams -- a strict superset of the private set, dropping none. We
    # assert containment (no private beam lost) rather than equality; the exact post-#154 set is a
    # private-reference comparison pending a private post-#154 replay.
    segments, d = _compute(rotation)
    for k, segment in enumerate(segments):
        # Beam ordering is eigensolver-invariant, so the coupling is the beam *set* per segment.
        assert _as_set(segment.beam_hkl) >= _as_set(d[f"seg{k}_hkl"]), f"segment {k} beam set"
    # (0,0,0) is always excited, so it is in every segment's union.
    for segment in segments:
        assert (0, 0, 0) in _as_set(segment.beam_hkl)


def test_split_boundaries_match_private() -> None:
    from diffBloch.preprocess.coupling import _split_boundaries

    d = np.load(REPLAY_ROOT / "rot_61.npz")
    boundaries = _split_boundaries(42, TiltSegmentUnion().n_splits)
    assert list(boundaries) == [int(b) for b in d["split_idx"]]


def test_policy_rejects_nonpositive_g_max() -> None:
    with pytest.raises(ValueError, match="g_max and sg_max must be positive"):
        TiltSegmentUnion(g_max=0.0)


# --- coverage guard (the O(1) invariant that makes validate=False sound on the coupled path) ------


def test_coverage_guard_passes_for_the_faithful_recipe() -> None:
    """Faithful quartz coupling (g_max 2.25) needs a grid >= 4.5, which the derived grid is."""
    assert_grid_covers_coupling(TiltSegmentUnion(), grid_g_max=4.5)  # does not raise


def test_coverage_guard_accepts_the_exact_boundary() -> None:
    """2*g_max == grid_g_max is sufficient (differences are strictly < 2*g_max), so equality ok."""
    policy = TiltSegmentUnion()
    assert_grid_covers_coupling(policy, grid_g_max=2.0 * policy.g_max)  # does not raise


def test_coverage_guard_raises_when_the_grid_is_too_small() -> None:
    """A grid g_max below 2*g_max would let a coupled difference fall outside the sphere -> silent
    zero under validate=False. The guard turns that into a loud, actionable error at setup."""
    policy = TiltSegmentUnion()  # g_max 2.25 -> needs grid g_max >= 4.5
    with pytest.raises(ValueError, match=r"grid g_max.*4\.5|silently gather zeros"):
        assert_grid_covers_coupling(policy, grid_g_max=4.0)

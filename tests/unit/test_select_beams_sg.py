"""``select_beams_sg``: excitation-error seed cut evaluated across the rocking-curve sweep."""

from __future__ import annotations

import numpy as np
import pytest
from tests.unit.synthetic import built_seed_system, seed_system

from diffBloch.core.crystal import orientation_basis
from diffBloch.core.dynamical import excitation_errors
from diffBloch.core.reciprocal import g_vectors
from diffBloch.preprocess import pipeline, select_beams_sg
from diffBloch.preprocess.orientation import rocking_curve_tilts
from diffBloch.specs import IntegrationGeometry, RockingCurve

_ROCKING = RockingCurve(sampling=41, integration=IntegrationGeometry(semiangle=2.0))


def _sg(plan, tilts):
    cp = plan.orientations[0]
    cell = np.asarray(plan.structure_factor_grid.cell)
    hkl = np.asarray(cp.beam_hkl, dtype=np.int64)
    rows = [
        np.abs(
            excitation_errors(
                g_vectors(hkl, orientation_basis(cell, t @ np.asarray(cp.orientation))),
                cp.energy,
                u0=cp.u0,
            )
        )
        for t in tilts
    ]
    return hkl, np.abs(rows[len(rows) // 2]), np.min(rows, axis=0)


def test_keeps_beams_excited_off_centre_and_drops_the_rest() -> None:
    _, seed = seed_system()
    tilts = rocking_curve_tilts(2.0, 41)
    hkl, centre, swept = _sg(seed, tilts)
    off_centre = (swept < centre) & (hkl != 0).any(axis=1)
    assert off_centre.any()
    i = int(np.argmax(np.where(off_centre, centre - swept, -np.inf)))
    sg_max = 0.5 * (swept[i] + centre[i])  # excited mid-sweep, not at the centre

    kept = np.asarray(select_beams_sg(sg_max, _ROCKING)(seed).orientations[0].beam_hkl)
    kept_rows = {tuple(r) for r in kept}

    assert tuple(hkl[i]) in kept_rows
    assert kept_rows == {tuple(r) for r in hkl[swept <= sg_max]}
    assert (0, 0, 0) in kept_rows


def test_records_scoped_cutoff_and_rocking() -> None:
    _, seed = seed_system()
    record = pipeline([select_beams_sg(0.01, _ROCKING)])(seed).provenance[-1]
    assert record.name == "select_beams_sg"
    assert "seed_sg_max" in str(record.params)


def test_rejects_built_plans_and_negative_cutoff() -> None:
    _, built = built_seed_system()
    with pytest.raises((TypeError, ValueError)):
        select_beams_sg(0.01, _ROCKING)(built)
    with pytest.raises(ValueError, match="non-negative"):
        select_beams_sg(-1.0, _ROCKING)

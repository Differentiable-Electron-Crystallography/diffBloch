"""``select_beams``: the Klar (2023) rsg/dsg active-beam filter and its ``Plan -> Plan`` step."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from diffBloch.config import load_config
from diffBloch.io import read_observations, read_structure
from diffBloch.preprocess import from_experiment, klar_beam_mask, select_beams

QUARTZ = Path(__file__).parent.parent / "fixtures" / "quartz_anchor"


def _quartz_train_plan():
    structure = read_structure(QUARTZ / "enantiomer_1.cif")
    observations = read_observations(QUARTZ / "exp_data.cif_pets")
    config = load_config(QUARTZ / "experiment.yaml")
    return from_experiment(structure, observations, config).plans.train, config


# --- the criterion --------------------------------------------------------------------------------


def test_klar_mask_keeps_near_ewald_in_plane_drops_on_axis() -> None:
    # Transverse plane is (x, y) for the -z beam. A near-Ewald reflection offset purely along x is
    # kept; one offset purely along z (along the beam, sg_max == 0) is dropped. This is the exact
    # swap of the private filter_hkls (y, z) convention (see KNOWN_ISSUES in diffBloch_private).
    g = np.array([[0.5, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]], dtype=np.float64)
    mask = klar_beam_mask(g, energy=200e3, rsg=0.9, dsg=0.0015, semiangle=1.0)
    assert mask.tolist() == [True, False, False]


def test_klar_mask_rsg_and_dsg_are_cutoffs() -> None:
    # One near-Ewald in-plane reflection; tightening rsg/dsg can only drop beams.
    g = np.array([[0.5, 0.0, 0.0]], dtype=np.float64)
    loose = klar_beam_mask(g, energy=200e3, rsg=0.9, dsg=0.0015, semiangle=1.0)
    tight_rsg = klar_beam_mask(g, energy=200e3, rsg=1e-6, dsg=0.0015, semiangle=1.0)
    tight_dsg = klar_beam_mask(g, energy=200e3, rsg=0.9, dsg=1e3, semiangle=1.0)
    assert loose.tolist() == [True]
    assert tight_rsg.tolist() == [False]  # relative excitation-error cutoff bites
    assert tight_dsg.tolist() == [False]  # minimum-margin cutoff bites


def test_klar_mask_rejects_non_3_column_g() -> None:
    import pytest

    with pytest.raises(ValueError, match="shape"):
        klar_beam_mask(np.zeros((4, 2)), energy=200e3, rsg=0.9, dsg=0.0015, semiangle=1.0)


# --- the Plan -> Plan step ------------------------------------------------------------------------


def test_select_beams_prunes_each_orientation_keeping_000_and_pattern() -> None:
    plan, config = _quartz_train_plan()
    step = select_beams(
        rsg=config.numerics.rsg,
        dsg=config.numerics.dsg,
        semiangle=config.numerics.integration_semiangle,
    )
    pruned = step(plan)

    # Plan -> Plan: the shared grid object is preserved; a new Plan is returned.
    assert pruned.grid is plan.grid
    assert pruned is not plan
    assert len(pruned.orientations) == len(plan.orientations)

    pruned_total = 0
    seed_total = 0
    for before, after in zip(plan.orientations, pruned.orientations, strict=True):
        seed = {tuple(row) for row in before.beam_hkl.tolist()}
        active = {tuple(row) for row in after.beam_hkl.tolist()}
        assert active <= seed  # selection only removes beams
        assert (0, 0, 0) in active  # 000 anchors psi0, always retained
        # pattern (observed data) is carried through untouched
        assert after.pattern is before.pattern
        pruned_total += len(active)
        seed_total += len(seed)

    assert pruned_total < seed_total  # the filter actually removes beams overall


def test_select_beams_keeps_orientation_energy_and_grid_coupling() -> None:
    plan, config = _quartz_train_plan()
    pruned = select_beams(
        rsg=config.numerics.rsg,
        dsg=config.numerics.dsg,
        semiangle=config.numerics.integration_semiangle,
    )(plan)
    before = plan.orientations[0]
    after = pruned.orientations[0]
    # Source/rebuild inputs are preserved; only the compiled beam set changes.
    assert np.allclose(after.orientation.numpy(), before.orientation.numpy())
    assert after.energy == before.energy
    assert after.u0 == before.u0
    # Rebuilt against the shared grid: the alignment bridges the pruned beams to the same pattern.
    assert after.alignment.hkl.shape[1] == 3

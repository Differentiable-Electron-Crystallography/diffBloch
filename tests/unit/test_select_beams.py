"""``select_beams``: the Klar (2023) rsg/dsg active-beam filter and its ``Plan -> Plan`` step."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from diffBloch.config import load_config
from diffBloch.io import read_observations, read_structure
from diffBloch.preprocess import (
    build_orientation_plans,
    from_experiment,
    klar_beam_mask,
    select_beams,
)
from diffBloch.preprocess.plan import CandidatePlan

QUARTZ = Path(__file__).parent.parent / "fixtures" / "quartz_anchor"


def _quartz_train_plan():
    structure = read_structure(QUARTZ / "enantiomer_1.cif")
    observations = read_observations(QUARTZ / "exp_data.cif_pets")
    config = load_config(QUARTZ / "experiment.yaml")
    setup = from_experiment(structure, observations, config)
    return setup.plans.train, config, setup.integration


# --- the criterion --------------------------------------------------------------------------------


def test_klar_mask_keeps_near_ewald_in_plane_drops_on_axis() -> None:
    # Continuous rotation rocks about the goniometer x axis, so sg_max is the distance from x =
    # |(g_y, g_z)|. A near-Ewald reflection offset perpendicular to the rock axis (along y) sweeps
    # and is kept; one offset purely along the rock axis (x) never sweeps (sg_max == 0) and is
    # dropped. This matches the private filter_hkls norm(k[:, 1:]).
    g = np.array([[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    mask = klar_beam_mask(g, energy=200e3, rsg=0.9, dsg=0.0015, semiangle=1.0)
    assert mask.tolist() == [True, False, False]


def test_klar_mask_precession_uses_beam_transverse() -> None:
    # Precession is an isotropic cone about the -z beam, so sg_max is the distance from z =
    # |(g_x, g_y)|. Now the x-offset reflection sweeps and is kept; the on-beam (z) one is dropped.
    g = np.array([[0.5, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]], dtype=np.float64)
    mask = klar_beam_mask(
        g, energy=200e3, rsg=0.9, dsg=0.0015, semiangle=1.0, geometry="precession"
    )
    assert mask.tolist() == [True, False, False]


def test_klar_mask_rsg_and_dsg_are_cutoffs() -> None:
    # One near-Ewald reflection off the rock axis (along y); tightening rsg/dsg can only drop beams.
    g = np.array([[0.0, 0.5, 0.0]], dtype=np.float64)
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
    plan, config, integration = _quartz_train_plan()
    step = select_beams(config.blochwave.to_beam_selection(integration))
    pruned = step(plan)

    # Plan -> Plan: the shared grid object is preserved; a new Plan is returned.
    assert pruned.structure_factor_grid is plan.structure_factor_grid
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


def test_select_beams_preserves_source_and_defers_the_build() -> None:
    plan, config, integration = _quartz_train_plan()
    pruned = select_beams(config.blochwave.to_beam_selection(integration))(plan)
    before = plan.orientations[0]
    after = pruned.orientations[0]
    # Source is preserved; select_beams stays on the candidate phase -- no gather is built here.
    assert isinstance(after, CandidatePlan)
    assert np.allclose(after.orientation, before.orientation)
    assert after.energy == before.energy
    assert after.u0 == before.u0
    # build_orientation_plans then bridges the pruned beams to the pattern via the alignment.
    built = build_orientation_plans()(pruned).orientations[0]
    assert built.alignment.hkl.shape[1] == 3

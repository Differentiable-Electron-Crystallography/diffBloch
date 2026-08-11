"""``select_beams``: the Klar (2023) rsg/dsg active-beam filter and its ``Plan -> Plan`` step."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from diffBloch.config import load_config
from diffBloch.core.products import MosaicSmoothed, PlainSum
from diffBloch.io import read_experimental_data, read_structure
from diffBloch.preprocess import (
    build_orientation_plans,
    from_experiment,
    klar_beam_mask,
    select_beams,
)
from diffBloch.preprocess.plan import CandidatePlan
from diffBloch.specs import IntegrationGeometry, Mosaicity, RockingCurve, UnionCoupling

QUARTZ = Path(__file__).parent.parent / "fixtures" / "quartz_anchor"


def _quartz_train_plan():
    structure = read_structure(QUARTZ / "enantiomer_1.cif")
    experimental_data = read_experimental_data(QUARTZ / "exp_data.cif_pets")
    config = load_config(QUARTZ / "experiment.yaml")
    setup = from_experiment(structure, experimental_data, config)
    return setup.plans.train, config, setup.integration


# --- the criterion --------------------------------------------------------------------------------


def test_klar_mask_keeps_near_ewald_in_plane_drops_on_axis() -> None:
    # Continuous rotation rocks about the goniometer x axis, so sg_max is the distance from x =
    # |(g_y, g_z)|. A near-Ewald reflection offset perpendicular to the rock axis (along y) sweeps
    # and is kept; one offset purely along the rock axis (x) never sweeps (sg_max == 0) and is
    # dropped.
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


def test_build_orientation_plans_directly_builds_final_rocking_geometry() -> None:
    plan, config, integration = _quartz_train_plan()
    pruned = select_beams(config.blochwave.to_beam_selection(integration))(plan)

    built = build_orientation_plans(
        config.blochwave.to_rocking_curve(integration),
        config.blochwave.mosaicity,
    )(pruned).orientations[0]

    assert built.tilts.shape == (config.blochwave.rocking_curve_sampling, 3, 3)
    assert len(built.beam_plans) == config.blochwave.rocking_curve_sampling
    assert isinstance(built.tilt_reduction, MosaicSmoothed)


def test_pets_mosaicity_sets_window_from_actual_tilt_spacing() -> None:
    plan, config, integration = _quartz_train_plan()
    candidate = replace(plan.orientations[0], mosaicity_degrees=0.6)
    plan = replace(plan, orientations=(candidate,))
    rocking = RockingCurve(
        integration=IntegrationGeometry(semiangle=1.0, geometry="continuous_rotation"), sampling=11
    )
    built = build_orientation_plans(rocking, True)(plan).orientations[0]

    assert len(built.beam_plans) == rocking.sampling
    assert isinstance(built.tilt_reduction, MosaicSmoothed)
    assert built.tilt_reduction.window == 3  # round(0.6 / (2 / (11 - 1)))


def test_pets_mosaicity_below_one_sample_uses_plain_sum() -> None:
    plan, _, _ = _quartz_train_plan()
    candidate = replace(plan.orientations[0], mosaicity_degrees=0.05)
    plan = replace(plan, orientations=(candidate,))
    rocking = RockingCurve(
        integration=IntegrationGeometry(semiangle=1.0, geometry="continuous_rotation"), sampling=11
    )
    built = build_orientation_plans(rocking, True)(plan).orientations[0]
    assert isinstance(built.tilt_reduction, PlainSum)


def test_legacy_mosaic_window_remains_supported() -> None:
    plan, config, integration = _quartz_train_plan()
    built = build_orientation_plans(
        config.blochwave.to_rocking_curve(integration), Mosaicity(window=3)
    )(plan).orientations[0]

    assert isinstance(built.tilt_reduction, MosaicSmoothed)
    assert built.tilt_reduction.window == 3


def test_build_orientation_plans_rejects_reduction_or_coupling_without_rocking() -> None:
    with pytest.raises(ValueError, match="mosaicity requires"):
        build_orientation_plans(mosaicity=Mosaicity(window=1))
    with pytest.raises(ValueError, match="coupling requires"):
        build_orientation_plans(coupling=UnionCoupling())


def test_build_orientation_plans_rejects_mosaic_window_larger_than_sampling() -> None:
    rocking = RockingCurve(
        integration=IntegrationGeometry(semiangle=1.0, geometry="continuous_rotation"),
        sampling=2,
    )
    with pytest.raises(ValueError, match="exceeds"):
        build_orientation_plans(rocking, Mosaicity(window=3))


def test_build_orientation_plans_builds_coupled_solve_geometry_before_alignment() -> None:
    plan, config, integration = _quartz_train_plan()

    built = build_orientation_plans(
        config.blochwave.to_rocking_curve(integration),
        config.blochwave.mosaicity,
        coupling=config.blochwave.to_policy(),
    )(plan).orientations[0]

    observed = {tuple(row) for row in plan.orientations[0].pattern.hkl.tolist()}
    scored = {tuple(row) for row in built.alignment.hkl.tolist()}
    simulated = {tuple(row) for row in built.beam_hkl.tolist()}
    assert scored == observed & simulated
    assert (0, 0, 0) in simulated


def test_build_orientation_plans_workers_preserve_exact_coupled_geometry() -> None:
    """Parallel rotation construction is an execution choice, not a scientific input."""
    plan, config, integration = _quartz_train_plan()
    plan = replace(plan, orientations=plan.orientations * 3)
    rocking = config.blochwave.to_rocking_curve(integration)
    mosaicity = config.blochwave.mosaicity
    coupling = config.blochwave.to_policy()
    sequential = build_orientation_plans(rocking, mosaicity, coupling=coupling, workers=1)(plan)
    threaded = build_orientation_plans(rocking, mosaicity, coupling=coupling, workers=3)(plan)

    for expected, actual in zip(sequential.orientations, threaded.orientations, strict=True):
        assert torch.equal(expected.beam_hkl, actual.beam_hkl)
        assert torch.equal(expected.alignment.hkl, actual.alignment.hkl)
        for expected_segment, actual_segment in zip(
            expected.segments, actual.segments, strict=True
        ):
            assert torch.equal(expected_segment.cover, actual_segment.cover)
            assert torch.equal(expected_segment.union_beam_index, actual_segment.union_beam_index)
            for expected_beam, actual_beam in zip(
                expected_segment.plan.beam_plans,
                actual_segment.plan.beam_plans,
                strict=True,
            ):
                assert torch.equal(expected_beam.diagonal, actual_beam.diagonal)
                assert torch.equal(
                    expected_beam.gather.beam_difference_indices,
                    actual_beam.gather.beam_difference_indices,
                )

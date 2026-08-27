"""``select_beams``: the Klar (2023) rsg/dsg active-beam filter and its ``Plan -> Plan`` step."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import torch

from diffBloch.config import load_config
from diffBloch.core.products import MosaicSmoothed, PlainSum, WeightedSum
from diffBloch.io import read_experimental_data, read_structure
from diffBloch.preprocess import (
    build_orientation_plans,
    from_experiment,
    klar_beam_mask,
    select_beams,
)
from diffBloch.preprocess.experiment import resolve_dataset_mosaicity
from diffBloch.preprocess.plan import CandidatePlan
from diffBloch.specs import IntegrationGeometry, IsotropicMosaicity, RockingCurve, UnionCoupling

QUARTZ = Path(__file__).parent.parent / "fixtures" / "quartz_anchor"


@lru_cache(maxsize=1)
def _quartz_train_plan():
    structure = read_structure(QUARTZ / "enantiomer_1.cif")
    experimental_data = read_experimental_data(QUARTZ / "exp_data.cif_pets")
    config = load_config(QUARTZ / "experiment.yaml")
    setup = from_experiment(structure, experimental_data, config)
    return setup.plans.train, config, setup.integration, setup.mosaicity


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
    plan, config, integration, _mosaicity = _quartz_train_plan()
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
    plan, config, integration, _mosaicity = _quartz_train_plan()
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
    plan, config, integration, mosaicity = _quartz_train_plan()
    pruned = select_beams(config.blochwave.to_beam_selection(integration))(plan)

    built = build_orientation_plans(
        config.blochwave.to_rocking_curve(integration),
        mosaicity,
    )(pruned).orientations[0]

    assert built.tilts.shape == (config.blochwave.rocking_curve_sampling, 3, 3)
    assert len(built.beam_plans) == config.blochwave.rocking_curve_sampling
    assert isinstance(built.tilt_reduction, PlainSum)


def test_enabled_pets_mosaicity_resolves_sample_span_from_actual_tilt_spacing() -> None:
    rocking = RockingCurve(
        integration=IntegrationGeometry(semiangle=1.0, geometry="continuous_rotation"), sampling=11
    )
    record = read_experimental_data(QUARTZ / "exp_data.cif_pets").model_copy(
        update={"mosaicity_degrees": 0.6}
    )
    reduction = resolve_dataset_mosaicity(True, record, rocking)
    assert reduction == MosaicSmoothed(samples=3)  # round(0.6 / (2 / 11))


def test_pets_mosaicity_below_one_sample_uses_plain_sum() -> None:
    rocking = RockingCurve(
        integration=IntegrationGeometry(semiangle=1.0, geometry="continuous_rotation"), sampling=11
    )
    record = read_experimental_data(QUARTZ / "exp_data.cif_pets").model_copy(
        update={"mosaicity_degrees": 0.05}
    )
    assert resolve_dataset_mosaicity(True, record, rocking) is None


def test_build_orientation_plans_rejects_reduction_or_coupling_without_rocking() -> None:
    with pytest.raises(ValueError, match="mosaicity requires"):
        build_orientation_plans(mosaicity=MosaicSmoothed(samples=1))
    with pytest.raises(ValueError, match="coupling requires"):
        build_orientation_plans(coupling=UnionCoupling())


def test_build_orientation_plans_rejects_mosaic_span_larger_than_sampling() -> None:
    rocking = RockingCurve(
        integration=IntegrationGeometry(semiangle=1.0, geometry="continuous_rotation"),
        sampling=2,
    )
    with pytest.raises(ValueError, match="exceeds"):
        build_orientation_plans(rocking, MosaicSmoothed(samples=3))


def test_build_orientation_plans_composes_isotropic_mosaic_tilts_onto_rocking_tilts() -> None:
    plan, config, integration, _mosaicity = _quartz_train_plan()
    pruned = select_beams(config.blochwave.to_beam_selection(integration))(plan)
    rocking = config.blochwave.to_rocking_curve(integration)
    mosaicity = IsotropicMosaicity(sigma_degrees=0.1, samples=5)

    built = build_orientation_plans(rocking, mosaicity)(pruned).orientations[0]

    # Every rocking tilt composed with every mosaic tilt -- not the axis_smoothed model's
    # moving-average reduction over the bare rocking tilts.
    n_combined = config.blochwave.rocking_curve_sampling * 5
    assert built.tilts.shape == (n_combined, 3, 3)
    assert len(built.beam_plans) == n_combined
    assert isinstance(built.tilt_reduction, WeightedSum)
    assert built.tilt_reduction.weights.shape == (n_combined,)
    # Weights repeat mosaic-minor (index = r * M + m): every block of 5 is the same 5 weights.
    weights = built.tilt_reduction.weights
    assert torch.equal(weights[:5], weights[5:10])


def test_build_orientation_plans_isotropic_mosaicity_is_not_bounded_by_sampling() -> None:
    # Unlike MosaicSmoothed, an isotropic sample count has nothing to do with rocking_curve_sampling
    # -- it composes its own extra tilts rather than smoothing the existing ones.
    rocking = RockingCurve(
        integration=IntegrationGeometry(semiangle=1.0, geometry="continuous_rotation"),
        sampling=2,
    )
    step = build_orientation_plans(rocking, IsotropicMosaicity(sigma_degrees=0.1, samples=50))
    assert step is not None  # construction alone must not raise


def test_isotropic_mosaicity_also_requires_rocking() -> None:
    with pytest.raises(ValueError, match="mosaicity requires"):
        build_orientation_plans(mosaicity=IsotropicMosaicity(sigma_degrees=0.1, samples=5))


def test_build_orientation_plans_builds_coupled_solve_geometry_before_alignment() -> None:
    plan, config, integration, mosaicity = _quartz_train_plan()

    built = build_orientation_plans(
        config.blochwave.to_rocking_curve(integration),
        mosaicity,
        coupling=config.blochwave.to_policy(),
    )(plan).orientations[0]

    observed = {tuple(row) for row in plan.orientations[0].pattern.hkl.tolist()}
    scored = {tuple(row) for row in built.alignment.hkl.tolist()}
    simulated = {tuple(row) for row in built.beam_hkl.tolist()}
    assert scored == observed & simulated
    assert (0, 0, 0) in simulated


def test_build_orientation_plans_workers_preserve_exact_coupled_geometry() -> None:
    """Parallel rotation construction is an execution choice, not a scientific input."""
    plan, config, integration, mosaicity = _quartz_train_plan()
    plan = replace(plan, orientations=plan.orientations * 3)
    rocking = config.blochwave.to_rocking_curve(integration)
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

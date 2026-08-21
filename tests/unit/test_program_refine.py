"""Output-writing behavior for the app-level refine path."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.unit.test_engine import _engine, _params

from diffBloch.app.program import (
    _mosaicity_components,
    _report_refinement_outcome,
    _thickness_networks,
    _write_refinement_outputs,
)
from diffBloch.config.manifest import read_refinement_lock
from diffBloch.config.schema import ExperimentConfig
from diffBloch.core.products import PatternBatch
from diffBloch.engine import (
    ModelRefinementResult,
    StructureFactorGrid,
    TrainableIsotropicMosaicity,
    build_refinement_model,
)
from diffBloch.io import read_structure
from diffBloch.io.record import ExperimentalRecord
from diffBloch.observability import (
    IsotropicMosaicityRefined,
    ObjectiveManifest,
    ObjectiveTerm,
    RecordingLogger,
    RefinementStep,
)
from diffBloch.preprocess import RefinementSetup, build_orientation_plans
from diffBloch.preprocess.plan import CandidatePlan, Plan
from diffBloch.specs import ApparentThicknessNetwork, IntegrationGeometry

_MINIMAL_CIF = """data_q
_cell_length_a 5
_cell_length_b 5
_cell_length_c 5
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_cell_volume 125
_symmetry_space_group_name_H-M 'P 1'
_symmetry_Int_Tables_number 1
loop_
_symmetry_equiv_pos_as_xyz
x,y,z
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_U_iso_or_equiv
_atom_site_thermal_displace_type
C1 C 0.1 0.2 0.3 1.0 0.02 Uiso
O1 O 0.4 0.5 0.6 1.0 0.03 Uani
loop_
_atom_site_aniso_label
_atom_site_aniso_U_11
_atom_site_aniso_U_22
_atom_site_aniso_U_33
_atom_site_aniso_U_23
_atom_site_aniso_U_13
_atom_site_aniso_U_12
O1 0.03 0.04 0.05 0.001 0.002 0.003
"""


def _record_with_alphas(alphas: tuple[float, ...]) -> ExperimentalRecord:
    n = len(alphas)
    return ExperimentalRecord(
        unit_cell=np.eye(3),
        cell_parameters=np.asarray([1.0, 1.0, 1.0, 90.0, 90.0, 90.0]),
        cell_parameters_su=np.full((6,), np.nan),
        wavelength=0.0251,
        ub_matrix=np.eye(3),
        zone_axis_ids=np.arange(1, n + 1),
        zone_axes=np.zeros((n, 3)),
        precession_angles=np.ones(n),
        alphas=np.asarray(alphas, dtype=np.float64),
        betas=np.zeros(n),
        omegas=np.zeros(n),
        scales=np.ones(n),
        hkl=np.asarray([[1, 0, 0]], dtype=np.int64),
        intensities=np.asarray([10.0]),
        sigmas=np.asarray([1.0]),
        reflection_zone_axis_ids=np.asarray([1]),
    )


def test_thickness_networks_partition_the_pooled_rotation_space() -> None:
    cfg = ExperimentConfig.model_validate(
        {
            "name": "q",
            "inputs": {
                "structure": "q.cif",
                "exp_data": ["a.cif_pets", "sub/b.cif_pets"],
                "multi_dataset": True,
            },
        }
    )
    records = (
        _record_with_alphas((0.0, 10.0)),
        _record_with_alphas((100.0, 110.0, 120.0)),
    )

    networks = _thickness_networks(cfg, records, ApparentThicknessNetwork())

    assert [network.key for network in networks] == [
        "apparent_thickness[a.cif_pets]",
        "apparent_thickness[sub/b.cif_pets]",
    ]
    assert [network.label for network in networks] == ["a.cif_pets", "sub/b.cif_pets"]
    assert [network.rotation_range for network in networks] == [(0, 2), (2, 5)]
    # Each dataset normalizes its own alpha span, so overlapping tilt ranges stay independent.
    assert networks[0].normalized_alphas == (-1.0, 1.0)
    assert networks[1].normalized_alphas == (-1.0, 0.0, 1.0)


def test_thickness_networks_treat_a_single_dataset_as_the_n_1_case() -> None:
    cfg = ExperimentConfig.model_validate(
        {"name": "q", "inputs": {"structure": "q.cif", "exp_data": "q.cif_pets"}}
    )
    records = (_record_with_alphas((0.0, 5.0, 10.0)),)

    networks = _thickness_networks(cfg, records, ApparentThicknessNetwork())

    assert len(networks) == 1
    assert networks[0].key == "apparent_thickness[q.cif_pets]"
    assert networks[0].label == "q.cif_pets"
    assert networks[0].rotation_range == (0, 3)
    assert networks[0].normalized_alphas == (-1.0, 0.0, 1.0)


def test_mosaicity_components_partition_the_pooled_rotation_space() -> None:
    cfg = ExperimentConfig.model_validate(
        {
            "name": "q",
            "inputs": {
                "structure": "q.cif",
                "exp_data": ["a.cif_pets", "sub/b.cif_pets"],
                "multi_dataset": True,
            },
            "blochwave": {"incoherent_mosaicity": True, "mosaicity_samples": 3},
            "refinement": {"trainable": {"mosaicity_sigma": True}},
        }
    )
    records = (
        _record_with_alphas((0.0, 10.0)).model_copy(update={"mosaicity_degrees": 0.02}),
        _record_with_alphas((100.0, 110.0, 120.0)).model_copy(update={"mosaicity_degrees": 0.03}),
    )
    integrations = (IntegrationGeometry(semiangle=1.0), IntegrationGeometry(semiangle=1.0))

    components = _mosaicity_components(cfg, records, integrations)

    assert [component.key for component in components] == [
        "isotropic_mosaicity[a.cif_pets]",
        "isotropic_mosaicity[sub/b.cif_pets]",
    ]
    assert [component.rotation_range for component in components] == [(0, 2), (2, 5)]
    assert [component.init_sigma_degrees for component in components] == [0.02, 0.03]
    assert len(components[0].polar_degrees) == 3


def test_mosaicity_components_is_empty_when_trainable_sigma_is_off() -> None:
    cfg = ExperimentConfig.model_validate(
        {
            "name": "q",
            "inputs": {"structure": "q.cif", "exp_data": "q.cif_pets"},
            "blochwave": {"incoherent_mosaicity": True},
        }
    )
    records = (_record_with_alphas((0.0,)).model_copy(update={"mosaicity_degrees": 0.02}),)
    integrations = (IntegrationGeometry(semiangle=1.0),)

    assert _mosaicity_components(cfg, records, integrations) == ()


def test_mosaicity_components_skips_a_dataset_with_zero_mosaicity() -> None:
    cfg = ExperimentConfig.model_validate(
        {
            "name": "q",
            "inputs": {"structure": "q.cif", "exp_data": "q.cif_pets"},
            "blochwave": {"incoherent_mosaicity": True},
            "refinement": {"trainable": {"mosaicity_sigma": True}},
        }
    )
    records = (_record_with_alphas((0.0,)).model_copy(update={"mosaicity_degrees": 0.0}),)
    integrations = (IntegrationGeometry(semiangle=1.0),)

    assert _mosaicity_components(cfg, records, integrations) == ()


def test_report_refinement_outcome_emits_the_refined_mosaicity_sigma() -> None:
    engine = _engine()  # its single orientation carries rotation_index 0
    structure_params = _params()
    component = TrainableIsotropicMosaicity(
        polar_degrees=(0.0,),
        init_sigma_degrees=0.02,
        rotation_range=(0, 1),
        key="isotropic_mosaicity[a.cif_pets]",
    )
    component_params = {
        component.key: component.initial_params(dtype=torch.float64, device=torch.device("cpu"))
    }
    model = build_refinement_model(
        initial=structure_params, components=(component,), component_params=component_params
    )
    result = ModelRefinementResult(
        model=model,
        losses=torch.zeros(1, dtype=torch.float64),
        best_model=model,
        best_step=0,
        artifacts={"refined_structure": "q/refined_structure.cif"},
    )
    logger = RecordingLogger()

    _report_refinement_outcome(
        logger,
        engine,
        result,
        validation_rotation_indices=frozenset(),
        thickness_nns=(),
        mosaicity_nns=(component,),
        raw_alphas=None,
    )

    [event] = [e for e in logger.events if isinstance(e, IsotropicMosaicityRefined)]
    assert event.label == "a.cif_pets"
    assert event.pets_sigma_degrees == 0.02
    assert event.sigma_degrees == pytest.approx(0.02)


def _refinement_result_for(
    tmp_path: Path,
    *,
    cell_parameters: np.ndarray | None = None,
    isotropic_displacements_only: bool = False,
) -> tuple[ExperimentConfig, RefinementSetup, ModelRefinementResult]:
    source = tmp_path / "q.cif"
    source.write_text(_MINIMAL_CIF)
    cfg = ExperimentConfig.model_validate(
        {
            "name": "q",
            "inputs": {
                "structure": "q.cif",
                "exp_data": "q.cif_pets",
                "isotropic_displacements_only": isotropic_displacements_only,
            },
        }
    )
    refinement = RefinementSetup.from_structure(
        read_structure(source),
        cell_parameters=cell_parameters,
        isotropic_displacements_only=isotropic_displacements_only,
    )
    params = replace(
        refinement.params,
        asu_positions=refinement.params.asu_positions
        + torch.tensor([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0]], dtype=torch.float64),
        occupancy_raw=torch.zeros(2, dtype=torch.float64),
    )
    model = build_refinement_model(initial=params)
    event = RefinementStep(
        iteration=0,
        loss=0.2,
        wr2=0.1,
        r_obs=0.05,
        diff_loss=0.2,
        objective_total=0.35,
        n_rotations=4,
        n_wr2_evaluated=3,
        n_r_obs_evaluated=2,
        components={
            "diffraction": {"raw": 0.2, "weight": 1.0, "contribution": 0.2},
            "bond_length": {"raw": 0.05, "weight": 3.0, "contribution": 0.15},
        },
    )
    result = ModelRefinementResult(
        model=model,
        losses=torch.tensor([0.2], dtype=torch.float64),
        best_model=model,
        best_step=0,
        history=(event,),
        reflection_counts={
            "matched": 12,
            "matched_i_gt_3sigma": 8,
            "matched_i_le_3sigma": 4,
            "unmatched_observed": 3,
        },
        objective_manifest=ObjectiveManifest(
            penalties=(ObjectiveTerm(name="bond_length", weight=3.0),)
        ),
    )
    return cfg, refinement, result


def test_write_refinement_outputs_persists_best_cif_and_params(tmp_path: Path) -> None:
    cfg, refinement, result = _refinement_result_for(tmp_path)

    written = _write_refinement_outputs(
        tmp_path, cfg, refinement, result, plan_lock_sha256s=("ab" * 32,)
    )

    assert set(written.artifacts) == {
        "refined_structure",
        "refined_parameters",
        "plan_q",
        "plan_lock_q",
        "refinement_lock",
    }
    refined = read_structure(tmp_path / "refined_structure.cif")
    assert np.allclose(refined.frac_positions, [[0.11, 0.2, 0.3], [0.4, 0.51, 0.6]])
    assert np.allclose(refined.occupancies, [0.5, 0.5])
    assert np.isfinite(refined.adp.u_iso[0])
    assert np.all(np.isfinite(refined.adp.uij_cif[1]))
    with np.load(tmp_path / "reproducibility" / "refined_parameters.npz") as params_file:
        assert params_file["asu_positions"].shape == (2, 3)
        assert params_file["occupancy_raw"].shape == (2,)
    assert not (tmp_path / "refinement_summary.json").exists()  # superseded by the .txt report


def test_write_refinement_outputs_strips_stale_aniso_row_when_forced_isotropic(
    tmp_path: Path,
) -> None:
    """``inputs.isotropic_displacements_only`` forces O1 (Uani in the CIF) onto Uiso; its
    ``_atom_site_aniso_*`` row must be removed, not left stale or refreshed with new anisotropic
    values -- otherwise re-reading the file classifies O1 back to Uani (io.cif._adp_for_site keys
    purely on aniso-row presence) and silently loses the override."""
    cfg, refinement, result = _refinement_result_for(tmp_path, isotropic_displacements_only=True)
    assert refinement.spec.adp_kind == ("Uiso", "Uiso")

    _write_refinement_outputs(tmp_path, cfg, refinement, result, plan_lock_sha256s=("ab" * 32,))

    text = (tmp_path / "refined_structure.cif").read_text()
    assert "_atom_site_aniso" not in text  # the whole now-empty aniso loop is gone

    refined = read_structure(tmp_path / "refined_structure.cif")
    assert refined.adp.kind == ("Uiso", "Uiso")
    assert np.all(np.isfinite(refined.adp.u_iso))
    assert np.all(np.isnan(refined.adp.uij_cif))


def test_write_refinement_outputs_rewrites_the_refined_cif_cell_header(tmp_path: Path) -> None:
    authoritative = np.array([6.0, 5.0, 5.0, 90.0, 90.0, 90.0], dtype=np.float64)
    cfg, refinement, result = _refinement_result_for(tmp_path, cell_parameters=authoritative)

    _write_refinement_outputs(tmp_path, cfg, refinement, result, plan_lock_sha256s=("ab" * 32,))

    refined = read_structure(tmp_path / "refined_structure.cif")
    np.testing.assert_allclose(refined.cell_parameters, authoritative)
    text = (tmp_path / "refined_structure.cif").read_text()
    assert "_cell_volume 150.00000" in text


def test_write_refinement_outputs_skips_the_lock_when_the_run_did_not_checkpoint(
    tmp_path: Path,
) -> None:
    """A leftover plan lock on disk is NOT this run's provenance -- no sha, no refinement.lock."""
    cfg, refinement, result = _refinement_result_for(tmp_path)
    (tmp_path / "reproducibility").mkdir()
    (tmp_path / "reproducibility" / "plan.q.lock").write_text("stale-lock-from-an-earlier-run")

    written = _write_refinement_outputs(tmp_path, cfg, refinement, result, plan_lock_sha256s=None)

    assert "refinement_lock" not in written.artifacts
    assert "plan_q" not in written.artifacts and "plan_lock_q" not in written.artifacts
    assert not (tmp_path / "reproducibility" / "refinement.lock").exists()


def test_write_refinement_outputs_chains_the_lock_to_this_runs_plan_lock_hashes(
    tmp_path: Path,
) -> None:
    cfg, refinement, result = _refinement_result_for(tmp_path)
    written = _write_refinement_outputs(
        tmp_path, cfg, refinement, result, plan_lock_sha256s=("cd" * 32,)
    )

    assert "refinement_lock" in written.artifacts
    lock = read_refinement_lock(tmp_path / "reproducibility" / "refinement.lock")
    assert lock.plan_lock_sha256s == ["cd" * 32]
    assert lock.refined_structure.path == "refined_structure.cif"
    assert lock.refined_parameters.path == "reproducibility/refined_parameters.npz"


def _built_plan_matching(cell: np.ndarray) -> Plan:
    """A one-rotation built ``Plan`` on ``cell`` -- geometry only, an arbitrary observed pattern."""
    grid = StructureFactorGrid.from_cell(cell, g_max=2.2)
    beams = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=np.int64)
    pattern = PatternBatch(
        hkl=torch.tensor(beams, dtype=torch.int64),
        intensities=torch.full((len(beams),), 10.0, dtype=torch.float64),
        sigmas=torch.ones(len(beams), dtype=torch.float64),
    )
    candidate = CandidatePlan.seed(beams, pattern, energy=200e3, thickness=(300.0,))
    seed = Plan(structure_factor_grid=grid, orientations=(candidate,))
    return build_orientation_plans()(seed)


def _built_plan_with_two_rotations(cell: np.ndarray) -> Plan:
    """Like :func:`_built_plan_matching`, but two rotations: index 0 (train) and 1 (validation)."""
    grid = StructureFactorGrid.from_cell(cell, g_max=2.2)
    beams = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=np.int64)
    candidates = tuple(
        CandidatePlan.seed(
            beams,
            PatternBatch(
                hkl=torch.tensor(beams, dtype=torch.int64),
                intensities=torch.full((len(beams),), 10.0, dtype=torch.float64),
                sigmas=torch.ones(len(beams), dtype=torch.float64),
                rotation_index=index,
            ),
            energy=200e3,
            thickness=(300.0,),
        )
        for index in range(2)
    )
    seed = Plan(structure_factor_grid=grid, orientations=candidates)
    return build_orientation_plans()(seed)


def test_write_refinement_outputs_writes_a_refinement_lock_from_a_relative_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: ``root`` is whatever the CLI's ``experiment_dir`` arg was -- often relative.

    ``structure_path``/``params_path`` are ``.resolve()``d (absolute); passing the unresolved
    (possibly relative) ``root`` straight to ``artifact_hash_for`` mismatched an absolute path
    against a relative base and raised ``ValueError: ... is not in the subpath of ...`` the moment
    a real run used a relative experiment directory (every CLI invocation does).
    """
    monkeypatch.chdir(tmp_path)
    relative_root = Path(".")
    cfg, refinement, result = _refinement_result_for(relative_root)
    written = _write_refinement_outputs(
        relative_root, cfg, refinement, result, plan_lock_sha256s=("cd" * 32,)
    )

    assert "refinement_lock" in written.artifacts
    lock_path = relative_root / "reproducibility" / "refinement.lock"
    assert read_refinement_lock(lock_path).refined_structure.path == "refined_structure.cif"

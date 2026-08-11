"""Output-writing behavior for the app-level refine path."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from diffBloch.app.program import _write_refinement_outputs
from diffBloch.config.manifest import read_refinement_lock
from diffBloch.config.schema import ExperimentConfig
from diffBloch.core.products import PatternBatch
from diffBloch.engine import (
    ModelRefinementResult,
    StructureFactorGrid,
    build_refinement_model,
)
from diffBloch.io import read_structure
from diffBloch.observability import ObjectiveManifest, ObjectiveTerm, RefinementStep
from diffBloch.preprocess import RefinementSetup, build_orientation_plans
from diffBloch.preprocess.plan import CandidatePlan, Plan

_MINIMAL_CIF = """data_q
_cell_length_a 5
_cell_length_b 5
_cell_length_c 5
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
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


def _refinement_result_for(
    tmp_path: Path,
) -> tuple[ExperimentConfig, RefinementSetup, ModelRefinementResult]:
    source = tmp_path / "q.cif"
    source.write_text(_MINIMAL_CIF)
    cfg = ExperimentConfig.model_validate(
        {"name": "q", "inputs": {"structure": "q.cif", "exp_data": "q.cif_pets"}}
    )
    refinement = RefinementSetup.from_structure(read_structure(source))
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

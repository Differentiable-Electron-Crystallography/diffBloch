"""Output-writing behavior for the app-level refine path."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from diffBloch.app.program import _write_refinement_outputs, _write_refinement_report
from diffBloch.config.manifest import read_refinement_lock
from diffBloch.config.schema import ExperimentConfig
from diffBloch.core.products import PatternBatch
from diffBloch.engine import (
    ApparentThicknessNN,
    ModelRefinementResult,
    StructureFactorGrid,
    ThicknessBounds,
    build_refinement_model,
)
from diffBloch.io import read_structure
from diffBloch.observability import RefinementStep
from diffBloch.preprocess import RefinementSetup, build_orientation_plans
from diffBloch.preprocess.plan import CandidatePlan, Plan
from diffBloch.preprocess.scoring import build_engine
from diffBloch.specs import IntegrationGeometry

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
    )
    return cfg, refinement, result


def test_write_refinement_outputs_persists_best_cif_params_and_summary(tmp_path: Path) -> None:
    cfg, refinement, result = _refinement_result_for(tmp_path)

    written = _write_refinement_outputs(tmp_path, cfg, refinement, result)

    assert set(written.artifacts) == {
        "refined_structure",
        "refined_parameters",
        "summary",
        "plan",
        "plan_lock",
    }
    refined = read_structure(tmp_path / "refined_structure.cif")
    assert np.allclose(refined.frac_positions, [[0.11, 0.2, 0.3], [0.4, 0.51, 0.6]])
    assert np.allclose(refined.occupancies, [0.5, 0.5])
    assert np.isfinite(refined.adp.u_iso[0])
    assert np.all(np.isfinite(refined.adp.uij_cif[1]))
    with np.load(tmp_path / "refined_parameters.npz") as params_file:
        assert params_file["asu_positions"].shape == (2, 3)
        assert params_file["occupancy_raw"].shape == (2,)
    summary = json.loads((tmp_path / "refinement_summary.json").read_text())
    assert summary["best_epoch"] == 0
    assert summary["total_hkl"] == "8 / 12"


def test_write_refinement_outputs_skips_the_lock_without_a_plan_lock(tmp_path: Path) -> None:
    cfg, refinement, result = _refinement_result_for(tmp_path)

    written = _write_refinement_outputs(tmp_path, cfg, refinement, result)

    assert "refinement_lock" not in written.artifacts
    assert not (tmp_path / "refinement.lock").exists()


def test_write_refinement_outputs_writes_a_refinement_lock_beside_an_existing_plan_lock(
    tmp_path: Path,
) -> None:
    cfg, refinement, result = _refinement_result_for(tmp_path)
    (tmp_path / "plan.lock").write_text("fake-plan-lock-bytes")

    written = _write_refinement_outputs(tmp_path, cfg, refinement, result)

    assert "refinement_lock" in written.artifacts
    lock = read_refinement_lock(tmp_path / "refinement.lock")
    assert lock.refined_structure.path == "refined_structure.cif"
    assert lock.refined_parameters.path == "refined_parameters.npz"


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


def test_write_refinement_report_without_a_thickness_nn(tmp_path: Path) -> None:
    cfg, refinement, result = _refinement_result_for(tmp_path)
    _write_refinement_outputs(tmp_path, cfg, refinement, result)
    plan = _built_plan_matching(np.eye(3, dtype=np.float64) * 5.0)
    engine = build_engine(plan, refinement)

    report_path = _write_refinement_report(
        tmp_path,
        cfg,
        IntegrationGeometry(semiangle=1.0),
        engine,
        result,
        elapsed_seconds=12.5,
    )

    assert report_path == tmp_path / "refinement_report.txt"
    text = report_path.read_text()
    assert "Simulation / refinement parameters" in text
    assert "Crystallographic parameters" in text
    assert "Per-rotation wR2 / R_obs" in text
    assert "Thickness NN" in text
    assert "enabled: no" in text
    assert "atom_site" in text
    assert "C1" in text and "O1" in text  # the CIF's own atom labels came through
    assert "12.5" in text  # elapsed time was reported


def test_write_refinement_report_with_a_thickness_nn(tmp_path: Path) -> None:
    cfg, refinement, result = _refinement_result_for(tmp_path)
    _write_refinement_outputs(tmp_path, cfg, refinement, result)
    plan = _built_plan_matching(np.eye(3, dtype=np.float64) * 5.0)
    engine = build_engine(plan, refinement)

    thickness_nn = ApparentThicknessNN(
        bounds=ThicknessBounds(min_angstrom=100.0, max_angstrom=1000.0),
        normalized_alphas=(0.0,),
    )
    dtype = refinement.params.asu_positions.dtype
    device = refinement.params.asu_positions.device
    model = build_refinement_model(
        initial=refinement.params,
        components=(thickness_nn,),
        component_params={
            thickness_nn.key: thickness_nn.initial_params(dtype=dtype, device=device)
        },
    )
    result_with_nn = replace(result, model=model, best_model=model)

    report_path = _write_refinement_report(
        tmp_path,
        cfg,
        IntegrationGeometry(semiangle=1.0),
        engine,
        result_with_nn,
        elapsed_seconds=1.0,
        thickness_nn=thickness_nn,
        raw_alphas=np.array([12.5], dtype=np.float64),
    )

    text = report_path.read_text()
    assert "enabled: yes" in text
    assert "alpha (degrees)" in text

    pytest.importorskip("matplotlib", reason="optional diffBloch[plot] extra")
    assert (tmp_path / "thickness_nn_shape.png").is_file()
    assert "plot: thickness_nn_shape.png" in text


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
    (relative_root / "plan.lock").write_text("fake-plan-lock-bytes")

    written = _write_refinement_outputs(relative_root, cfg, refinement, result)

    assert "refinement_lock" in written.artifacts
    assert read_refinement_lock(relative_root / "refinement.lock").refined_structure.path == (
        "refined_structure.cif"
    )

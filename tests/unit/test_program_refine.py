"""Output-writing behavior for the app-level refine path."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from diffBloch.app.program import _write_refinement_outputs
from diffBloch.config.manifest import read_refinement_lock
from diffBloch.config.schema import ExperimentConfig
from diffBloch.engine import ModelRefinementResult, build_refinement_model
from diffBloch.io import read_structure
from diffBloch.observability import RefinementStep
from diffBloch.preprocess import RefinementSetup

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

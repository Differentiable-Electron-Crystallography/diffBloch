from pathlib import Path

import numpy as np
import torch
from tests.unit.synthetic import make_constraint_spec

from diffBloch.config import load_experiment
from diffBloch.core import build_asu_expansion_plan
from diffBloch.core.crystal import reciprocal_cell
from diffBloch.io import read_experimental_data, read_structure
from diffBloch.params import RefinableParams, constrain

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures"


def test_read_abiraterone_observations_parses_a_large_pets_reflection_loop() -> None:
    # The loop_rows O(N) fix on the largest committed PETS loop: the quadratic parse this used to
    # trip is why loop_rows binds loop.values once. Kept on the biggest fixture available so a
    # regression shows up as a hang rather than passing unnoticed on a toy file.
    obs = read_experimental_data(
        FIXTURE_ROOT / "abiraterone_anchor" / "abiraterone_exp_data.cif_pets"
    )
    assert obs.n_rotations == 55
    assert obs.n_reflections == 10670


def test_quartz_experiment_lock_verifies_the_committed_inputs() -> None:
    # Enforce the hash-verification the fixture ships: load_experiment checks the structure /
    # exp_data against experiment.lock, so a fixture that drifts from its lock (e.g. a whitespace
    # hook silently rewriting the PETS file) fails loudly here, not silently at Tier 2.
    cfg, _lock = load_experiment(FIXTURE_ROOT / "quartz_anchor")
    assert cfg.name == "quartz-anchor"


def test_read_paracetamol_uiso_fixture() -> None:
    record = read_structure(FIXTURE_ROOT / "paracetamol_min" / "enantiomer_1.cif")

    assert record.n_atoms == 11
    assert record.labels == ("O1", "O2", "N1", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8")
    assert set(record.numbers.tolist()) == {6, 7, 8}
    assert record.adp.kind == ("Uiso",) * 11
    assert np.all(np.isfinite(record.adp.u_iso))
    assert np.all(record.adp.u_iso > 0.0)
    assert np.isnan(record.adp.uij_cif).all()
    assert record.spacegroup_hm == "P b c a"
    assert record.spacegroup_number == 61
    assert record.n_symops == 8

    plan = build_asu_expansion_plan(record.frac_positions, record.symops_R, record.symops_t)
    assert plan.n_asu_sites == 11
    assert plan.n_expanded_sites == 88


def test_constrain_accepts_paracetamol_uiso_adps() -> None:
    record = read_structure(FIXTURE_ROOT / "paracetamol_min" / "enantiomer_1.cif")
    positions = torch.tensor(record.frac_positions, dtype=torch.float64)
    u_iso_raw = torch.full((record.n_atoms,), -4.0, dtype=torch.float64, requires_grad=True)
    params = RefinableParams(
        asu_positions=positions,
        u_iso_raw=u_iso_raw,
    )
    spec = make_constraint_spec(
        occupancies=torch.tensor(record.occupancies, dtype=torch.float64),
        adp_kind=record.adp.kind,
        reciprocal_basis=torch.tensor(reciprocal_cell(record.unit_cell), dtype=torch.float64),
    )

    state = constrain(params, spec)
    state.uij_star.sum().backward()

    # Uiso -> U* = Uiso * G* (reciprocal metric): symmetric positive-semidefinite, generally with
    # non-zero off-diagonals once the cell is non-orthogonal.
    assert state.uij_star.shape == (record.n_atoms, 3, 3)
    assert torch.allclose(state.uij_star, state.uij_star.transpose(-1, -2))
    assert torch.all(torch.linalg.eigvalsh(state.uij_star) >= 0.0)
    assert u_iso_raw.grad is not None
    assert torch.all(u_iso_raw.grad > 0.0)

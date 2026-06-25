import pytest
import torch

from diffBloch.core import (
    apply_symmetry_mask,
    cholesky_adp,
    cholesky_raw_from_adp,
    equivalent_isotropic_adp,
    isotropic_adp,
    positive,
    unit_interval,
)
from diffBloch.params import ConstraintSpec, RefinableParams, constrain


def test_cholesky_adp_outputs_symmetric_psd_matrices() -> None:
    raw = torch.tensor(
        [
            [[0.2, 0.0, 0.0], [0.05, 0.15, 0.0], [0.01, 0.02, 0.18]],
            [[0.1, 0.03, 0.02], [0.0, 0.2, 0.01], [0.0, 0.0, 0.12]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    uij = cholesky_adp(raw)

    assert torch.allclose(uij, uij.transpose(-1, -2), atol=1e-12)
    assert torch.all(torch.linalg.eigvalsh(uij) >= -1e-12)
    uij.sum().backward()
    assert raw.grad is not None
    assert torch.any(raw.grad != 0)


def test_cholesky_adp_uses_only_lower_triangle() -> None:
    lower = torch.tensor(
        [[[0.2, 0.0, 0.0], [0.05, 0.15, 0.0], [0.01, 0.02, 0.18]]],
        dtype=torch.float64,
    )
    noisy_upper = lower.clone()
    noisy_upper[0, 0, 1] = 10.0
    noisy_upper[0, 0, 2] = -8.0
    noisy_upper[0, 1, 2] = 4.0

    assert torch.allclose(cholesky_adp(noisy_upper), cholesky_adp(lower))


def test_cholesky_raw_round_trips_initial_adp() -> None:
    uij = torch.tensor(
        [[[0.06, 0.01, 0.0], [0.01, 0.04, 0.0], [0.0, 0.0, 0.05]]],
        dtype=torch.float64,
    )

    raw = cholesky_raw_from_adp(uij)

    assert torch.allclose(cholesky_adp(raw), uij)


def test_equivalent_isotropic_adp_uses_trace_average() -> None:
    uij = torch.tensor(
        [[[0.06, 0.01, 0.0], [0.01, 0.04, 0.0], [0.0, 0.0, 0.05]]],
        dtype=torch.float64,
    )

    assert equivalent_isotropic_adp(uij).tolist() == pytest.approx([0.05])


def test_isotropic_adp_expands_to_scaled_identity() -> None:
    u_iso = torch.tensor([0.04, 0.05], dtype=torch.float64)

    uij = isotropic_adp(u_iso)

    assert uij.shape == (2, 3, 3)
    assert torch.allclose(uij[0], torch.eye(3, dtype=torch.float64) * 0.04)


def test_unit_interval_and_positive_bijectors() -> None:
    raw = torch.tensor([-10.0, 0.0, 10.0], dtype=torch.float64)

    occupancies = unit_interval(raw)
    assert torch.all((occupancies > 0.0) & (occupancies < 1.0))
    assert occupancies[1].item() == pytest.approx(0.5)
    assert torch.all(positive(raw) > 0.0)


def test_symmetry_mask_freezes_fixed_dofs_and_preserves_free_gradients() -> None:
    raw = torch.tensor([[0.2, 0.3, 0.4]], dtype=torch.float64, requires_grad=True)
    fixed = torch.tensor([[0.1, 0.1, 0.1]], dtype=torch.float64)
    mask = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float64)

    constrained = apply_symmetry_mask(raw, mask=mask, fixed=fixed)
    constrained.sum().backward()

    assert torch.allclose(constrained, torch.tensor([[0.2, 0.1, 0.4]], dtype=torch.float64))
    assert raw.grad is not None
    assert torch.allclose(raw.grad, torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float64))


def test_constrain_composes_raw_params_to_physical_state() -> None:
    positions = torch.tensor(
        [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]], dtype=torch.float64, requires_grad=True
    )
    uij_raw = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1).requires_grad_()
    occupancy_raw = torch.tensor([0.0, 2.0], dtype=torch.float64, requires_grad=True)
    thickness_raw = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    b_dose_raw = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    params = RefinableParams(
        asu_positions=positions,
        uij_raw=uij_raw,
        occupancy_raw=occupancy_raw,
        thickness_raw=thickness_raw,
        b_dose_raw=b_dose_raw,
    )
    spec = ConstraintSpec(
        fixed_positions=torch.zeros_like(positions),
        position_mask=torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=torch.float64),
        occupancies=torch.ones(2, dtype=torch.float64),
    )

    state = constrain(params, spec)
    loss = state.positions.sum() + state.uij_cif.sum() + state.occupancies.sum()
    loss.backward()

    assert torch.allclose(
        state.positions,
        torch.tensor([[0.2, 0.0, 0.4], [0.0, 0.6, 0.7]], dtype=torch.float64),
    )
    assert torch.all(torch.linalg.eigvalsh(state.uij_cif) >= 0.0)
    assert torch.all((state.occupancies > 0.0) & (state.occupancies < 1.0))
    assert state.thicknesses is not None and torch.all(state.thicknesses > 0.0)
    assert state.b_dose is not None and torch.all(state.b_dose > 0.0)
    assert positions.grad is not None
    assert torch.allclose(
        positions.grad,
        torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=torch.float64),
    )
    assert uij_raw.grad is not None
    assert occupancy_raw.grad is not None


def test_constrain_honors_mixed_adp_kinds() -> None:
    positions = torch.zeros((2, 3), dtype=torch.float64, requires_grad=True)
    uij_raw = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1).requires_grad_()
    u_iso_raw = torch.tensor([-3.0, -2.0], dtype=torch.float64, requires_grad=True)
    params = RefinableParams(
        asu_positions=positions,
        uij_raw=uij_raw,
        u_iso_raw=u_iso_raw,
    )
    spec = ConstraintSpec(
        fixed_positions=torch.zeros_like(positions),
        position_mask=torch.ones_like(positions),
        occupancies=torch.ones(2, dtype=torch.float64),
        adp_kind=("Uani", "Uiso"),
    )

    state = constrain(params, spec)
    state.uij_cif.sum().backward()

    assert torch.allclose(state.uij_cif[0], torch.eye(3, dtype=torch.float64))
    assert torch.allclose(
        state.uij_cif[1],
        torch.eye(3, dtype=torch.float64) * torch.nn.functional.softplus(u_iso_raw[1]),
    )
    assert uij_raw.grad is not None
    assert torch.any(uij_raw.grad[0] != 0)
    assert torch.all(uij_raw.grad[1] == 0)
    assert u_iso_raw.grad is not None
    assert u_iso_raw.grad[1] != 0
    assert u_iso_raw.grad[0] == 0


def test_constrain_rejects_missing_adps_until_policy_exists() -> None:
    positions = torch.zeros((1, 3), dtype=torch.float64)
    params = RefinableParams(asu_positions=positions)
    spec = ConstraintSpec(
        fixed_positions=torch.zeros_like(positions),
        position_mask=torch.ones_like(positions),
        occupancies=torch.ones(1, dtype=torch.float64),
        adp_kind=("missing",),
    )

    with pytest.raises(ValueError, match="missing ADPs"):
        constrain(params, spec)


def test_constrain_uses_default_occupancies_when_not_refined() -> None:
    positions = torch.zeros((1, 3), dtype=torch.float64)
    params = RefinableParams(
        asu_positions=positions,
        uij_raw=torch.eye(3, dtype=torch.float64)[None, :, :],
    )
    spec = ConstraintSpec(
        fixed_positions=positions,
        position_mask=torch.ones_like(positions),
        occupancies=torch.tensor([0.75], dtype=torch.float64),
    )

    state = constrain(params, spec)

    assert state.occupancies.tolist() == pytest.approx([0.75])


def test_constrain_validates_shapes() -> None:
    params = RefinableParams(
        asu_positions=torch.zeros((1, 3), dtype=torch.float64),
        uij_raw=torch.zeros((2, 3, 3), dtype=torch.float64),
    )
    spec = ConstraintSpec(
        fixed_positions=torch.zeros((1, 3), dtype=torch.float64),
        position_mask=torch.ones((1, 3), dtype=torch.float64),
        occupancies=torch.ones(1, dtype=torch.float64),
    )

    with pytest.raises(ValueError, match="uij_raw"):
        constrain(params, spec)

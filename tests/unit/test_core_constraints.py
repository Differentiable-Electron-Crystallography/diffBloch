import pytest
import torch
from tests.unit.synthetic import make_constraint_spec

from diffBloch.core import (
    apply_adp_constraints,
    apply_symmetry_projection,
    cartesian_adp_to_star,
    cholesky_adp,
    cholesky_raw_from_adp,
    cif_adp_to_star,
    diagonal_projection,
    equivalent_isotropic_adp,
    isotropic_adp,
    positive,
    unit_interval,
)
from diffBloch.params import ConstraintSpec, RefinableParams, constrain


def test_cif_adp_to_star_cubic_closed_form() -> None:
    # Cubic cell, edge a: d* = 1/a, so U*_ij = U_cif_ij / a^2.
    a = 5.0
    reciprocal_lengths = torch.full((3,), 1.0 / a, dtype=torch.float64)
    uij = torch.tensor(
        [[[0.10, 0.02, 0.01], [0.02, 0.12, 0.03], [0.01, 0.03, 0.14]]], dtype=torch.float64
    )
    star = cif_adp_to_star(uij, reciprocal_lengths)
    assert torch.allclose(star, uij / a**2)
    # identity cell is a no-op
    assert torch.allclose(cif_adp_to_star(uij, torch.ones(3, dtype=torch.float64)), uij)


def test_cartesian_adp_to_star_isotropic_gives_uiso_metric() -> None:
    # Orthorhombic diag(a, b, c): B = diag(1/a, 1/b, 1/c), so Uiso*I -> Uiso * diag(1/a^2, ...).
    a, b, c = 4.0, 5.0, 6.0
    reciprocal_basis = torch.diag(torch.tensor([1 / a, 1 / b, 1 / c], dtype=torch.float64))
    u_iso = 0.03
    uij_cart = isotropic_adp(torch.tensor(u_iso, dtype=torch.float64))
    star = cartesian_adp_to_star(uij_cart, reciprocal_basis)
    expected = torch.diag(
        torch.tensor([u_iso / a**2, u_iso / b**2, u_iso / c**2], dtype=torch.float64)
    )
    assert torch.allclose(star, expected)
    # U* = B U_cart B^T must stay symmetric for a general cell
    general = torch.tensor(
        [[0.9, 0.1, 0.05], [0.1, 1.1, 0.2], [0.05, 0.2, 1.3]], dtype=torch.float64
    )
    out = cartesian_adp_to_star(torch.eye(3, dtype=torch.float64) * 0.02, general)
    assert torch.allclose(out, out.T)


def test_constrain_requires_reciprocal_basis_for_adps() -> None:
    positions = torch.zeros((1, 3), dtype=torch.float64)
    params = RefinableParams(
        asu_positions=positions, uij_raw=torch.eye(3, dtype=torch.float64)[None]
    )
    spec = make_constraint_spec()
    with pytest.raises(ValueError, match="reciprocal_basis is required"):
        constrain(params, spec)


def test_constrain_coerces_reciprocal_basis_to_param_dtype() -> None:
    # float32 params with a float64 reciprocal_basis must yield float32 ADPs, not silently upcast.
    positions = torch.zeros((1, 3), dtype=torch.float32)
    params = RefinableParams(
        asu_positions=positions,
        u_iso_raw=torch.full((1,), -4.0, dtype=torch.float32),
    )
    spec = make_constraint_spec(
        adp_kind=("Uiso",),
        reciprocal_basis=torch.eye(3, dtype=torch.float64),
        dtype=torch.float32,
    )
    state = constrain(params, spec)
    assert state.uij_star.dtype == torch.float32


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


def test_symmetry_projection_freezes_fixed_dofs_and_preserves_free_gradients() -> None:
    raw = torch.tensor([[0.2, 0.3, 0.4]], dtype=torch.float64, requires_grad=True)
    fixed = torch.tensor([[0.1, 0.1, 0.1]], dtype=torch.float64)
    mask = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float64)
    projection, offset = diagonal_projection(mask, fixed)

    constrained = apply_symmetry_projection(raw, projection=projection, offset=offset)
    constrained.sum().backward()

    assert torch.allclose(constrained, torch.tensor([[0.2, 0.1, 0.4]], dtype=torch.float64))
    assert raw.grad is not None
    assert torch.allclose(raw.grad, torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float64))


def test_symmetry_projection_enforces_coupled_degrees_of_freedom() -> None:
    # A coupled site x = y = z (e.g. pyrite S on (x,x,x)): the projector onto span{(1,1,1)}
    # keeps the
    # three coordinates equal for any raw, and a diagonal mask cannot express this.
    raw = torch.tensor([[0.2, 0.4, 0.9]], dtype=torch.float64, requires_grad=True)
    projection = torch.full((1, 3, 3), 1.0 / 3.0, dtype=torch.float64)
    offset = torch.zeros((1, 3), dtype=torch.float64)

    constrained = apply_symmetry_projection(raw, projection=projection, offset=offset)
    constrained.sum().backward()

    mean = (0.2 + 0.4 + 0.9) / 3.0
    assert torch.allclose(constrained, torch.full((1, 3), mean, dtype=torch.float64))
    # The gradient of the mean is spread equally across the coupled coordinates.
    assert raw.grad is not None
    assert torch.allclose(raw.grad, torch.full((1, 3), 1.0, dtype=torch.float64))


def test_apply_adp_constraints_enforces_equalities_and_kills_dependent_gradients() -> None:
    # One atom on a site requiring U22 = U11 and U23 = U13/2; the other unconstrained.
    uij = torch.tensor(
        [
            [[0.10, 0.02, 0.03], [0.02, 0.99, 0.04], [0.03, 0.04, 0.12]],
            [[0.10, 0.02, 0.03], [0.02, 0.11, 0.04], [0.03, 0.04, 0.12]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    constraints = (((1, 1, 0, 0, 1.0), (1, 2, 0, 2, 0.5)), ())

    constrained = apply_adp_constraints(uij, constraints)
    constrained.sum().backward()

    # Constrained components follow their base component (symmetrically); the second atom is intact.
    assert constrained[0, 1, 1].item() == pytest.approx(constrained[0, 0, 0].item())  # U22 = U11
    assert constrained[0, 1, 2].item() == pytest.approx(0.5 * constrained[0, 0, 2].item())  # U23
    assert torch.allclose(constrained[0, 2, 1], constrained[0, 1, 2])  # stays symmetric
    assert torch.allclose(constrained[1], uij[1].detach())
    # The overwritten (dependent) raw entry no longer affects the output -> zero gradient.
    assert uij.grad is not None
    assert uij.grad[0, 1, 1] == 0.0


def test_constrain_composes_raw_params_to_physical_state() -> None:
    positions = torch.tensor(
        [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]], dtype=torch.float64, requires_grad=True
    )
    uij_raw = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1).requires_grad_()
    occupancy_raw = torch.tensor([0.0, 2.0], dtype=torch.float64, requires_grad=True)
    params = RefinableParams(
        asu_positions=positions,
        uij_raw=uij_raw,
        occupancy_raw=occupancy_raw,
    )
    projection, offset = diagonal_projection(
        torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=torch.float64),
        torch.zeros_like(positions),
    )
    spec = ConstraintSpec(
        position_projection=projection,
        position_offset=offset,
        occupancies=torch.ones(2, dtype=torch.float64),
        reciprocal_basis=torch.eye(3, dtype=torch.float64),
    )

    state = constrain(params, spec)
    loss = state.positions.sum() + state.uij_star.sum() + state.occupancies.sum()
    loss.backward()

    assert torch.allclose(
        state.positions,
        torch.tensor([[0.2, 0.0, 0.4], [0.0, 0.6, 0.7]], dtype=torch.float64),
    )
    assert torch.all(torch.linalg.eigvalsh(state.uij_star) >= 0.0)
    assert torch.all((state.occupancies > 0.0) & (state.occupancies < 1.0))
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
    spec = make_constraint_spec(
        n_atoms=2,
        adp_kind=("Uani", "Uiso"),
        reciprocal_basis=torch.eye(3, dtype=torch.float64),
    )

    state = constrain(params, spec)
    state.uij_star.sum().backward()

    assert torch.allclose(state.uij_star[0], torch.eye(3, dtype=torch.float64))
    assert torch.allclose(
        state.uij_star[1],
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
    spec = make_constraint_spec(adp_kind=("missing",))

    with pytest.raises(ValueError, match="missing ADPs"):
        constrain(params, spec)


def test_constrain_uses_default_occupancies_when_not_refined() -> None:
    positions = torch.zeros((1, 3), dtype=torch.float64)
    params = RefinableParams(
        asu_positions=positions,
        uij_raw=torch.eye(3, dtype=torch.float64)[None, :, :],
    )
    spec = make_constraint_spec(
        occupancies=torch.tensor([0.75], dtype=torch.float64),
        reciprocal_basis=torch.eye(3, dtype=torch.float64),
    )

    state = constrain(params, spec)

    assert state.occupancies.tolist() == pytest.approx([0.75])


def test_constrain_validates_shapes() -> None:
    params = RefinableParams(
        asu_positions=torch.zeros((1, 3), dtype=torch.float64),
        uij_raw=torch.zeros((2, 3, 3), dtype=torch.float64),
    )
    spec = make_constraint_spec()

    with pytest.raises(ValueError, match="uij_raw"):
        constrain(params, spec)

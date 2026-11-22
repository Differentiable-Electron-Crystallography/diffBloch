"""Native site-symmetry projector and ADP constraints (``io.symmetry_setup``).

The correctness proof for special-position constraints: the projector holds atoms on their sites
(including *coupled* sites a boolean mask cannot express), the ADP equalities are enforced in every
setting and origin choice, and the position projector agrees with diffpy used as an independent
oracle on the standard-setting fixtures.
"""

from dataclasses import replace
from pathlib import Path

import gemmi
import numpy as np
import pytest
import torch
from diffpy.structure.spacegroups import get_space_group
from diffpy.structure.symmetryutilities import SymmetryConstraints as DiffpyConstraints

from diffBloch.core.constraints import apply_adp_constraints
from diffBloch.io.cif import parse_structure_block, read_structure
from diffBloch.io.symmetry_setup import _site_stabilizer, symmetry_constraints
from diffBloch.params import constrain
from diffBloch.preprocess.experiment import RefinementSetup

FIXTURES = Path(__file__).parent.parent / "fixtures"
QUARTZ = FIXTURES / "quartz_anchor" / "enantiomer_1.cif"  # Si axis-aligned special, O general
PYRITE = FIXTURES / "pyrite" / "pyrite.cif"  # S on coupled (x,x,x), Fe fixed at (0,0,0)
# NOTE: the (x,x,z) and (0,y,y) coupling patterns are currently uncovered -- LTA (zeolite A,
# Pm-3m) carried them and was removed with its unused anchor fixture. See the test-fixture
# variety issue.


def _projectors(cif: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    record = read_structure(cif)
    constraints = symmetry_constraints(record)
    return constraints.position_projection, constraints.position_offset, list(record.labels)


@pytest.mark.parametrize("cif", [QUARTZ, PYRITE])
def test_projectors_are_idempotent_and_round_trip_cif_positions(cif: Path) -> None:
    record = read_structure(cif)
    projection, offset, _ = _projectors(cif)
    positions = np.asarray(record.frac_positions)
    for p, x0, off in zip(projection, positions, offset, strict=True):
        assert np.allclose(p @ p, p)  # a projection
        assert np.allclose(p @ x0 + off, x0)  # seeded raw = x0 stays on site


def test_general_position_is_unconstrained() -> None:
    projection, offset, labels = _projectors(QUARTZ)
    o = labels.index("O1")
    assert np.allclose(projection[o], np.eye(3))
    assert np.allclose(offset[o], 0.0)


def test_axis_aligned_special_frees_only_its_open_axis() -> None:
    # Quartz Si sits on (0, y, 5/6): x and z are frozen (all-zero projector rows), y is free.
    projection, _, labels = _projectors(QUARTZ)
    si = projection[labels.index("Si1")]
    assert np.allclose(si[0], 0.0)  # x frozen
    assert np.allclose(si[2], 0.0)  # z frozen
    assert not np.allclose(si[1], 0.0)  # y free
    assert np.linalg.matrix_rank(si, tol=1e-9) == 1


def test_coupled_special_position_enforces_equality_for_any_raw() -> None:
    # Pyrite S on (x, x, x): the projector keeps the three coordinates equal even when the raw
    # parameters disagree -- the case a per-coordinate boolean mask [1,1,1] cannot express.
    setup = RefinementSetup.from_structure(read_structure(PYRITE))
    raw = setup.params.asu_positions.clone()
    raw[1] = torch.tensor([0.30, 0.40, 0.55], dtype=torch.float64)  # x != y != z
    state = constrain(replace(setup.params, asu_positions=raw), setup.spec)

    sulfur = state.positions[1]
    assert torch.allclose(sulfur, sulfur.mean().expand(3))  # x == y == z (the coupled mean)
    assert torch.allclose(state.positions[0], torch.zeros(3, dtype=torch.float64))  # Fe fixed


def test_refinement_cannot_walk_a_coupled_atom_off_its_site() -> None:
    setup = RefinementSetup.from_structure(read_structure(PYRITE))
    raw = setup.params.asu_positions.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([raw], lr=0.05)
    for _ in range(25):
        optimizer.zero_grad()
        state = constrain(replace(setup.params, asu_positions=raw), setup.spec)
        # A loss that actively tries to prise S's coordinates apart.
        loss = -state.positions[1, 0] + (state.positions[1, 1] - 0.1) ** 2
        loss.backward()
        optimizer.step()

    state = constrain(replace(setup.params, asu_positions=raw), setup.spec)
    sulfur = state.positions[1]
    assert torch.allclose(sulfur, sulfur.mean().expand(3), atol=1e-9)  # still coupled
    assert torch.allclose(state.positions[0], torch.zeros(3, dtype=torch.float64))  # Fe still fixed


def test_adp_equalities_are_enforced_when_the_raw_violates_them() -> None:
    # Quartz Si requires U12 = U11/2 and U23 = U13/2; a hexagonal cell has a* = b*, so those ratios
    # survive unchanged into the U* frame, letting us assert them on the constrained output.
    setup = RefinementSetup.from_structure(read_structure(QUARTZ))
    # A raw Cholesky factor whose *unconstrained* Uij would break the site symmetry.
    uij_raw = setup.params.uij_raw.clone()
    uij_raw[0] = torch.tensor(
        [[0.20, 0.0, 0.0], [0.09, 0.15, 0.0], [0.07, 0.02, 0.18]], dtype=torch.float64
    )
    state = constrain(replace(setup.params, uij_raw=uij_raw), setup.spec)

    si = state.uij_star[0]
    assert si[0, 1] == pytest.approx(0.5 * si[0, 0])  # U12 = U11 / 2
    assert si[1, 2] == pytest.approx(0.5 * si[0, 2])  # U23 = U13 / 2


def test_coupled_adp_equalities_for_pyrite() -> None:
    # Pyrite is cubic (a* = b* = c*), so U11 = U22 = U33 and U12 = U13 = U23 survive into U*.
    setup = RefinementSetup.from_structure(read_structure(PYRITE))
    uij_raw = setup.params.uij_raw.clone()
    uij_raw[1] = torch.tensor(
        [[0.10, 0.0, 0.0], [0.03, 0.12, 0.0], [0.02, 0.05, 0.14]], dtype=torch.float64
    )
    state = constrain(replace(setup.params, uij_raw=uij_raw), setup.spec)

    s = state.uij_star[1]
    assert s[0, 0].item() == pytest.approx(s[1, 1].item())
    assert s[1, 1].item() == pytest.approx(s[2, 2].item())
    assert s[0, 1].item() == pytest.approx(s[0, 2].item())
    assert s[0, 2].item() == pytest.approx(s[1, 2].item())


# --- diffpy oracle: the native projector must agree with an independent symmetry engine ---

_PARAM = np.vectorize(lambda f: any(c in "xyz" for c in f))


def _diffpy_sites(cif: Path) -> list[tuple[list[bool], int]]:
    """Per atom: which coordinates are fixed-to-constant, and the number of free parameters."""
    record = read_structure(cif)
    space_group = get_space_group(record.spacegroup_hm)
    diffpy = DiffpyConstraints(space_group, positions=record.frac_positions.tolist())
    sites = []
    for equations in diffpy.poseqns:
        formulas = [equations["x"], equations["y"], equations["z"]]
        fixed = [not any(c in "xyz" for c in f) for f in formulas]
        free_params = {tok for f in formulas for tok in _free_tokens(f)}
        sites.append((fixed, len(free_params)))
    return sites


def _free_tokens(formula: str) -> set[str]:
    tokens, current = set(), ""
    for char in formula:
        if char in "xyz" or (current and char.isdigit()):
            current += char
        elif current:
            tokens.add(current)
            current = ""
    if current:
        tokens.add(current)
    return {t for t in tokens if t and t[0] in "xyz"}


@pytest.mark.parametrize("cif", [QUARTZ, PYRITE])
def test_native_projector_matches_diffpy_oracle(cif: Path) -> None:
    projection, _, _ = _projectors(cif)
    reference = _diffpy_sites(cif)
    assert len(reference) == len(projection)
    for p, (fixed_coords, n_free) in zip(projection, reference, strict=True):
        # A coordinate is fixed-to-constant iff its projector row is all zero.
        native_fixed = [bool(np.allclose(p[coord], 0.0)) for coord in range(3)]
        assert native_fixed == fixed_coords
        # The dimension of freedom (rank of P) must match diffpy's free-parameter count -- this is
        # what catches a coupled site as well as any CIF-symops vs diffpy setting/origin mismatch.
        assert np.linalg.matrix_rank(p, tol=1e-9) == n_free


# --- ADP constraints follow the CIF's own operators, in any setting or origin choice ---

_MONOCLINIC = (9.0, 6.0, 8.0, 90.0, 105.0, 90.0)
_HEXAGONAL = (5.0, 5.0, 12.0, 90.0, 90.0, 120.0)
_RHOMBOHEDRAL = (5.0, 5.0, 5.0, 75.0, 75.0, 75.0)
_CUBIC = (8.0, 8.0, 8.0, 90.0, 90.0, 90.0)
_TETRAGONAL = (4.0, 4.0, 6.0, 90.0, 90.0, 90.0)

# (operators, H-M name written in the CIF, IT number or None, cell, special site, free Uij count).
# Each name is one that diffpy's table misreads or does not know, so a name-keyed lookup either
# fails to load the structure or silently constrains Uij in the wrong setting.
_SETTINGS = [
    ("I 1 2/a 1", "I 2/a", 15, _MONOCLINIC, (0.25, 0.21, 0.0), 4),  # C2/c, other cell choice
    ("A 1 2/a 1", "A 2/a", 15, _MONOCLINIC, (0.25, 0.21, 0.0), 4),
    ("P 1 21/n 1", "P 21/n", None, _MONOCLINIC, (0.0, 0.0, 0.0), 6),  # loads without a number
    ("R -3 m:H", "R -3 m", 166, _HEXAGONAL, (0.0, 0.0, 0.21), 2),  # hexagonal axes, bare name
    ("R -3 m:R", "R -3 m", 166, _RHOMBOHEDRAL, (0.21, 0.21, 0.21), 2),  # rhombohedral axes
    ("F d -3 m:2", "F d -3 m :2", 227, _CUBIC, (0.5, 0.5, 0.5), 2),  # origin choice 2 (spinel)
    ("F d -3 m:2", "F d -3 m :2", 227, _CUBIC, (0.125, 0.125, 0.125), 1),
    ("P 4/n m m:2", "P 4/n m m :2", 129, _TETRAGONAL, (0.25, 0.25, 0.21), 2),  # origin choice 2
]


def _record_from_operators(setting, name, number, cell, site):
    operators = [op.triplet() for op in gemmi.find_spacegroup_by_name(setting).operations()]
    lines = [
        "data_setting",
        f"_symmetry_space_group_name_H-M '{name}'",
        *([f"_symmetry_Int_Tables_number {number}"] if number is not None else []),
        *(
            f"_cell_{key} {value}"
            for key, value in zip(
                ("length_a", "length_b", "length_c", "angle_alpha", "angle_beta", "angle_gamma"),
                cell,
                strict=True,
            )
        ),
        "loop_",
        "_symmetry_equiv_pos_as_xyz",
        *(f"'{op}'" for op in operators),
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_U_iso_or_equiv",
        f"Si1 Si {site[0]} {site[1]} {site[2]} 0.01",
    ]
    return parse_structure_block(gemmi.cif.read_string("\n".join(lines) + "\n").sole_block())


@pytest.mark.parametrize(("setting", "name", "number", "cell", "site", "n_free"), _SETTINGS)
def test_adp_constraints_hold_the_site_symmetry_in_every_setting(
    setting, name, number, cell, site, n_free
) -> None:
    record = _record_from_operators(setting, name, number, cell, site)
    (constraints,) = symmetry_constraints(record).adp_constraints
    stabilizer = _site_stabilizer(
        np.asarray(site, dtype=np.float64),
        np.asarray(record.symops_R),
        np.asarray(record.symops_t),
        symprec=1e-3,
    )
    d = np.diag(np.linalg.norm(np.linalg.pinv(np.asarray(record.unit_cell)).T, axis=1))
    assert 6 - len(constraints) == n_free

    # A general U, once constrained, is invariant under every site rotation (in the U* frame).
    a = np.random.default_rng(0).normal(size=(3, 3))
    u = a @ a.T * 0.01
    star = d @ apply_adp_constraints(torch.tensor(u[None]), (constraints,))[0].numpy() @ d
    for rotation in stabilizer:
        assert np.allclose(rotation @ star @ rotation.T, star, atol=1e-15)

    # A U that already has the site symmetry passes through unchanged (nothing over-constrained).
    symmetric_star = np.mean([r @ (d @ u @ d) @ r.T for r in stabilizer], axis=0)
    symmetric = np.linalg.inv(d) @ symmetric_star @ np.linalg.inv(d)
    constrained = apply_adp_constraints(torch.tensor(symmetric[None]), (constraints,))[0].numpy()
    assert np.allclose(constrained, symmetric, atol=1e-15)


def test_adp_constraints_ignore_the_space_group_name() -> None:
    # The operators decide the constraints. A missing or misleading name changes nothing.
    setting, _, number, cell, site, _ = _SETTINGS[0]
    reference = symmetry_constraints(_record_from_operators(setting, "I 2/a", number, cell, site))
    for name, label in (("", None), ("P 1", None), ("C 1 2/c 1", number)):
        other = symmetry_constraints(_record_from_operators(setting, name, label, cell, site))
        assert other.adp_constraints == reference.adp_constraints

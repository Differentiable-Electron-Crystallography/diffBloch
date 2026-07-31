"""Native site-symmetry projector + diffpy ADP constraints (``io.symmetry_setup``).

The correctness proof for special-position constraints: the projector holds atoms on their sites
(including *coupled* sites a boolean mask cannot express), the ADP equalities are enforced, and
both agree with diffpy used as an independent oracle (which also trips on any space-group
setting/origin mismatch between the CIF symops and diffpy).
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from diffpy.structure.spacegroups import get_space_group
from diffpy.structure.symmetryutilities import SymmetryConstraints as DiffpyConstraints

from diffBloch.io.cif import read_structure
from diffBloch.io.symmetry_setup import symmetry_constraints
from diffBloch.params import constrain
from diffBloch.preprocess.experiment import RefinementSetup

FIXTURES = Path(__file__).parent.parent / "fixtures"
QUARTZ = FIXTURES / "quartz_anchor" / "enantiomer_1.cif"  # Si axis-aligned special, O general
PYRITE = FIXTURES / "pyrite" / "pyrite.cif"  # S on coupled (x,x,x), Fe fixed at (0,0,0)
# LTA (zeolite A, Pm-3m): a real coupled material -- O2 on (x,x,z) [x=y coupled], O3 on (0,y,y)
# [y=z coupled]. The coupled special-position proof on real experimental data, not a synthetic.
LTA = FIXTURES / "lta_anchor" / "lta.cif"


def _projectors(cif: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    record = read_structure(cif)
    constraints = symmetry_constraints(record)
    return constraints.position_projection, constraints.position_offset, list(record.labels)


@pytest.mark.parametrize("cif", [QUARTZ, PYRITE, LTA])
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


def test_lta_coupled_oxygen_sites_hold_under_refinement() -> None:
    # The real-data coupled-special-position proof (Tier 1): LTA O2 on (x,x,z) [x=y coupled, z free]
    # and O3 on (0,y,y) [x fixed at 0, y=z coupled]. An optimizer actively prising the coupled
    # coordinates apart cannot walk either atom off its site -- the projector's image *is* the
    # allowed subspace, so every gradient step stays on the manifold (as for pyrite's (x,x,x), now
    # on a real material with two *distinct* couplings in one structure).
    record = read_structure(LTA)
    setup = RefinementSetup.from_structure(record)
    labels = list(record.labels)
    o2, o3 = labels.index("O2"), labels.index("O3")
    seed_o2_z = setup.params.asu_positions[o2, 2].item()  # O2's free axis, before refinement

    raw = setup.params.asu_positions.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([raw], lr=0.05)
    for _ in range(30):
        optimizer.zero_grad()
        state = constrain(replace(setup.params, asu_positions=raw), setup.spec)
        loss = (
            -state.positions[o2, 0]  # pull O2 x...
            + (state.positions[o2, 1] - 0.4) ** 2  # ...while prising O2 y away from it
            - state.positions[o2, 2]  # and drive O2 z, its free axis
            + (state.positions[o3, 0] - 0.2) ** 2  # push O3 x off its fixed 0
            - state.positions[o3, 2]  # pull O3 z away from y
        )
        loss.backward()
        optimizer.step()

    state = constrain(replace(setup.params, asu_positions=raw), setup.spec)
    o2p, o3p = state.positions[o2], state.positions[o3]
    assert torch.allclose(o2p[0], o2p[1], atol=1e-9)  # O2 stays on (x,x,z): x == y (coupling holds)
    assert not np.isclose(o2p[2].item(), seed_o2_z)  # ...but z is genuinely free -- it moved
    assert torch.allclose(o3p[0], torch.zeros((), dtype=torch.float64), atol=1e-9)  # O3: x fixed 0
    assert torch.allclose(o3p[1], o3p[2], atol=1e-9)  # O3 stays on (0,y,y): y == z


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


@pytest.mark.parametrize("cif", [QUARTZ, PYRITE, LTA])
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

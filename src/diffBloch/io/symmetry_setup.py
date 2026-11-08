"""Symmetry-constraint setup: extract special-position constraints at the io boundary.

Extracts special-position constraints from a validated :class:`StructureRecord`,
as plain-value arrays the torch-only :func:`diffBloch.params.constrain` then applies. The position
constraint is the **site-symmetry projector** built natively from the record's symmetry operators
(no diffpy on the position path); the diffpy-backed ADP (``Uij``) equality constraints are extracted
at this same boundary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from diffpy.structure.spacegroupmod import SpaceGroup
from diffpy.structure.spacegroups import get_space_group
from diffpy.structure.symmetryutilities import SymmetryConstraints as _DiffpyConstraints
from numpy.typing import NDArray

from diffBloch.core.constraints import AdpConstraint, AdpConstraints
from diffBloch.io.record import StructureRecord


@dataclass(frozen=True)
class SymmetryConstraints:
    """Special-position constraint data for a validated structure record.

    ``position_projection`` is the per-atom site-symmetry projector ``P`` (``(N, 3, 3)``) and
    ``position_offset`` the on-site offset ``(I - P) @ x0`` (``(N, 3)``); together they hold each
    atom on its special-position manifold via ``P @ raw + offset`` (see
    :func:`diffBloch.core.constraints.apply_symmetry_projection`). A general-position atom yields
    ``P = I`` (unconstrained). ``adp_constraints`` holds, per atom, the
    ``(i, j, src_i, src_j, coeff)`` ADP equalities ``Uij[i,j] = coeff * Uij[src_i, src_j]`` (empty
    for an unconstrained ADP). ``adp_site_average`` is the per-atom ``(N, 9, 9)`` Reynolds operator
    ``(1/|S|) * sum_{R in S} kron(R, R)`` acting on the row-major flattened ``U*`` (fractional
    displacement covariance, which transforms as ``R U* R^T``); it maps any ``U*`` onto the
    site-symmetry-invariant subspace and, being an average of congruences, keeps a positive-definite
    tensor positive-definite. The identity for a general-position atom.
    """

    n_asymmetric_sites: int
    n_symops: int
    position_projection: NDArray[np.float64]
    position_offset: NDArray[np.float64]
    adp_constraints: AdpConstraints
    adp_site_average: NDArray[np.float64]


def symmetry_constraints(record: StructureRecord, *, symprec: float = 1e-3) -> SymmetryConstraints:
    """Return special-position constraint data for a validated structure record.

    For each asymmetric-unit atom the site stabilizer ``S = {(R, t) : R @ x0 + t == x0 (mod 1)}`` is
    collected from the record's symmetry operators, and the projector onto the allowed-displacement
    subspace is the Reynolds average ``P = (1/|S|) * sum_{R in S} R`` with offset ``(I - P) @ x0``.
    ``P`` is idempotent and its image is exactly the free degrees of freedom, so
    ``P @ raw + offset`` keeps the atom on its site for any ``raw``. The projector is built from the
    record's symmetry operators (no diffpy); the ADP (``Uij``) equalities are extracted from diffpy
    ``SymmetryConstraints.Ueqns`` keyed on the record's H-M symbol.

    A CIF routinely rounds special coordinates (``0.3333`` for 1/3). The site is recognised within
    ``symprec``, then each atom is snapped onto the exact special position -- the average of its
    stabilizer images -- and that exact position feeds both the offset and diffpy (with the same
    ``symprec``). Without the snap, diffpy's much tighter default tolerance finds only part of the
    site group and returns incomplete ADP constraints (e.g. U13/U23 left free on a 3m site), while
    the position projector correctly treats the atom as special.
    """
    if symprec <= 0.0:
        raise ValueError("symprec must be positive")
    positions = np.asarray(record.frac_positions, dtype=np.float64)
    rotations = np.asarray(record.symops_R, dtype=np.float64)
    translations = np.asarray(record.symops_t, dtype=np.float64)

    identity = np.eye(3, dtype=np.float64)
    projection = np.empty((positions.shape[0], 3, 3), dtype=np.float64)
    offset = np.empty((positions.shape[0], 3), dtype=np.float64)
    adp_site_average = np.empty((positions.shape[0], 9, 9), dtype=np.float64)
    snapped = np.empty_like(positions)
    for index, position in enumerate(positions):
        stabilizer = _site_stabilizer(position, rotations, translations, symprec=symprec)
        snapped[index] = _snap_to_site(position, rotations, translations, symprec=symprec)
        projector = stabilizer.mean(axis=0)
        projection[index] = projector
        offset[index] = (identity - projector) @ snapped[index]
        adp_site_average[index] = np.mean([np.kron(r, r) for r in stabilizer], axis=0)

    return SymmetryConstraints(
        n_asymmetric_sites=record.n_atoms,
        n_symops=record.n_symops,
        position_projection=projection,
        position_offset=offset,
        adp_constraints=_adp_constraints(record, snapped, symprec=symprec),
        adp_site_average=adp_site_average,
    )


def general_position_mask(record: StructureRecord, *, symprec: float = 1e-3) -> NDArray[np.bool_]:
    """Per-atom mask, ``True`` where the atom is on a general position (trivial site stabilizer).

    An atom is on a *special* position when a non-identity symmetry operator fixes its site
    (mod 1) -- its stabilizer holds more than the identity, so some coordinates are
    symmetry-constrained. Callers that reparameterize positions freely (e.g. hydrogen riding) must
    not move such atoms, which would push them off their site-symmetry manifold.
    """
    if symprec <= 0.0:
        raise ValueError("symprec must be positive")
    positions = np.asarray(record.frac_positions, dtype=np.float64)
    rotations = np.asarray(record.symops_R, dtype=np.float64)
    translations = np.asarray(record.symops_t, dtype=np.float64)
    return np.array(
        [
            _site_stabilizer(position, rotations, translations, symprec=symprec).shape[0] == 1
            for position in positions
        ],
        dtype=np.bool_,
    )


def _site_stabilizer(
    position: NDArray[np.float64],
    rotations: NDArray[np.float64],
    translations: NDArray[np.float64],
    *,
    symprec: float,
) -> NDArray[np.float64]:
    """Return the rotation parts of the symmetry operators that fix ``position`` (mod 1)."""
    stabilizer: list[NDArray[np.float64]] = []
    for rotation, translation in zip(rotations, translations, strict=True):
        displacement = rotation @ position + translation - position
        displacement = displacement - np.round(displacement)  # wrap to (-1/2, 1/2]
        if np.all(np.abs(displacement) < symprec):
            stabilizer.append(rotation)
    # The identity operator always fixes the site, so the stabilizer is never empty.
    return np.stack(stabilizer)


def _snap_to_site(
    position: NDArray[np.float64],
    rotations: NDArray[np.float64],
    translations: NDArray[np.float64],
    *,
    symprec: float,
) -> NDArray[np.float64]:
    """The exact special position near ``position``: the mean of its stabilizer images.

    Each stabilizing operator maps ``position`` to within ``symprec`` of itself (mod 1); unwrapped
    back next to ``position``, those images average to the point the whole site group fixes
    exactly (e.g. ``0.3333 -> 1/3``). A general position has only the identity and is unchanged.
    """
    images: list[NDArray[np.float64]] = []
    for rotation, translation in zip(rotations, translations, strict=True):
        image = rotation @ position + translation
        displacement = image - position
        displacement = displacement - np.round(displacement)
        if np.all(np.abs(displacement) < symprec):
            images.append(position + displacement)
    snapped: NDArray[np.float64] = np.mean(images, axis=0)
    return snapped


# CIF anisotropic-ADP component -> (row, col) in the symmetric 3x3 Uij tensor.
_UIJ_COMPONENT = {
    "11": (0, 0),
    "22": (1, 1),
    "33": (2, 2),
    "12": (0, 1),
    "13": (0, 2),
    "23": (1, 2),
}
# One Ueqn term: optional sign, optional coefficient, then U + two component digits + atom-index.
_UEQN_TERM = re.compile(r"^([+-]?)(\d*\.?\d+)?\*?U(\d)(\d)\d*$")


def _adp_constraints(
    record: StructureRecord, positions: NDArray[np.float64], *, symprec: float
) -> AdpConstraints:
    """Extract per-atom ``Uij`` equality tuples from the space-group ADP constraints (diffpy)."""
    space_group = _space_group(record)
    diffpy_constraints = _DiffpyConstraints(space_group, positions=positions.tolist(), eps=symprec)
    return tuple(_atom_adp_constraints(equations) for equations in diffpy_constraints.Ueqns)


def _space_group(record: StructureRecord) -> SpaceGroup:
    """Resolve the diffpy space group, preferring the CIF H-M symbol, then the number."""
    try:
        return get_space_group(record.spacegroup_hm)
    except (ValueError, KeyError):
        if record.spacegroup_number is None:
            raise
        return get_space_group(record.spacegroup_number)


def _atom_adp_constraints(equations: dict[str, str]) -> tuple[AdpConstraint, ...]:
    """Turn one diffpy ``Ueqns`` dict into ``(i, j, src_i, src_j, coeff)`` equality tuples."""
    constraints: list[AdpConstraint] = []
    for key, (i, j) in _UIJ_COMPONENT.items():
        formula = equations["U" + key].replace(" ", "")
        constraint = _parse_ueqn(i, j, formula)
        if constraint is not None:
            constraints.append(constraint)
    return tuple(constraints)


def _parse_ueqn(i: int, j: int, formula: str) -> AdpConstraint | None:
    """Parse one ``Ueqns`` formula for component ``(i, j)`` into a constraint, or ``None`` if free.

    A pure constant (site symmetry gives homogeneous relations, so the constant is 0) zeroes the
    component; a self-reference with unit coefficient is a free parameter (no constraint); a single
    ``coeff * Ukl`` term becomes ``Uij[i,j] = coeff * Uij[src_i, src_j]``. Anything else (e.g. a
    multi-term sum) raises rather than being silently mishandled.
    """
    if _is_zero(formula):
        return (i, j, i, j, 0.0)  # Uij[i,j] = 0 * Uij[i,j] = 0
    match = _UEQN_TERM.fullmatch(formula)
    if match is None:
        raise ValueError(f"unsupported ADP constraint formula: {formula!r}")
    sign, magnitude, first, second = match.groups()
    coeff = (float(magnitude) if magnitude else 1.0) * (-1.0 if sign == "-" else 1.0)
    src_i, src_j = _UIJ_COMPONENT[first + second]
    if (src_i, src_j) == (i, j) and coeff == 1.0:
        return None  # the component is its own free parameter
    return (i, j, src_i, src_j, coeff)


def _is_zero(formula: str) -> bool:
    try:
        return float(formula) == 0.0
    except ValueError:
        return False

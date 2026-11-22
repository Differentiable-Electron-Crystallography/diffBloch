"""Symmetry-constraint setup: extract special-position constraints at the io boundary.

Extracts special-position constraints from a validated :class:`StructureRecord`,
as plain-value arrays the torch-only :func:`diffBloch.params.constrain` then applies. Both
constraints come from the site stabilizer of each atom under the record's own symmetry operators,
so they follow the setting and origin choice of the CIF. The position constraint is the
**site-symmetry projector**. The ADP (``Uij``) equalities are read off the same site group.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
    for an unconstrained ADP).
    """

    n_asymmetric_sites: int
    n_symops: int
    position_projection: NDArray[np.float64]
    position_offset: NDArray[np.float64]
    adp_constraints: AdpConstraints


def symmetry_constraints(record: StructureRecord, *, symprec: float = 1e-3) -> SymmetryConstraints:
    """Return special-position constraint data for a validated structure record.

    For each asymmetric-unit atom the site stabilizer ``S = {(R, t) : R @ x0 + t == x0 (mod 1)}`` is
    collected from the record's symmetry operators, and the projector onto the allowed-displacement
    subspace is the Reynolds average ``P = (1/|S|) * sum_{R in S} R`` with offset ``(I - P) @ x0``.
    ``P`` is idempotent and its image is exactly the free degrees of freedom, so
    ``P @ raw + offset`` keeps the atom on its site for any ``raw``. The projector is built from the
    record's symmetry operators. The ADP (``Uij``) equalities come from the same stabilizer (see
    :func:`_adp_equalities`), so they never depend on the H-M name or the space-group number.
    """
    if symprec <= 0.0:
        raise ValueError("symprec must be positive")
    positions = np.asarray(record.frac_positions, dtype=np.float64)
    rotations = np.asarray(record.symops_R, dtype=np.float64)
    translations = np.asarray(record.symops_t, dtype=np.float64)

    identity = np.eye(3, dtype=np.float64)
    projection = np.empty((positions.shape[0], 3, 3), dtype=np.float64)
    offset = np.empty((positions.shape[0], 3), dtype=np.float64)
    reciprocal_lengths = np.linalg.norm(
        np.linalg.pinv(np.asarray(record.unit_cell, dtype=np.float64)).T, axis=1
    )
    adp_constraints: list[tuple[AdpConstraint, ...]] = []
    for index, position in enumerate(positions):
        stabilizer = _site_stabilizer(position, rotations, translations, symprec=symprec)
        projector = stabilizer.mean(axis=0)
        projection[index] = projector
        offset[index] = (identity - projector) @ position
        adp_constraints.append(_adp_equalities(stabilizer, reciprocal_lengths))

    return SymmetryConstraints(
        n_asymmetric_sites=record.n_atoms,
        n_symops=record.n_symops,
        position_projection=projection,
        position_offset=offset,
        adp_constraints=tuple(adp_constraints),
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


def _adp_equalities(
    stabilizer: NDArray[np.float64], reciprocal_lengths: NDArray[np.float64]
) -> tuple[AdpConstraint, ...]:
    """Return the ``Uij`` equalities that the site group ``stabilizer`` imposes, in the CIF frame.

    A site rotation ``R`` maps ``U*`` to ``R @ U* @ R.T``, and the site symmetry requires that
    ``U*`` is unchanged by every rotation in the stabilizer. The mean of that map over the
    stabilizer is the projector onto the allowed ``Uij``. It is built here in the CIF frame,
    where ``U*_ij = d*_i d*_j Uij`` (see :func:`diffBloch.core.adp.cif_adp_to_star`).

    The reduced row echelon form of the allowed subspace chooses the free components in the order
    ``U11, U22, U33, U12, U13, U23``, so diagonal terms are free where possible, as in the CIF
    tables. Each other component becomes ``Uij[i,j] = coeff * Uij[src_i, src_j]`` with a free
    source, or ``coeff = 0`` when the site forces it to zero. Crystallographic site groups only
    give single-term relations. A relation with more than one term would mean a wrong stabilizer,
    so it raises rather than being silently mishandled.
    """
    scale = np.outer(reciprocal_lengths, reciprocal_lengths)
    columns = []
    for i, j in _UIJ_COMPONENTS:
        unit = np.zeros((3, 3), dtype=np.float64)
        unit[i, j] = unit[j, i] = 1.0
        star = unit * scale
        averaged = np.mean([rot @ star @ rot.T for rot in stabilizer], axis=0) / scale
        columns.append([averaged[a, b] for a, b in _UIJ_COMPONENTS])
    projector = np.asarray(columns, dtype=np.float64).T  # column k: image of component k

    reduced, pivots = _row_echelon(projector.T)
    constraints: list[AdpConstraint] = []
    for k, (i, j) in enumerate(_UIJ_COMPONENTS):
        if k in pivots:
            continue
        terms = [
            (row, reduced[row, k]) for row in range(len(pivots)) if abs(reduced[row, k]) > _TOL
        ]
        if not terms:
            constraints.append((i, j, i, j, 0.0))  # Uij[i,j] = 0
        elif len(terms) == 1:
            row, coeff = terms[0]
            src_i, src_j = _UIJ_COMPONENTS[pivots[row]]
            constraints.append((i, j, src_i, src_j, float(round(coeff, 12))))
        else:
            raise ValueError(f"site symmetry gives a multi-term Uij relation for U{i + 1}{j + 1}")
    return tuple(constraints)


def _row_echelon(matrix: NDArray[np.float64]) -> tuple[NDArray[np.float64], list[int]]:
    """Return the reduced row echelon form of ``matrix`` and its pivot columns, left to right."""
    reduced = np.array(matrix, dtype=np.float64)
    pivots: list[int] = []
    row = 0
    for col in range(reduced.shape[1]):
        if row == reduced.shape[0]:
            break
        best = row + int(np.argmax(np.abs(reduced[row:, col])))
        if abs(reduced[best, col]) < _TOL:
            continue
        reduced[[row, best]] = reduced[[best, row]]
        reduced[row] /= reduced[row, col]
        for other in range(reduced.shape[0]):
            if other != row:
                reduced[other] -= reduced[other, col] * reduced[row]
        pivots.append(col)
        row += 1
    return reduced[: len(pivots)], pivots


# CIF anisotropic-ADP components in CIF order, as (row, col) in the symmetric 3x3 Uij tensor.
_UIJ_COMPONENTS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
_TOL = 1e-9

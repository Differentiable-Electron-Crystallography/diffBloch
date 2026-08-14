"""Hard molecular constraints on the physical state.

A :class:`ConstraintTransform` is a hard *reparameterization* of the bounded
:class:`~diffBloch.params.PhysicalState`, distinct from a soft
:class:`~diffBloch.engine.penalties.BondLengthPenalty`: a constraint rewrites the state so an
invariant holds exactly at every optimizer step (e.g. deriving hydrogen positions from their
parents), rather than adding a cost the optimizer may trade against. Constraints are applied in the
refinement objective after ``constrain`` and before the diffraction term, so the diffraction *and*
the penalties see the transformed state.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor

from diffBloch.engine.chemistry import covalent_radius
from diffBloch.engine.refine import AtomSelection, TrainableSpec
from diffBloch.io.record import StructureRecord
from diffBloch.io.symmetry_setup import general_position_mask
from diffBloch.params import PhysicalState

__all__ = [
    "ConstraintTransform",
    "HydrogenRiding",
    "perceive_hydrogen_riding",
    "with_hydrogen_riding",
]


@runtime_checkable
class ConstraintTransform(Protocol):
    """A hard constraint applied to the physical state during refinement.

    ``name`` identifies the transform for deterministic ordering and duplicate rejection; ``apply``
    returns a *new* :class:`~diffBloch.params.PhysicalState` with the constraint enforced (no
    in-place mutation), so gradients flow through the reparameterization to the free parameters it
    depends on.

    ``name`` is a read-only member: constraints are immutable value-types (frozen dataclasses), so
    the protocol must not demand a *settable* attribute (a settable one excludes frozen
    implementations).
    """

    @property
    def name(self) -> str:
        """A stable identifier for deterministic ordering and duplicate rejection."""
        ...

    def apply(self, state: PhysicalState) -> PhysicalState:
        """Return a new physical state with this constraint enforced."""
        ...


@dataclasses.dataclass(frozen=True)
class HydrogenRiding:
    """Ride each hydrogen on its parent heavy atom -- a hard constraint on the physical state.

    Constant-offset riding: a hydrogen's position is its parent's position plus a fixed fractional
    ``offset`` taken from the input structure, and its displacement is ``u_iso_scale`` times the
    parent's ``uij_star``. Both are reparameterizations -- the H rows are overwritten so they carry
    no independent degree of freedom, and gradients flow through to the parent -- so hydrogens stay
    in the forward scattering while tracking the refined heavy-atom frame. Pair with
    ``TrainableSpec(positions=exclude_elements("H"), adp=exclude_elements("H"))`` so the H rows are
    never optimizer leaves.

    Limitations (the model deliberately kept simple; upgrades slot into this same transform):
    the fixed offset does not re-point H when the parent's local frame *rotates* (only translation
    is followed); the ``uij_star`` scale is exact for an isotropic parent (all-Uiso) and approximate
    for an anisotropic one.
    """

    name: str
    h_index: Tensor  # (H,) long -- riding-hydrogen ASU rows
    parent_index: Tensor  # (H,) long -- heavy-parent ASU row per hydrogen
    offset: Tensor  # (H, 3) fixed fractional parent->H vector
    u_iso_scale: Tensor  # (H,) per-hydrogen displacement multiplier

    def __post_init__(self) -> None:
        h = self.h_index
        if h.dtype != torch.int64 or self.parent_index.dtype != torch.int64:
            raise ValueError("h_index and parent_index must be int64 tensors")
        if h.ndim != 1 or self.parent_index.shape != h.shape:
            raise ValueError("h_index and parent_index must be 1-D tensors of equal length")
        if self.offset.shape != (h.shape[0], 3):
            raise ValueError("offset must have shape (H, 3)")
        if self.u_iso_scale.shape != h.shape:
            raise ValueError("u_iso_scale must have shape (H,)")
        if bool((h < 0).any()) or bool((self.parent_index < 0).any()):
            raise ValueError("h_index and parent_index must be non-negative")
        h_rows = h.tolist()
        if len(set(h_rows)) != len(h_rows):
            raise ValueError("h_index must not contain duplicate hydrogen rows")
        if set(h_rows) & set(self.parent_index.tolist()):
            raise ValueError("riding-hydrogen rows and parent rows must be disjoint")

    def apply(self, state: PhysicalState) -> PhysicalState:
        positions = state.positions
        h_index = self.h_index.to(positions.device)
        parent_index = self.parent_index.to(positions.device)
        offset = self.offset.to(dtype=positions.dtype, device=positions.device)
        derived = positions.index_select(0, parent_index) + offset.detach()
        new_positions = positions.index_copy(0, h_index, derived)

        uij = state.uij_star
        scale = self.u_iso_scale.to(dtype=uij.dtype, device=uij.device)
        parent_uij = uij.index_select(0, parent_index.to(uij.device))
        new_uij = uij.index_copy(0, h_index.to(uij.device), scale[:, None, None] * parent_uij)
        return dataclasses.replace(state, positions=new_positions, uij_star=new_uij)


def perceive_hydrogen_riding(
    structure: StructureRecord,
    *,
    cutoff_scale: float = 1.2,
    cutoff_margin_angstrom: float = 0.1,
    u_iso_scale: float = 1.2,
) -> HydrogenRiding | None:
    """Build a :class:`HydrogenRiding` from a structure: each H rides its nearest bonded heavy atom.

    For every hydrogen (atomic number 1) the parent is the *nearest* heavy atom within the covalent
    cutoff ``cutoff_scale * (r_H + r_heavy) + cutoff_margin_angstrom`` (Cartesian distance via the
    cell). The stored ``offset`` is the fractional parent->H vector from the input structure; the
    molecule is assumed ASU-contiguous, so no minimum-image wrapping is applied (matching the
    penalties layer).

    Riding is for **general-position hydrogens only**: it overwrites the H coordinate *after* the
    crystallographic projector, so a special-position H would be pushed off its site-symmetry
    manifold. A hydrogen on a special position is therefore rejected. Returns ``None`` when the
    structure has no hydrogens; raises when a hydrogen has no heavy neighbour within the cutoff.
    """
    if cutoff_scale <= 0:
        raise ValueError("cutoff_scale must be positive")
    if cutoff_margin_angstrom < 0:
        raise ValueError("cutoff_margin_angstrom must be non-negative")
    if u_iso_scale <= 0:
        raise ValueError("u_iso_scale must be positive")

    numbers = np.asarray(structure.numbers, dtype=np.int64)
    if not np.any(numbers == 1):
        return None
    frac = np.asarray(structure.frac_positions, dtype=np.float64)
    cart = np.asarray(frac @ structure.unit_cell, dtype=np.float64)
    general = general_position_mask(structure)

    h_rows: list[int] = []
    parent_rows: list[int] = []
    offsets: list[np.ndarray] = []
    for i in range(structure.n_atoms):
        if numbers[i] != 1:
            continue
        if not general[i]:
            raise ValueError(
                f"hydrogen {structure.labels[i]!r} (row {i}) is on a special position; riding is "
                "only supported for general-position hydrogens"
            )
        radius_h = covalent_radius(numbers[i])
        nearest: tuple[float, int] | None = None
        for j in range(structure.n_atoms):
            if numbers[j] == 1:
                continue
            distance = float(np.linalg.norm(cart[j] - cart[i]))
            cutoff = (
                cutoff_scale * (radius_h + covalent_radius(numbers[j])) + cutoff_margin_angstrom
            )
            if 1e-8 < distance <= cutoff and (nearest is None or distance < nearest[0]):
                nearest = (distance, j)
        if nearest is None:
            raise ValueError(
                f"hydrogen {structure.labels[i]!r} (row {i}) has no covalently-bonded heavy "
                "neighbour within the cutoff"
            )
        h_rows.append(i)
        parent_rows.append(nearest[1])
        offsets.append(frac[i] - frac[nearest[1]])

    return HydrogenRiding(
        name="hydrogen_riding",
        h_index=torch.tensor(h_rows, dtype=torch.int64),
        parent_index=torch.tensor(parent_rows, dtype=torch.int64),
        offset=torch.tensor(np.asarray(offsets), dtype=torch.float64),
        u_iso_scale=torch.full((len(h_rows),), u_iso_scale, dtype=torch.float64),
    )


def with_hydrogen_riding(
    structure: StructureRecord, trainable: TrainableSpec
) -> tuple[TrainableSpec, tuple[ConstraintTransform, ...]]:
    """Compose hydrogen riding onto a base trainable selection (Python/API scientific composition).

    Riding *derives* every hydrogen from its parent heavy atom each step -- position (constant
    parent->H offset) and Uiso (scaled from the parent) -- so the hydrogens must not also be
    optimizer leaves. This freezes them (excludes H from both ``positions`` and ``adp``) and
    perceives the :class:`HydrogenRiding` constraint from the structure geometry, returning the
    ``(trainable, constraints)`` pair to place on
    :class:`~diffBloch.engine.StructureComponent` via
    :func:`~diffBloch.engine.build_refinement_model`. This is expressed here in Python, not in
    config: it is scientific composition, not a stable default-path knob.

    When the structure has no hydrogens the constraint tuple is empty and the freeze is harmless
    (no H rows to exclude), so the call is safe to apply unconditionally.
    """
    frozen = dataclasses.replace(
        trainable,
        positions=_freeze_hydrogens(trainable.positions),
        adp=_freeze_hydrogens(trainable.adp),
    )
    riding = perceive_hydrogen_riding(structure)
    constraints: tuple[ConstraintTransform, ...] = () if riding is None else (riding,)
    return frozen, constraints


def _freeze_hydrogens(selection: AtomSelection) -> AtomSelection:
    """Add H to a selecting group's exclusion set, preserving any existing element filters.

    A group that trains nothing stays untouched. Existing include/exclude filters are kept -- an API
    caller's ``include_elements("C")`` stays C-only rather than broadening to every non-H atom -- H
    is merely added to the exclusion set. (A selection that *includes* H is contradictory with
    freezing H and is rejected at :class:`AtomSelection` construction.)
    """
    if not selection.selects_any:
        return selection
    return dataclasses.replace(
        selection, element_exclude=tuple(sorted({*selection.element_exclude, "H"}))
    )

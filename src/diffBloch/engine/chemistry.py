"""Shared chemistry constants for the engine (covalent radii for connectivity perception).

Both the soft bond penalties and the hard hydrogen-riding constraint perceive likely connectivity
from covalent radii, so the table and lookup live here rather than in one of them -- neither should
import an internal helper of the other.
"""

from __future__ import annotations

import numpy as np

__all__ = ["COVALENT_RADII_ANGSTROM", "covalent_radius"]

# Single-bond covalent radii (Angstrom), from Cordero et al., Dalton Trans. 2008,
# "Covalent radii revisited". A perception heuristic only; an explicit connectivity
# table can replace it.
#
# The inorganic entries are here because every example this repository ships needs
# one: quartz contains Si, borane contains B, and CsPbBr3 contains Cs and Pb. Without
# them covalent_radius raises and the bond-length restraint cannot run on any of them.
COVALENT_RADII_ANGSTROM: dict[int, float] = {
    1: 0.31,  # H
    5: 0.84,  # B
    6: 0.76,  # C
    7: 0.71,  # N
    8: 0.66,  # O
    9: 0.57,  # F
    14: 1.11,  # Si
    15: 1.07,  # P
    16: 1.05,  # S
    17: 1.02,  # Cl
    35: 1.20,  # Br
    53: 1.39,  # I
    55: 2.44,  # Cs
    82: 1.46,  # Pb
}


def covalent_radius(number: np.integer | int) -> float:
    """The single-bond covalent radius (Angstrom) for an atomic number; raise if unknown."""
    atomic_number = int(number)
    try:
        return COVALENT_RADII_ANGSTROM[atomic_number]
    except KeyError as exc:
        raise ValueError(f"no covalent radius for atomic number {atomic_number}") from exc

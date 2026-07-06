"""Regenerate ``orientation_oracle.npz`` — the quartz per-orientation excitation-error golden.

The GOLDEN is the *private* geometry path (``reciprocal_cell`` of the rotated real cell -> ``g`` ->
``excitation_errors``) evaluated on real quartz orientation matrices from ``optim_orientation.csv``
(co-located); the INPUT (cellpar, hkl, M) is shared so the public test
(``tests/unit/test_orientation_oracle.py``) can reconstruct it natively and pin its own geometry
against this golden. Oracle independence lives in the golden, so this script must run under a
``diffBloch_private`` checkout's venv (it imports the private implementation; it is NOT part of
this repo's test deps and is never imported by the tests)::

    cd <diffBloch_private checkout>
    .venv/bin/python <this repo>/tests/fixtures/quartz_anchor/generate_orientation_oracle.py

Writes ``orientation_oracle.npz`` + ``orientation_oracle_provenance.json`` next to this script.
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import numpy as np
from ase.geometry import cellpar_to_cell

# The *private* diffBloch package (same import name as this repo's; resolution comes from running
# under the private venv, where this repo's package is not installed).
from diffBloch.utils import excitation_errors, reciprocal_cell

FIXTURE = Path(__file__).resolve().parent
CSV = FIXTURE / "optim_orientation.csv"
OUT = FIXTURE / "orientation_oracle.npz"
PROV = FIXTURE / "orientation_oracle_provenance.json"

CELLPAR = [4.92260, 4.92260, 5.40030, 90.0, 90.0, 120.0]  # enantiomer_1.cif
ENERGY = 200e3  # eV (quartz config.yml)
G_MAX_REFINE = 1.6  # beam selection cutoff (untilted)
ROTATIONS = [10, 52, 78]  # first three rotation_idx in optim_orientation.csv


def load_orientation(rotation_idx: int) -> np.ndarray:
    with CSV.open() as f:
        for row in csv.DictReader(f):
            if int(row["Rotation Index"]) == rotation_idx:
                return np.array(json.loads(row["Orientation Matrix"]), dtype=np.float64)
    raise KeyError(rotation_idx)


def beam_hkl(recip: np.ndarray) -> np.ndarray:
    # Box wide enough to fully contain the g_max_refine sphere (|g|_max over -5..5 is ~2.23 > 1.6).
    rng = range(-5, 6)
    hkl = np.array(
        [[h, k, l] for h in rng for k in rng for l in rng],  # noqa: E741
        dtype=np.int64,
    )
    g = np.linalg.norm(hkl @ recip, axis=1)
    return hkl[g <= G_MAX_REFINE]


def main() -> None:
    cell = cellpar_to_cell(CELLPAR)
    recip = reciprocal_cell(cell)  # private, ASE-compatible (pinv(cell).T)
    hkl = beam_hkl(recip)

    sg = np.empty((len(ROTATIONS), len(hkl)), dtype=np.float64)
    rotated_recip = np.empty((len(ROTATIONS), 3, 3), dtype=np.float64)
    orientations = np.empty((len(ROTATIONS), 3, 3), dtype=np.float64)
    for i, idx in enumerate(ROTATIONS):
        m = load_orientation(idx)
        rr = reciprocal_cell(cell @ m.T)  # rotated real cell -> rotated reciprocal cell
        g = hkl @ rr  # private get_k
        sg[i] = excitation_errors(g, ENERGY)  # private Spence & Zuo, K along -z
        rotated_recip[i] = rr
        orientations[i] = m

    np.savez(
        OUT,
        cellpar=np.array(CELLPAR, dtype=np.float64),
        energy=np.float64(ENERGY),
        hkl=hkl,
        rotation_idx=np.array(ROTATIONS, dtype=np.int64),
        orientation=orientations,
        reciprocal_basis_untilted=recip,
        rotated_reciprocal_basis=rotated_recip,
        sg=sg,
    )
    PROV.write_text(
        json.dumps(
            {
                "fixture": "orientation_oracle.npz",
                "purpose": (
                    "Real-orientation excitation-error (Sg) golden: pins the native "
                    "per-orientation reciprocal-basis geometry (reciprocal_cell(cell @ M.T) -> g "
                    "-> Sg) against the private implementation on real quartz orientation "
                    "matrices. M is non-orthonormal (folds the measured-vs-ideal cell correction, "
                    "~1% anisotropic), so g0 @ M^-1 is faithful while g0 @ M.T is wrong by "
                    "~0.008 A^-1."
                ),
                "generated_by": (
                    "generate_orientation_oracle.py (co-located; run with the private venv)"
                ),
                "oracle_source": "diffBloch_private @ c865c3d66337f16c9f15b2adf2701416146d71be",
                "oracle_functions": {
                    "reciprocal_cell": "utils.py:35-58 (pinv(cell).T)",
                    "excitation_errors": "utils.py:261-300 (K along -z, Spence & Zuo)",
                },
                "input": {
                    "material": "alpha-quartz (enantiomer_1.cif)",
                    "cellpar": CELLPAR,
                    "energy_eV": ENERGY,
                    "g_max_refine": G_MAX_REFINE,
                    "rotation_idx": ROTATIONS,
                    "orientation_source": (
                        "optim_orientation.csv (orientation = R_goni . UB . B^-1)"
                    ),
                    "cell_convention": "ASE cellpar_to_cell == native cell_matrix_from_parameters",
                },
                "date": date.today().isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUT} : sg{sg.shape}, hkl{hkl.shape}, rotations {ROTATIONS}")
    print(f"Sg range: [{sg.min():.5f}, {sg.max():.5f}]")


if __name__ == "__main__":
    main()

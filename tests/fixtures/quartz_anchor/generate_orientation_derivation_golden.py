"""Regenerate ``orientation_derivation_golden.npz`` — the as-collected orientation golden.

Runs an independent as-collected orientation construction (``process_file`` ->
``generate_u_matrix`` + ``generate_crystal_orientations``) on the co-located quartz PETS anchor
(``exp_data.cif_pets``) and saves the resulting per-rotation orientation matrices as a golden. The
native ``preprocess`` derivation is pinned against it by
``tests/unit/test_orientation_derivation.py`` -- oracle independence lives here, not in the input.

Writes the ``.npz`` + provenance ``.json`` next to this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# An external diffBloch package (same import name as this repo's; not part of this repo's deps).
from diffBloch.rotation_dataset import (
    generate_crystal_orientations,
    generate_u_matrix,
    load_data,
)

ANCHOR = Path(__file__).resolve().parent
PETS = ANCHOR / "exp_data.cif_pets"
OUT_NPZ = ANCHOR / "orientation_derivation_golden.npz"
OUT_JSON = ANCHOR / "orientation_derivation_golden_provenance.json"


def main() -> None:
    cif_file = load_data(str(PETS))
    pets = cif_file["pets"]

    u_matrix = generate_u_matrix(cif_file)
    rotations, alphas = generate_crystal_orientations(cif_file, u_matrix)

    rotation_ids = np.array([int(x) for x in pets["_diffrn_zone_axis_id"]], dtype=np.int64)
    betas = np.array([float(x) for x in pets["_diffrn_zone_axis_beta"]], dtype=np.float64)
    omegas = np.array([float(x) for x in pets["_diffrn_zone_axis_omega"]], dtype=np.float64)
    ub = np.array(
        [[float(pets[f"_diffrn_orient_matrix_UB_{i}{j}"]) for j in (1, 2, 3)] for i in (1, 2, 3)],
        dtype=np.float64,
    )
    cell_params = np.array(
        [
            float(pets["_cell_length_a"]),
            float(pets["_cell_length_b"]),
            float(pets["_cell_length_c"]),
            float(pets["_cell_angle_alpha"]),
            float(pets["_cell_angle_beta"]),
            float(pets["_cell_angle_gamma"]),
        ],
        dtype=np.float64,
    )
    cell_volume = float(pets["_cell_volume"])

    matrices = np.stack([np.asarray(r, dtype=np.float64) for r in rotations])
    assert matrices.shape == (rotation_ids.shape[0], 3, 3)

    np.savez(
        OUT_NPZ,
        rotation_ids=rotation_ids,
        orientation_matrices=matrices,
        ub_matrix=ub,
        cell_params=cell_params,
        cell_volume=cell_volume,
        alphas=np.array(alphas, dtype=np.float64),
        betas=betas,
        omegas=omegas,
    )

    provenance = {
        "what": "as-collected orientation matrices for the quartz PETS anchor",
        "generated_by": "generate_orientation_derivation_golden.py (co-located)",
        "convention": "orientation = R_z(omega).R_x(alpha).R_y(beta) @ (UB @ B_inv); "
        "B = Busing-Levy from PETS cell params + volume; angles active, degrees",
        "input": PETS.name,
        "n_rotations": int(rotation_ids.shape[0]),
        "det_u_matrix": float(np.linalg.det(u_matrix)),
        "note": "geometry uses reciprocal_cell(cell @ orientation.T); orientation matrices are "
        "non-orthonormal (fold the ~1% measured-vs-ideal cell correction).",
    }
    OUT_JSON.write_text(json.dumps(provenance, indent=2) + "\n")

    print(
        f"wrote {OUT_NPZ} ({rotation_ids.shape[0]} rotations), det(U)={np.linalg.det(u_matrix):.5f}"
    )


if __name__ == "__main__":
    main()

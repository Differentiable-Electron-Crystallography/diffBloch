"""``import_orientations``: seed ``CandidatePlan`` orientations from an external CSV.

A ``Plan -> Plan`` step for bypassing the orientation search entirely with known-good orientation
matrices (e.g. from a prior reference run). Matches rows to candidates by
``pattern.rotation_index``; runs on the source-only ``CandidatePlan`` phase (before
``build_orientation_plans``), so no geometry is rebuilt here.
"""

from __future__ import annotations

import ast
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from diffBloch.preprocess.pipeline import PlanStep, as_step
from diffBloch.preprocess.plan import Plan, require_candidate_plans

__all__ = ["import_orientations"]


def _read_orientation_csv(path: Path) -> dict[int, NDArray[np.float64]]:
    """Parse a ``Rotation Index`` / ``Orientation Matrix`` CSV into ``{rotation_index: (3, 3)}``."""
    orientations: dict[int, NDArray[np.float64]] = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"Rotation Index", "Orientation Matrix"} <= set(
            reader.fieldnames
        ):
            raise ValueError(
                f"{path}: expected columns 'Rotation Index' and 'Orientation Matrix', "
                f"got {reader.fieldnames}"
            )
        for row in reader:
            idx = int(row["Rotation Index"])
            matrix = np.asarray(ast.literal_eval(row["Orientation Matrix"]), dtype=np.float64)
            if matrix.shape != (3, 3):
                raise ValueError(
                    f"{path}: rotation {idx}: orientation matrix must have shape (3, 3), "
                    f"got {matrix.shape}"
                )
            orientations[idx] = matrix  # last row wins for a duplicated rotation index
    return orientations


def import_orientations(csv_path: str | Path) -> PlanStep:
    """Return a step that overwrites every candidate's orientation from ``csv_path``.

    The CSV needs a ``Rotation Index`` column (matched against ``pattern.rotation_index``) and an
    ``Orientation Matrix`` column (a Python-literal 3x3 nested list per row) -- the format this
    repo's own orientation-search results use, and the format the private predecessor's
    ``optim_orientation*.csv`` exports use. Every rotation present in the plan must have a
    matching row; a rotation missing from the CSV raises rather than silently keeping the seed
    orientation. Compose this *before* the fitting steps in ``pipeline([...])`` -- typically in
    place of ``optimize_orientation`` (to run entirely on the imported orientations) or ahead of it (to
    seed the search from a better starting point than the native UB-derived orientation).
    """
    path = Path(csv_path)

    def run(plan: Plan) -> Plan:
        orientations = _read_orientation_csv(path)
        candidates = require_candidate_plans(plan)
        missing = sorted(
            {
                cp.pattern.rotation_index
                for cp in candidates
                if cp.pattern.rotation_index not in orientations
            }
        )
        if missing:
            raise ValueError(f"{path}: missing orientations for rotations {missing}")
        updated = tuple(
            replace(cp, orientation=orientations[cp.pattern.rotation_index]) for cp in candidates
        )
        return replace(plan, orientations=updated)

    return as_step("import_orientations", {"csv_path": str(path)}, run)

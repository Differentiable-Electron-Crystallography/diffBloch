"""``import_orientations``: seeding ``CandidatePlan`` orientations from an external CSV."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from tests.unit.synthetic import seed_system

from diffBloch.preprocess import import_orientations
from diffBloch.preprocess.plan import require_candidate_plans


def _write_csv(path: Path, rows: list[tuple[int, object]]) -> Path:
    csv_path = path / "orientations.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Rotation Index", "Orientation Matrix"])
        for index, matrix in rows:
            writer.writerow([index, repr(matrix)])
    return csv_path


_IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
_SWAP_XY = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]


def test_import_orientations_overwrites_the_matching_candidate(tmp_path: Path) -> None:
    _refinement, plan = seed_system()
    (candidate,) = require_candidate_plans(plan)
    assert candidate.pattern.rotation_index == 0

    csv_path = _write_csv(tmp_path, [(0, _SWAP_XY)])
    result = import_orientations(csv_path)(plan)

    (updated,) = require_candidate_plans(result)
    assert np.allclose(updated.orientation, _SWAP_XY)
    assert not np.allclose(candidate.orientation, _SWAP_XY)  # the input plan is untouched


def test_import_orientations_raises_for_a_missing_rotation(tmp_path: Path) -> None:
    _refinement, plan = seed_system()
    csv_path = _write_csv(tmp_path, [(7, _IDENTITY)])  # rotation 0 (the only candidate) absent

    with pytest.raises(ValueError, match="missing orientations for rotations \\[0\\]"):
        import_orientations(csv_path)(plan)


def test_import_orientations_last_duplicate_row_wins(tmp_path: Path) -> None:
    _refinement, plan = seed_system()
    csv_path = _write_csv(tmp_path, [(0, _IDENTITY), (0, _SWAP_XY)])

    result = import_orientations(csv_path)(plan)
    (updated,) = require_candidate_plans(result)
    assert np.allclose(updated.orientation, _SWAP_XY)


def test_import_orientations_rejects_a_non_3x3_matrix(tmp_path: Path) -> None:
    _refinement, plan = seed_system()
    csv_path = _write_csv(tmp_path, [(0, [[1.0, 0.0], [0.0, 1.0]])])

    with pytest.raises(ValueError, match=r"shape \(3, 3\)"):
        import_orientations(csv_path)(plan)


def test_import_orientations_rejects_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("Index,Matrix\n0,[[1,0,0],[0,1,0],[0,0,1]]\n")
    _refinement, plan = seed_system()

    with pytest.raises(ValueError, match="expected columns"):
        import_orientations(csv_path)(plan)


def test_import_orientations_records_the_csv_path_as_its_step_identity(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path, [(0, _IDENTITY)])
    step = import_orientations(csv_path)
    assert step.record.name == "import_orientations"
    assert step.record.params == {"csv_path": str(csv_path)}

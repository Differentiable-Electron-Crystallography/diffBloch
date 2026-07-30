"""Read saved per-rotation orientation matrices at the typed input boundary."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def read_orientation_matrices(path: str | Path, *, n_rotations: int) -> FloatArray:
    """Read a legacy orientation CSV into PETS rotation order.

    Rows are keyed by ``Rotation Index`` rather than CSV row order. Every PETS rotation must occur
    exactly once, and every ``Orientation Matrix`` must be a finite 3x3 value.
    """
    source = Path(path)
    by_index: dict[int, FloatArray] = {}
    with source.open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                rotation_index = int(row["Rotation Index"])
                matrix = np.asarray(json.loads(row["Orientation Matrix"]), dtype=np.float64)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{source}: invalid orientation CSV row") from exc
            if rotation_index in by_index:
                raise ValueError(f"{source}: duplicate Rotation Index {rotation_index}")
            if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
                raise ValueError(
                    f"{source}: Rotation Index {rotation_index} matrix must be finite shape (3, 3)"
                )
            by_index[rotation_index] = matrix

    expected = set(range(n_rotations))
    actual = set(by_index)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{source}: orientation indices must cover 0..{n_rotations - 1}; "
            f"missing={missing}, extra={extra}"
        )
    matrices = np.stack([by_index[index] for index in range(n_rotations)])
    matrices.setflags(write=False)
    return matrices

"""Forward inference on real materials, checked per rotation against stored references."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from diffBloch.config import load_experiment
from diffBloch.core.solver import SolverMethod
from diffBloch.io import read_experimental_data, read_structure
from diffBloch.preprocess import (
    build_orientation_plans,
    couple_beams,
    from_experiment,
    integrate_rocking_curve,
    pipeline,
    require_candidate_plans,
    run_inference,
    select_beams,
)

pytestmark = pytest.mark.e2e

INFERENCE_ROOT = Path(__file__).parent
PROJECT_ROOT = INFERENCE_ROOT.parents[2]
ALL_MATERIALS = ("quartz",)


class Mode(StrEnum):
    """Reference-file behavior for an e2e invocation."""

    COMPARE = "compare"
    GENERATE = "generate"


@dataclass(frozen=True)
class Case:
    """One material's self-contained e2e execution settings."""

    fixture: Path
    orientation_file: Path
    thickness: float
    seed: int
    compare_n_rotations: int
    rotation_indices: tuple[int, ...]
    solver: SolverMethod
    union_adaptive: bool
    atol: float


def _material_dir(material: str) -> Path:
    return INFERENCE_ROOT / material


def _load_case(material: str) -> Case:
    raw = yaml.safe_load((_material_dir(material) / "case.yaml").read_text())
    return Case(
        fixture=PROJECT_ROOT / raw["fixture"],
        orientation_file=PROJECT_ROOT / raw["orientation_file"],
        thickness=float(raw["thickness"]),
        seed=int(raw["seed"]),
        compare_n_rotations=int(raw["compare_n_rotations"]),
        rotation_indices=tuple(int(index) for index in raw["rotation_indices"]),
        solver=raw["solver"],
        union_adaptive=bool(raw["union_adaptive"]),
        atol=float(raw["atol"]),
    )


def _orientations(path: Path) -> dict[int, np.ndarray[Any, np.dtype[np.float64]]]:
    data = json.loads(path.read_text())
    return {int(index): np.asarray(matrix, dtype=np.float64) for index, matrix in data.items()}


def _run_material(material: str, *, count: int) -> dict[str, Any]:
    case = _load_case(material)
    cfg, _lock = load_experiment(case.fixture)
    structure = read_structure(
        case.fixture / cfg.inputs.structure,
        load_hydrogens=cfg.inputs.load_hydrogens,
    )
    experimental_data = read_experimental_data(case.fixture / cfg.inputs.exp_data)
    setup = from_experiment(structure, experimental_data, cfg)
    train = iter(require_candidate_plans(setup.plans.train))
    validation = iter(require_candidate_plans(setup.plans.validation))
    candidates = tuple(
        next(validation) if (index + 1) % 10 == 0 else next(train)
        for index in range(experimental_data.n_rotations)
    )
    if not 1 <= count <= len(case.rotation_indices):
        raise ValueError(f"rotation count must be in 1..{len(case.rotation_indices)}, got {count}")
    indices = case.rotation_indices[:count]
    optimized = _orientations(case.orientation_file)
    sampled = replace(
        setup.plans.combined,
        orientations=tuple(
            replace(
                candidates[index],
                orientation=optimized[index],
                thickness=np.asarray([case.thickness], dtype=np.float64),
            )
            for index in indices
        ),
    )
    prepare = pipeline(
        [
            select_beams(cfg.blochwave.to_beam_selection(setup.integration)),
            build_orientation_plans(),
            integrate_rocking_curve(cfg.blochwave.to_rocking_curve(setup.integration)),
            couple_beams(replace(cfg.blochwave.to_policy(), union_adaptive=case.union_adaptive)),
        ]
    )

    started = time.perf_counter()
    result = run_inference(
        sampled,
        setup.refinement,
        prepare=prepare,
        method=case.solver,
    )
    elapsed = time.perf_counter() - started

    rotations = [
        {
            "rotation_index": index,
            "r_obs": row.r_obs,
            "wr2": row.wr2,
            "n_observed": row.n_observed,
            "n_beams": row.n_beams,
        }
        for index, row in zip(indices, result.per_rotation, strict=True)
    ]
    return {
        "seed": case.seed,
        "solver": case.solver,
        "n_rotations": len(rotations),
        "wall_time_seconds": elapsed,
        "mean_r_obs": result.mean_r_obs,
        "mean_wr2": result.mean_wr2,
        "rotations": rotations,
    }


def _write_anchor_metrics(material: str, actual: dict[str, Any]) -> None:
    """Append this material's run-level means to ``$DIFFBLOCH_ANCHOR_METRICS``, if set.

    The file CI reads to append a row to the published anchor history. Written *before* the
    comparison, deliberately: a drifted anchor is exactly the run whose measurement the trend needs
    to show, so recording it must not depend on the assertion passing.

    Keys are ``{material}_mean_r_obs`` / ``{material}_mean_wr2``. Unset variable means no file, so
    an ordinary local run writes nothing.
    """
    destination = os.environ.get("DIFFBLOCH_ANCHOR_METRICS")
    if not destination:
        return
    path = Path(destination)
    metrics: dict[str, Any] = {}
    if path.exists():  # several materials append to one file
        metrics = json.loads(path.read_text())
    metrics[f"{material}_mean_r_obs"] = actual["mean_r_obs"]
    metrics[f"{material}_mean_wr2"] = actual["mean_wr2"]
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def _write_reference(material: str, actual: dict[str, Any]) -> None:
    path = _material_dir(material) / "results.json"
    path.write_text(json.dumps(actual, indent=2) + "\n")


def _compare(material: str, actual: dict[str, Any]) -> None:
    case = _load_case(material)
    reference = json.loads((_material_dir(material) / "results.json").read_text())
    expected = {row["rotation_index"]: row for row in reference["rotations"]}

    assert actual["seed"] == reference["seed"]
    assert actual["solver"] == reference["solver"]
    assert actual["n_rotations"] <= reference["n_rotations"]
    for row in actual["rotations"]:
        index = row["rotation_index"]
        assert index in expected, f"{material}: rotation {index} missing from reference"
        pinned = expected[index]
        assert row["r_obs"] == pytest.approx(pinned["r_obs"], abs=case.atol)
        assert row["n_observed"] == pinned["n_observed"]
        assert row["n_beams"] == pinned["n_beams"]


@pytest.mark.parametrize("material", ALL_MATERIALS)
def test_inference(material: str) -> None:
    """Run one material and compare or generate its per-rotation reference."""
    mode = Mode(os.environ.get("E2E_MODE", Mode.COMPARE))
    case = _load_case(material)
    count = int(os.environ.get("E2E_N_ROTATIONS", case.compare_n_rotations))
    actual = _run_material(material, count=count)

    assert actual["n_rotations"] > 0
    assert all(math.isfinite(row["r_obs"]) for row in actual["rotations"])

    _write_anchor_metrics(material, actual)

    if mode is Mode.GENERATE:
        _write_reference(material, actual)
    else:
        _compare(material, actual)

"""Serialize a preprocessed ``Plan`` to a portable ``.npz`` checkpoint and read it back.

The persistence primitive behind the ``run`` program's checkpoint/resume: *serialize the whole
``Plan``*, never per-facet state. It
exploits the plan types' own design -- both :class:`~diffBloch.engine.plan.OrientationPlan` and
:class:`~diffBloch.engine.plan.CoupledOrientationPlan` separate their **source / rebuild inputs**
(orientation / tilts / thickness / beam set(s) / observed ``pattern`` / ``energy`` / ``u0`` /
``tilt_reduction``, plus the segmented plan's per-chunk ``(union_hkl, covered_tilt_indices)`` and
pinned scored set) from their **built geometry** (``beam_plans`` incl. the heavy
``StructureFactorGather``, ``alignment``, the union + ``union_beam_index``), and
``StructureFactorGrid.from_cell`` + ``.build`` rebuild
the built parts from the source. So we persist only the source and rebuild the derived geometry
on read -- a stored gather is a pure function of the beam set + grid and could only desync.

Format: one ``.npz`` (numpy is already a core dependency; portable, non-pickle hence auditable, and
it holds the ragged per-rotation/per-segment arrays natively as index-keyed entries). Scalars,
structure, the per-orientation ``kind`` discriminant + ``tilt_reduction`` discriminant, and the
plan's ``provenance`` (the recipe that produced it) ride in a JSON string stored as the reserved
``__meta__`` array, so the file is self-describing. Read uses ``allow_pickle=False``: a checkpoint
is data, never code.

This lives in ``preprocess`` (where ``Plan`` lives), not ``io`` -- ``io`` is the input-record layer
*below* ``core`` / ``engine`` / ``preprocess``, so a ``Plan`` serializer there would invert the
dependency. The ``plan.lock`` provenance that binds a checkpoint to the inputs + config + software
version that produced it is a separate, data-free concern in ``config.manifest``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from diffBloch.core.products import MosaicSmoothed, PatternBatch, PlainSum, TiltReduction
from diffBloch.engine.plan import (
    CoupledOrientationPlan,
    OrientationPlan,
    OrientationPlanLike,
    StructureFactorGrid,
)
from diffBloch.preprocess.pipeline import StepRecord
from diffBloch.preprocess.plan import Plan, require_built_plans

__all__ = ["plan_is_readable", "read_plan", "write_plan"]

_FORMAT_VERSION = 6


def write_plan(plan: Plan, path: str | Path) -> None:
    """Write ``plan`` to ``path`` as a portable ``.npz`` checkpoint (source arrays + JSON meta)."""
    arrays: dict[str, np.ndarray] = {"cell": _numpy(plan.structure_factor_grid.cell)}
    per_rotation: list[dict[str, Any]] = []
    for i, op in enumerate(require_built_plans(plan)):
        arrays[f"orient_{i}"] = _numpy(op.orientation)
        arrays[f"tilts_{i}"] = _numpy(op.tilts)
        arrays[f"thickness_{i}"] = _numpy(op.thickness)
        arrays[f"pat_hkl_{i}"] = _numpy(op.pattern.hkl)
        arrays[f"pat_int_{i}"] = _numpy(op.pattern.intensities)
        arrays[f"pat_sig_{i}"] = _numpy(op.pattern.sigmas)
        entry: dict[str, Any] = {
            "energy": op.energy,
            "u0": op.u0,
            "rotation_index": op.pattern.rotation_index,
            "dataset": op.pattern.dataset,
            "tilt_reduction": _dump_reduction(op.tilt_reduction),
        }
        if isinstance(op, CoupledOrientationPlan):
            entry["kind"] = "segmented"
            entry["n_segments"] = len(op.segments)
            arrays[f"scored_hkl_{i}"] = _numpy(op.alignment.hkl)  # the pinned scored set
            for j, segment in enumerate(op.segments):
                arrays[f"union_hkl_{i}_{j}"] = _numpy(segment.plan.beam_hkl)
                arrays[f"covered_tilt_indices_{i}_{j}"] = _numpy(segment.cover)
        else:
            entry["kind"] = "plain"
            arrays[f"beam_hkl_{i}"] = _numpy(op.beam_hkl)
        per_rotation.append(entry)
    meta = {
        "format_version": _FORMAT_VERSION,
        "g_max": plan.structure_factor_grid.g_max,
        "n_orientations": len(plan.orientations),
        "orientations": per_rotation,
        "provenance": [{"name": r.name, "params": r.params} for r in plan.provenance],
    }
    arrays["__meta__"] = np.array(json.dumps(meta))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as file:
        np.savez_compressed(file, **arrays)  # type: ignore[arg-type]


def plan_is_readable(path: str | Path) -> bool:
    """Whether :func:`read_plan` can read ``path``'s checkpoint format (and the file at all).

    ``read_plan`` *raises* on a format it cannot read, but the checkpoint reuse gate
    (:func:`~diffBloch.config.preprocess_lock_status`) keys on the release ``__version__``, not on
    this format -- so a checkpoint written by an earlier format within the same release still passes
    the lock and would then blow up inside ``read_plan``. Callers ask this first and treat ``False``
    as *stale* (recompute + rewrite), which is what a checkpoint this build cannot read actually is.
    A corrupt or truncated ``.npz`` is ``False`` for the same reason. Reads only the JSON metadata
    array, never the geometry, so it stays cheap enough to call on every checkpoint hit.
    """
    try:
        with np.load(Path(path), allow_pickle=False) as data:
            meta = json.loads(str(data["__meta__"].item()))
        return int(meta["format_version"]) == _FORMAT_VERSION
    except (OSError, ValueError, KeyError, TypeError):
        return False


def read_plan(path: str | Path) -> Plan:
    """Read a ``.npz`` checkpoint from :func:`write_plan`, rebuilding the built geometry."""
    with np.load(Path(path), allow_pickle=False) as data:
        meta = json.loads(str(data["__meta__"].item()))
        if meta["format_version"] != _FORMAT_VERSION:
            raise ValueError(
                f"unsupported plan checkpoint format {meta['format_version']!r}: "
                f"expected {_FORMAT_VERSION}"
            )
        grid = StructureFactorGrid.from_cell(np.asarray(data["cell"]), g_max=float(meta["g_max"]))
        orientations = tuple(
            _read_orientation(data, grid, i, entry) for i, entry in enumerate(meta["orientations"])
        )
        provenance = tuple(
            StepRecord(name=e["name"], params=e["params"]) for e in meta["provenance"]
        )
    return Plan(structure_factor_grid=grid, orientations=orientations, provenance=provenance)


def _read_orientation(
    data: Any, grid: StructureFactorGrid, i: int, entry: dict[str, Any]
) -> OrientationPlanLike:
    pattern = PatternBatch(
        hkl=torch.as_tensor(data[f"pat_hkl_{i}"]),
        intensities=torch.as_tensor(data[f"pat_int_{i}"]),
        sigmas=torch.as_tensor(data[f"pat_sig_{i}"]),
        rotation_index=int(entry["rotation_index"]),
        dataset=str(entry["dataset"]),
    )
    energy = float(entry["energy"])
    thickness = torch.as_tensor(data[f"thickness_{i}"])
    u0 = float(entry["u0"])
    orientation = np.asarray(data[f"orient_{i}"])
    tilts = np.asarray(data[f"tilts_{i}"])
    reduction = _load_reduction(entry["tilt_reduction"])
    if entry["kind"] == "segmented":
        segments = [
            (
                np.asarray(data[f"union_hkl_{i}_{j}"]),
                tuple(int(c) for c in data[f"covered_tilt_indices_{i}_{j}"]),
            )
            for j in range(int(entry["n_segments"]))
        ]
        return CoupledOrientationPlan.build(
            grid,
            segments,
            pattern,
            energy=energy,
            thickness=thickness,
            u0=u0,
            orientation=orientation,
            tilts=tilts,
            tilt_reduction=reduction,
            scored_hkl=np.asarray(data[f"scored_hkl_{i}"]),
        )
    return OrientationPlan.build(
        grid,
        np.asarray(data[f"beam_hkl_{i}"]),
        pattern,
        energy=energy,
        thickness=thickness,
        u0=u0,
        orientation=orientation,
        tilts=tilts,
        tilt_reduction=reduction,
    )


def _numpy(tensor: Tensor) -> np.ndarray:
    """Detach to a host NumPy array, preserving dtype (checkpoint tensors carry no grad)."""
    return tensor.detach().cpu().numpy()


def _dump_reduction(reduction: TiltReduction) -> dict[str, Any]:
    if isinstance(reduction, MosaicSmoothed):
        return {"kind": "mosaic", "samples": reduction.samples}
    return {"kind": "plain"}


def _load_reduction(payload: dict[str, Any]) -> TiltReduction:
    if payload["kind"] == "mosaic":
        samples = payload["samples"]
        assert isinstance(samples, int)
        return MosaicSmoothed(samples=samples)
    return PlainSum()

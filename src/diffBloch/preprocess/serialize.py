"""Serialize a preprocessed ``Plan`` to a portable ``.npz`` checkpoint and read it back.

The persistence primitive behind the ``run`` program's preprocess cache: *serialize the whole
``Plan``*, never per-facet state (see ``design/decisions/effects-and-observability.md``). It
exploits the ``Plan``'s own design -- :class:`~diffBloch.engine.plan.OrientationPlan` separates its
**source / rebuild inputs** (``orientation`` / ``tilts`` / ``thickness`` / ``beam_hkl`` / the
observed ``pattern`` / ``energy`` / ``u0`` / ``tilt_reduction``) from its **compiled geometry**
(``beam_plans`` incl. the heavy ``StructureFactorGather``, ``alignment``), and both
:meth:`~diffBloch.engine.plan.ScatteringGrid.from_cell` and
:meth:`~diffBloch.engine.plan.OrientationPlan.build` reconstruct the compiled parts from the source.
So we persist only the source (a modest bundle of small arrays) and rebuild the derived geometry on
read -- the compiled gather is a pure function of ``beam_hkl`` + grid, so a stored copy could only
desync.

Format: one ``.npz`` (numpy is already a core dependency, so no new one; portable, non-pickle hence
auditable, inspectable, and it holds the ragged per-rotation arrays natively). Structure and scalars
-- ``g_max``, per-rotation ``energy`` / ``u0`` and the ``tilt_reduction`` discriminant -- ride in a
JSON string stored as the reserved ``__meta__`` array, so the file is self-describing. Read uses
``allow_pickle=False``: the checkpoint is data, never code.

This lives in ``preprocess`` (where ``Plan`` lives), not ``io`` -- ``io`` is the input-record layer
*below* ``core`` / ``engine`` / ``preprocess``, so a ``Plan`` serializer there would invert the
dependency. The ``plan.lock`` provenance that binds a checkpoint to the inputs + config that
produced it is a separate, data-free concern in ``config.manifest`` (beside ``experiment.lock``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from diffBloch.core.products import MosaicSmoothed, PatternBatch, PlainSum, TiltReduction
from diffBloch.engine.plan import OrientationPlan, ScatteringGrid
from diffBloch.preprocess.plan import Plan

__all__ = ["read_plan", "write_plan"]

_FORMAT_VERSION = 1


def write_plan(plan: Plan, path: str | Path) -> None:
    """Write ``plan`` to ``path`` as a portable ``.npz`` checkpoint (source arrays + JSON meta)."""
    arrays: dict[str, np.ndarray] = {"cell": _numpy(plan.grid.cell)}
    per_rotation: list[dict[str, object]] = []
    for index, op in enumerate(plan.orientations):
        arrays[f"orient_{index}"] = _numpy(op.orientation)
        arrays[f"tilts_{index}"] = _numpy(op.tilts)
        arrays[f"thickness_{index}"] = _numpy(op.thickness)
        arrays[f"beam_hkl_{index}"] = _numpy(op.beam_hkl)
        arrays[f"pat_hkl_{index}"] = _numpy(op.pattern.hkl)
        arrays[f"pat_int_{index}"] = _numpy(op.pattern.intensities)
        arrays[f"pat_sig_{index}"] = _numpy(op.pattern.sigmas)
        per_rotation.append(
            {
                "energy": op.energy,
                "u0": op.u0,
                "tilt_reduction": _dump_reduction(op.tilt_reduction),
            }
        )
    meta = {
        "format_version": _FORMAT_VERSION,
        "g_max": plan.grid.g_max,
        "n_orientations": len(plan.orientations),
        "orientations": per_rotation,
    }
    arrays["__meta__"] = np.array(json.dumps(meta))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as file:
        np.savez_compressed(file, **arrays)  # type: ignore[arg-type]


def read_plan(path: str | Path) -> Plan:
    """Read a ``.npz`` checkpoint from :func:`write_plan`, rebuilding the compiled geometry."""
    with np.load(Path(path), allow_pickle=False) as data:
        meta = json.loads(str(data["__meta__"].item()))
        if meta["format_version"] != _FORMAT_VERSION:
            raise ValueError(
                f"unsupported plan checkpoint format {meta['format_version']!r}: "
                f"expected {_FORMAT_VERSION}"
            )
        grid = ScatteringGrid.from_cell(np.asarray(data["cell"]), g_max=float(meta["g_max"]))
        orientations = tuple(
            OrientationPlan.build(
                grid,
                np.asarray(data[f"beam_hkl_{index}"]),
                PatternBatch(
                    hkl=torch.as_tensor(data[f"pat_hkl_{index}"]),
                    intensities=torch.as_tensor(data[f"pat_int_{index}"]),
                    sigmas=torch.as_tensor(data[f"pat_sig_{index}"]),
                ),
                energy=float(entry["energy"]),
                thickness=torch.as_tensor(data[f"thickness_{index}"]),
                u0=float(entry["u0"]),
                orientation=np.asarray(data[f"orient_{index}"]),
                tilts=np.asarray(data[f"tilts_{index}"]),
                tilt_reduction=_load_reduction(entry["tilt_reduction"]),
            )
            for index, entry in enumerate(meta["orientations"])
        )
    return Plan(grid=grid, orientations=orientations)


def _numpy(tensor: Tensor) -> np.ndarray:
    """Detach to a host NumPy array, preserving dtype (checkpoint tensors carry no grad)."""
    return tensor.detach().cpu().numpy()


def _dump_reduction(reduction: TiltReduction) -> dict[str, object]:
    if isinstance(reduction, MosaicSmoothed):
        return {"kind": "mosaic", "window": reduction.window}
    return {"kind": "plain"}


def _load_reduction(payload: dict[str, object]) -> TiltReduction:
    if payload["kind"] == "mosaic":
        window = payload["window"]
        assert isinstance(window, int)
        return MosaicSmoothed(window=window)
    return PlainSum()

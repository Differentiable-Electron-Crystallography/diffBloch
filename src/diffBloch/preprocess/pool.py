"""``pool``: merge settled per-dataset ``Plan``\\ s into the one ``Plan`` refinement consumes.

The multi-dataset (``inputs.multi_dataset``) merge seam. Each dataset's plan is seeded by
:func:`~diffBloch.preprocess.experiment.setup_datasets`, sharpened by its own recipe run, and
checkpointed on its own (``plan.<stem>.npz``); ``pool`` then concatenates the settled plans --
in memory, in ``inputs.exp_data`` order, just before refinement. The pooled plan is never
checkpointed (it is cheap to recompute and would only drift from its constituents), and it is
deliberately **not** a ``Plan -> Plan`` recipe step: it merges plans rather than transforming one,
so it never enters a recipe list, a checkpoint lock, or the opaque-step rules.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import torch

from diffBloch.preprocess.plan import Plan

__all__ = ["pool"]


def pool(plans: Sequence[Plan], *, offsets: Sequence[int]) -> Plan:
    """Concatenate per-dataset plans onto the pooled global ``rotation_index`` space.

    ``offsets[k]`` is dataset ``k``'s start in the pooled index space -- the cumulative sum of the
    *full pre-ignore* rotation counts of the datasets before it -- and every orientation of
    ``plans[k]`` is renumbered ``global = offsets[k] + file_local``. File-local indices already
    carry gaps where rotations were ignored, so the pooled indices are exactly the non-ignored
    subset of ``0..total-1`` and everything keyed on them (``ignore_orientations``, the train/val
    mask, the thickness lookup's concatenated alphas) addresses the same rotation before and after
    pooling. Renumbering is a ``dataclasses.replace`` of each orientation's ``pattern`` --
    ``rotation_index`` is a plain label nothing derives from. (For a ``CoupledOrientationPlan``
    the segments' sub-plans keep the stale file-local pattern object; nothing consumes a segment's
    pattern -- the solve reads ``segment.plan.beam_plans`` and serialization writes only the parent
    pattern -- so it is left alone rather than rebuilt.)

    Two guards protect the single-experiment invariants downstream:

    - **one grid**: every plan must ride the same structure-factor support. Same *object* passes
      trivially (the fresh-construction path); plans read back from per-dataset checkpoints rebuild
      equal-but-distinct grids, so equality of the defining pair (``cell``, ``g_max``) is accepted
      -- everything else on the grid derives deterministically from those.
    - **one energy**: the engine computes structure factors at a single beam energy for the whole
      model (``RefinementEngine`` reads ``orientations[0].energy``), so pooling datasets whose
      snapped energies differ would silently score every other dataset at the first one's voltage.
      Rejected here, loudly, instead.

    ``provenance`` is carried from ``plans[0]``: every dataset ran the same step list (differing
    only in per-dataset geometry params), and the pooled plan is never checkpointed, so provenance
    is display-only downstream.
    """
    if not plans:
        raise ValueError("pool needs at least one plan")
    if len(offsets) != len(plans):
        raise ValueError(f"pool got {len(plans)} plans but {len(offsets)} offsets")

    grid = plans[0].structure_factor_grid
    for plan in plans[1:]:
        other = plan.structure_factor_grid
        if other is grid:
            continue
        if other.g_max != grid.g_max or not torch.equal(other.cell, grid.cell):
            raise ValueError(
                "pooled plans disagree on the structure-factor grid (cell or g_max); every "
                "dataset must be preprocessed against the same structure and blochwave.g_max"
            )

    energies = sorted({op.energy for plan in plans for op in plan.orientations})
    if len(energies) > 1:
        raise ValueError(
            f"pooled datasets have different beam energies {energies}; the engine solves the "
            "whole experiment at one energy, so datasets recorded at different accelerating "
            "voltages cannot be pooled"
        )

    pooled = tuple(
        replace(op, pattern=replace(op.pattern, rotation_index=offset + op.pattern.rotation_index))
        for plan, offset in zip(plans, offsets, strict=True)
        for op in plan.orientations
    )
    indices = [op.pattern.rotation_index for op in pooled]
    if len(set(indices)) != len(indices):
        raise ValueError("pooled rotation indices collide; offsets must partition the index space")
    return Plan(structure_factor_grid=grid, orientations=pooled, provenance=plans[0].provenance)

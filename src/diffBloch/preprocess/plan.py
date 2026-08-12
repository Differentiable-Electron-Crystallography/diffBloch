"""The ``Plan``: the invariant geometry the differentiable refinement is conditioned on.

``Plan`` is the *spine* of the preprocess pipeline -- one immutable value bundling the shared
:class:`~diffBloch.engine.plan.StructureFactorGrid` and the per-rotation
:class:`~diffBloch.engine.plan.OrientationPlan`\\ s. Each ``Plan -> Plan`` step returns a sharpened
copy (via :func:`dataclasses.replace`); ``refine`` consumes the final ``Plan``. The dependency
points ``preprocess -> engine`` (this module imports the engine's geometry plans), never the
reverse -- the engine stays unaware of ``Plan`` and remains a pure consumer of grid + orientations.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from diffBloch.core.products import PatternBatch
from diffBloch.engine.plan import (
    CoupledOrientationPlan,
    OrientationPlan,
    OrientationPlanLike,
    StructureFactorGrid,
)

if TYPE_CHECKING:
    from diffBloch.preprocess.pipeline import StepRecord

__all__ = [
    "CandidatePlan",
    "Plan",
    "coupling_stats",
    "require_built_plans",
    "require_candidate_plans",
    "require_orientation_plans",
    "summarize_plan",
    "unique_hkl_count",
]


@dataclass(frozen=True)
class CandidatePlan:
    """The pre-build *candidate* phase for one rotation: source only, no built geometry.

    ``from_experiment`` lays down a :class:`CandidatePlan` per rotation (the difference-safe
    candidate ``beam_hkl`` + orientation source), ``select_beams`` prunes ``beam_hkl`` (cheap,
    source-only), and :func:`~diffBloch.preprocess.steps.beams.build_orientation_plans` then
    builds it into an :class:`~diffBloch.engine.plan.OrientationPlan` -- the *one* place the
    structure-factor gather is built, over the already-pruned beam set. A ``CandidatePlan`` has no
    ``beam_plans``, so it is unsolvable by construction (the engine consumes only built
    ``OrientationPlan``\\ s); building the expensive gather over the full candidate pool is thereby
    avoided entirely.
    """

    orientation: NDArray[np.float64]
    energy: float
    u0: float
    thickness: NDArray[np.float64]
    beam_hkl: NDArray[np.int64]
    pattern: PatternBatch

    @classmethod
    def seed(
        cls,
        beam_hkl: NDArray[np.int64],
        pattern: PatternBatch,
        *,
        energy: float,
        thickness: Tensor | NDArray[np.float64] | Sequence[float],
        u0: float = 0.0,
        orientation: Tensor | NDArray[np.float64] | None = None,
    ) -> CandidatePlan:
        """Assemble a candidate seed for one orientation (plain-numpy source; builds no gather)."""
        rotation = np.eye(3) if orientation is None else np.asarray(orientation, dtype=np.float64)
        return cls(
            orientation=rotation,
            energy=float(energy),
            u0=float(u0),
            thickness=np.atleast_1d(np.asarray(thickness, dtype=np.float64)),
            beam_hkl=np.asarray(beam_hkl, dtype=np.int64),
            pattern=pattern,
        )


@dataclass(frozen=True)
class Plan:
    """Shared ``structure_factor_grid`` plus per-rotation ``orientations`` (the refinement spine).

    ``structure_factor_grid`` fixes the ``Fgb`` support and metric; ``orientations`` is one
    :class:`~diffBloch.engine.plan.OrientationPlan` per rotation (each already coupled to the grid
    at build time). Immutable: preprocess steps return :func:`dataclasses.replace` copies rather
    than mutating in place.

    ``provenance`` is the ordered tuple of :class:`~diffBloch.preprocess.pipeline.StepRecord`\\ s
    that produced this plan -- :func:`~diffBloch.preprocess.pipeline.pipeline` appends one per step
    as it runs. A freshly built plan (from ``from_experiment``) has empty provenance; the recipe
    identity a checkpoint locks against is this tuple. Steps do not touch it (their ``replace``
    preserves it); the combinator owns it.
    """

    structure_factor_grid: StructureFactorGrid
    orientations: tuple[CandidatePlan | OrientationPlanLike, ...]
    provenance: tuple[StepRecord, ...] = field(default_factory=tuple)


def require_orientation_plans(plan: Plan) -> tuple[OrientationPlan, ...]:
    """Narrow a plan's orientations to plain :class:`OrientationPlan`\\ s (reject segmented ones).

    The tilt-independent-only plan-shaping steps (``select_beams``, ``integrate_rocking_curve``)
    transform the :class:`OrientationPlan`, which carries one shared beam set. ``couple_beams``
    replaces each orientation with a
    :class:`~diffBloch.engine.plan.CoupledOrientationPlan` (a per-tilt-chunk beam set) that those
    steps cannot consume, so they must all precede ``couple_beams`` in a pipeline. This helper
    enforces that ordering with a clear error and narrows the element type for the caller.

    The fitting steps (``optimize_orientation``, ``optimize_thickness``) are deliberately *not* narrowed: they
    are plan-agnostic (they rebuild via :meth:`OrientationPlan.with_orientation` /
    ``replace(thickness=...)``, both defined on the segmented plan too), so they iterate
    ``plan.orientations`` directly and run either before or after ``couple_beams``.
    """
    narrowed: list[OrientationPlan] = []
    for op in plan.orientations:
        if not isinstance(op, OrientationPlan):
            raise TypeError(
                "this step transforms tilt-independent OrientationPlans, but this plan holds a "
                "CoupledOrientationPlan; couple_beams produces those and no other Plan -> Plan "
                "step can consume them, so couple_beams must be the final step in the pipeline"
            )
        narrowed.append(op)
    return tuple(narrowed)


def require_built_plans(plan: Plan) -> tuple[OrientationPlanLike, ...]:
    """Parse the orientation *phase* to built (``OrientationPlan`` / ``CoupledOrientationPlan``).

    ``Plan.orientations`` is a phase-union (:class:`CandidatePlan` before the build, built plans
    after) rather than a phase-indexed ``Plan[P]``, as the recipe is a runtime ``pipeline`` list
    of homogeneous ``Plan -> Plan`` steps: the phase cannot ride in the type across that list (nor
    across ``read_plan``, which reconstructs a ``Plan`` from ``.npz`` bytes). This is the parse that
    re-establishes the phase at that erased boundary -- *parse, don't validate*: it returns the
    narrowed type once, and the terminals (inference, the fits, ``couple_beams``, checkpoint
    serialize) then hold built geometry without re-checking. The phase-indexed alternative
    (``Plan[P]``) does not survive contact with Python -- there is no phase-changing composition over
    a homogeneous step list, and ``disallow_any_generics`` rejects the erased reconstruction.

    A :class:`CandidatePlan` has no ``beam_plans`` and is unsolvable, so this raises unless
    ``build_orientation_plans`` has run (the default recipe runs it right after ``select_beams``).
    Returns the plan's own ``orientations`` tuple (identity preserved) once narrowed.
    """
    for op in plan.orientations:
        if isinstance(op, CandidatePlan):
            raise TypeError(
                "this operation needs a built plan (OrientationPlan / CoupledOrientationPlan), "
                "but the plan holds a CandidatePlan; run build_orientation_plans after select_beams"
            )
    return cast("tuple[OrientationPlanLike, ...]", plan.orientations)


def require_candidate_plans(plan: Plan) -> tuple[CandidatePlan, ...]:
    """Parse the orientation *phase* to the pre-build candidate (:class:`CandidatePlan`).

    The candidate-phase counterpart of :func:`require_built_plans` (which documents why the phase is
    parsed at a runtime boundary rather than carried in the type). ``select_beams`` and
    ``build_orientation_plans`` operate on the candidate phase ``from_experiment`` lays down; this
    raises with a clear error if the plan is already built (holds
    :class:`~diffBloch.engine.plan.OrientationPlan`\\ s), since ``build_orientation_plans`` is the
    single step that builds them and runs once, right after ``select_beams``.
    """
    narrowed: list[CandidatePlan] = []
    for op in plan.orientations:
        if not isinstance(op, CandidatePlan):
            raise TypeError(
                "this step operates on the pre-build candidate phase, but this plan is already "
                "built (holds OrientationPlans); build_orientation_plans builds them once, right "
                "after select_beams"
            )
        narrowed.append(op)
    return tuple(narrowed)


def coupling_stats(op: CandidatePlan | OrientationPlanLike) -> dict[str, int]:
    """One rotation's solve-geometry shape: the ``(unions, tilts-per-union, beams-per-union)`` cost.

    Phase-robust (a plan is summarised after every pipeline step, from the pre-build candidate on):
    a :class:`~diffBloch.engine.plan.CoupledOrientationPlan` reports its real coupling
    (``n_coupling_segments`` unions, per-union ``cover`` widths and ``union_beam_index`` beam
    counts); a built :class:`~diffBloch.engine.plan.OrientationPlan` is one implicit union spanning
    all its tilts; a pre-build :class:`CandidatePlan` knows only its beam-pool size (no
    tilts/segments yet). These are exactly the ``(B, T, N)`` drivers of the segmented Bloch solve
    the refinement loop repeats.
    """
    if isinstance(op, CoupledOrientationPlan):
        covers = [len(segment.cover) for segment in op.segments]
        seg_beams = [int(segment.union_beam_index.shape[0]) for segment in op.segments]
        return {
            "n_coupling_segments": len(op.segments),
            "n_tilts": int(op.tilts.shape[0]),
            "max_tilts_per_segment": max(covers, default=0),
            "n_union_beams": int(op.beam_hkl.shape[0]),
            "max_beams_per_segment": max(seg_beams, default=0),
        }
    if isinstance(op, OrientationPlan):
        n_tilts = len(op.beam_plans)
        beams = int(op.beam_hkl.shape[0])
        return {
            "n_coupling_segments": 1,
            "n_tilts": n_tilts,
            "max_tilts_per_segment": n_tilts,
            "n_union_beams": beams,
            "max_beams_per_segment": beams,
        }
    beams = int(np.asarray(op.beam_hkl).shape[0])  # CandidatePlan: only the beam pool is known
    return {
        "n_coupling_segments": 0,
        "n_tilts": 0,
        "max_tilts_per_segment": 0,
        "n_union_beams": beams,
        "max_beams_per_segment": beams,
    }


def unique_hkl_count(hkl_batches: Iterable[Tensor]) -> int:
    """Count of *distinct* ``(h, k, l)`` triples across ``hkl_batches`` (each ``(M, 3)``).

    A reflection recorded in multiple rotations (rotation electron diffraction frames overlap in
    angle, so the same reciprocal-lattice point is routinely re-observed in several consecutive
    frames) is counted once here, not once per rotation it appears in -- a plain
    ``sum(len(batch) ...)`` double-counts exactly those overlaps.
    """
    non_empty = [batch for batch in hkl_batches if batch.shape[0] > 0]
    if not non_empty:
        return 0
    return int(torch.unique(torch.cat(non_empty, dim=0), dim=0).shape[0])


def summarize_plan(plan: Plan) -> dict[str, float]:
    """Plan-level shape as numeric measurements (the observability summary of a settled/mid Plan).

    Emitted after every pipeline step (and once for the seed, as :class:`PlanSeeded`), so consecutive
    summaries *are* the per-stage survival counts: how many solve beams and scored reflections each
    filtering step left behind. The names are scoped because the sets are independent -- SOLVE
    (``n_solve_beams_*``, the beams that couple dynamically) is not SCORED (``n_matched_hkl``, the
    reflections that enter the R-factor) is not the structure-factor support (``n_grid_hkl``).
    ``n_observed_hkl``/``n_matched_hkl`` are *deduplicated* distinct ``(h, k, l)`` counts across every
    rotation (:func:`unique_hkl_count`), not a sum of each rotation's own count -- a reflection
    re-observed (or matched) in more than one rotation is counted once, not once per rotation.

    ``n_matched_hkl`` is **absent**, not zero, before ``build_orientation_plans`` runs: a
    :class:`CandidatePlan` has no alignment, and reporting ``0`` there would make "not built yet"
    indistinguishable from "matched nothing".
    """
    beams = [coupling_stats(op)["n_union_beams"] for op in plan.orientations]
    summary = {
        "n_orientations": float(len(plan.orientations)),
        "n_grid_hkl": float(plan.structure_factor_grid.structure_factor_hkl.shape[0]),
        "n_solve_beams_total": float(sum(beams)),
        "n_solve_beams_max": float(max(beams, default=0)),
        "n_observed_hkl": float(unique_hkl_count(op.pattern.hkl for op in plan.orientations)),
    }
    built = [op for op in plan.orientations if not isinstance(op, CandidatePlan)]
    if len(built) == len(plan.orientations):
        summary["n_matched_hkl"] = float(unique_hkl_count(op.alignment.hkl for op in built))
    return summary

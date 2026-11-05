"""``score_integrated_reflections``: score each measured intensity once, where it is integrated.

A ``Plan -> Plan`` step run once the orientations and thicknesses are settled. Each measured
``(hkl, I_exp)`` is scored in exactly one virtual frame: among the frames whose sweep contains its
Bragg crossing, the one whose centre is closest to it. A crossing near a frame's edge therefore goes
to the neighbouring frame it is more central in, and at the ends of the data the edge frame keeps
it. The SOLVE beam sets are untouched.

The rocking-curve width is measured from the simulation rather than set by hand: a few rotations
are solved over a sweep wider than the frame, and the median ``Sg`` span holding
``spec.fraction`` of each reflection's intensity is the width (:func:`rocking_curve_width`).
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from diffBloch.core.crystal import orientation_basis
from diffBloch.core.dynamical import build_bloch_systems, stack_beam_plans
from diffBloch.core.dynamical.primitives import excitation_errors
from diffBloch.core.products import AlignmentPlan
from diffBloch.core.reciprocal import g_vectors
from diffBloch.core.solver import SolverMethod, memory_safe_max_batch, propagate
from diffBloch.engine import CoupledOrientationPlan, RefinementEngine, StructureFactorGrid
from diffBloch.engine.plan import OrientationPlan, OrientationPlanLike
from diffBloch.observability import IntegratedRockingCurves
from diffBloch.params import Device
from diffBloch.preprocess.coupling import build_coupling_segments
from diffBloch.preprocess.experiment import RefinementSetup
from diffBloch.preprocess.integrated import (
    bragg_crossing_angles,
    integrated_mask,
    sg_per_radian,
)
from diffBloch.preprocess.orientation import rocking_curve_tilts
from diffBloch.preprocess.pipeline import PlanStep, as_step
from diffBloch.preprocess.plan import Plan, require_built_plans
from diffBloch.preprocess.scoring import build_engine
from diffBloch.specs import (
    NO_ABSORPTION,
    Absorption,
    BeamSelection,
    IntegratedScoring,
    RockingCurve,
    UnionCoupling,
)

__all__ = ["rocking_curve_report", "rocking_curve_width", "score_integrated_reflections"]

_log = logging.getLogger(__name__)


def score_integrated_reflections(
    refinement: RefinementSetup,
    spec: IntegratedScoring,
    *,
    rocking: RockingCurve,
    solve_g_max: float,
    sg_max: float,
    method: SolverMethod = "matrix_exp",
    absorption: Absorption = NO_ABSORPTION,
    device: Device | None = None,
    max_batch: int | None = None,
) -> PlanStep:
    """Return the step narrowing every rotation's scored set to integrated, once-only reflections.

    ``rocking`` is the frame geometry (semiangle, tilt count). ``solve_g_max`` / ``sg_max`` select
    the beams for the wide width-measuring solve, as the coupling does for the frame's own solve.
    ``method``, ``absorption``, ``device`` and ``max_batch`` configure that solve; ``device`` and
    ``max_batch`` are execution-only. The report's rocking curves come from
    :func:`rocking_curve_report`, run on the settled plan whatever the scoring method.
    """

    def run(plan: Plan) -> Plan:
        ops = require_built_plans(plan)
        params = refinement.params if device is None else refinement.params.to(device)
        engine = build_engine(
            plan, refinement, method=method, max_batch=max_batch, absorption=absorption
        )
        with torch.no_grad():
            fgb = engine.fgb(params)
        cell = np.asarray(plan.structure_factor_grid.cell)

        # For each measurement (hkl, I_exp), the rotation and row where it is most central.
        best: dict[tuple[int, int, int, float], tuple[float, int, int]] = {}
        for index, op in enumerate(ops):
            hkl = op.alignment.hkl.cpu().numpy().astype(np.int64)
            i_exp = op.pattern.intensities[op.alignment.pattern_index].cpu().numpy()
            g = g_vectors(hkl, orientation_basis(cell, np.asarray(op.orientation)))
            # Every frame whose sweep contains the crossing is a candidate; the one whose centre is
            # closest wins below. A crossing near a frame's edge therefore goes to the neighbour
            # it is more central in.
            keep, margin = integrated_mask(
                g, energy=op.energy, u0=op.u0, integration=rocking.integration
            )
            keep &= _inside_outer_limit(ops, index, g, op, rocking)
            for row in np.flatnonzero(keep).tolist():
                h, k, ell = (int(v) for v in hkl[row])
                key = (h, k, ell, float(i_exp[row]))
                if key not in best or margin[row] > best[key][0]:
                    best[key] = (float(margin[row]), index, row)

        rows_by_op: dict[int, list[int]] = {}
        for _, chosen_op, chosen_row in best.values():
            rows_by_op.setdefault(chosen_op, []).append(chosen_row)
        scored = tuple(
            _keep_rows(op, sorted(rows_by_op.get(index, []))) for index, op in enumerate(ops)
        )
        # Keep a reflection only when its frame holds at least spec.min_inside of its simulated
        # rocking curve: a crossing near an edge, or a curve too broad for any one frame, fails.
        edge_common = {
            "rocking": rocking,
            "solve_g_max": solve_g_max,
            "sg_max": sg_max,
            "method": method,
            "absorption": absorption,
            "max_batch": max_batch,
            "fraction": spec.min_inside,
            "engine": engine,
        }
        n_before_drop = sum(int(op.alignment.hkl.shape[0]) for op in scored)
        checked = [
            _drop_spilling(plan, op, fgb, **edge_common)  # type: ignore[arg-type]
            for op in scored
        ]
        scored = tuple(op for op, _ in checked)
        n_dropped = sum(dropped for _, dropped in checked)
        if n_dropped:
            share = n_dropped / max(n_before_drop, 1)
            message = (
                "integrated scoring: dropped %d of %d scored reflections (%.0f%%) with under "
                "%.0f%% of their rocking curve inside their frame, in %d of %d frames"
            )
            args = (
                n_dropped,
                n_before_drop,
                100.0 * share,
                100.0 * spec.min_inside,
                sum(1 for _, dropped in checked if dropped),
                len(checked),
            )
            if share > _DROP_WARNING_FRACTION:
                _log.warning(message + "; consider widening virtual frames", *args)
            else:
                _log.info(message, *args)
        # The assignment above needs no simulation. One solve over the first two frames gives
        # both the report's curves and the rocking-curve width the broad-reflection warning uses.
        common = {
            "rocking": rocking,
            "solve_g_max": solve_g_max,
            "sg_max": sg_max,
            "method": method,
            "absorption": absorption,
            "max_batch": max_batch,
        }
        if len(ops) >= 2:
            width = _rocking_curves_event(
                plan,
                scored,
                fgb,
                spec,
                start=0,
                **common,  # type: ignore[arg-type]
            ).width_sg
        else:
            width = rocking_curve_width(plan, fgb, spec, **common)  # type: ignore[arg-type]
        _warn_if_broad(scored, cell, width, rocking)
        _log.info("integrated scoring: rocking-curve width %.3g A^-1 in Sg", width)
        return replace(plan, orientations=scored)

    return as_step("score_integrated_reflections", spec, run)


def rocking_curve_report(
    plan: Plan,
    refinement: RefinementSetup,
    *,
    rocking: RockingCurve,
    solve_g_max: float,
    sg_max: float,
    method: SolverMethod = "matrix_exp",
    absorption: Absorption = NO_ABSORPTION,
    device: Device | None = None,
    max_batch: int | None = None,
    which: Literal["first2", "all"] = "first2",
    max_pairs: int | None = None,
    klar: BeamSelection | None = None,
) -> tuple[IntegratedRockingCurves, ...]:
    """The report's rocking-curve events for a settled ``plan``, whatever its scoring method.

    One :class:`~diffBloch.observability.IntegratedRockingCurves` for the first two rotations, or
    with ``which="all"`` one per overlapping adjacent pair (0-1, 1-2, 2-3, ...). Empty for a plan
    of fewer than two rotations. A
    reflection one scoring method scores in several frames (the Klar window can) lists the others
    in ``also_scored_in``. ``klar`` is the Klar ``rsg`` / ``dsg`` window when that scored the plan;
    each event then carries every reflection's acceptance half-width (``klar_window_deg``) so the
    report can shade it. Report-only: nothing here changes the plan.
    """
    ops = require_built_plans(plan)
    if len(ops) < 2:
        return ()
    params = refinement.params if device is None else refinement.params.to(device)
    engine = build_engine(
        plan, refinement, method=method, max_batch=max_batch, absorption=absorption
    )
    with torch.no_grad():
        fgb = engine.fgb(params)
    if max_pairs is not None and max_pairs < 1:
        raise ValueError(f"max_pairs must be at least 1, got {max_pairs}")
    starts = [0] if which == "first2" else _pair_starts(len(ops))[:max_pairs]

    def event_at(start: int) -> IntegratedRockingCurves:
        return _rocking_curves_event(
            plan,
            ops,
            fgb,
            IntegratedScoring(),
            start=start,
            report_mode=which,
            klar=klar,
            rocking=rocking,
            solve_g_max=solve_g_max,
            sg_max=sg_max,
            method=method,
            absorption=absorption,
            max_batch=max_batch,
        )

    events = {start: event_at(start) for start in starts}
    # In "all" mode, a reflection scored in a frame no shown pair contains is listed as scored
    # elsewhere with nothing to show it in. Add the pair starting at that frame (its nearest neighbour is the
    # next frame, or the previous one for the last frame) so the owning frame is on the page.
    if which == "all":
        shown = {rotation for event in events.values() for rotation in event.rotation_indices}
        owners = {owner for event in events.values() for owner in event.scored_in if owner != -1}
        position = {int(op.pattern.rotation_index): k for k, op in enumerate(ops)}
        for owner in sorted(owners - shown):
            if owner not in position:
                continue
            k = position[owner]
            start = k if k + 1 < len(ops) else k - 1
            if start not in events:
                events[start] = event_at(start)
    return tuple(events[start] for start in sorted(events))


def _pair_starts(n_rotations: int) -> list[int]:
    """Start of every overlapping adjacent frame pair through the final rotation."""
    return list(range(n_rotations - 1))


def _rx(angles_deg: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rotations about the goniometer ``x`` axis, ``(N, 3, 3)``."""
    t = np.deg2rad(angles_deg)
    c, s = np.cos(t), np.sin(t)
    out = np.zeros((len(t), 3, 3))
    out[:, 0, 0] = 1.0
    out[:, 1, 1], out[:, 1, 2], out[:, 2, 1], out[:, 2, 2] = c, -s, s, c
    return out


def _rocking_curves_event(
    plan: Plan,
    scored: tuple[OrientationPlanLike, ...],
    fgb: Tensor,
    spec: IntegratedScoring,
    *,
    start: int = 0,
    report_mode: Literal["first2", "all"] = "first2",
    klar: BeamSelection | None = None,
    rocking: RockingCurve,
    solve_g_max: float,
    sg_max: float,
    method: SolverMethod,
    absorption: Absorption,
    max_batch: int | None,
) -> IntegratedRockingCurves:
    """Curves for rotations ``start .. start + 1`` on the PETS absolute alpha axis.

    The same solve measures the rocking-curve width (:func:`_median_width`), so the step needs
    no separate width simulation.
    """
    ops = require_built_plans(plan)
    frames = ops[start : start + 2]
    middle = frames[-1]
    centre = np.asarray(middle.orientation)
    # The plot coordinates are the measured PETS alpha values, while the wide solve is expressed
    # relative to the latter frame's orientation.
    centres = [float(op.pattern.alpha_degrees) for op in frames]
    middle_alpha = float(middle.pattern.alpha_degrees)
    semi = rocking.integration.semiangle
    lo, hi = min(centres) - semi, max(centres) + semi  # exactly the two frames
    step = 2.0 * semi / max(rocking.sampling - 1, 1)
    angles = np.linspace(lo, hi, int(round((hi - lo) / step)) + 1)
    tilts = _rx(angles - middle_alpha)

    grid = plan.structure_factor_grid
    basis = orientation_basis(np.asarray(grid.cell), centre)
    beam_hkl = _wide_sweep_beams(grid, basis, tilts, middle, solve_g_max, sg_max)
    wide = OrientationPlan.build(
        grid,
        beam_hkl,
        middle.pattern,
        energy=middle.energy,
        thickness=middle.thickness,
        u0=middle.u0,
        orientation=centre,
        tilts=tilts,
        build_alignment=False,
    )
    curves = _tilt_intensities(wide, fgb, middle.thickness, method, absorption, max_batch)
    column = {tuple(h): i for i, h in enumerate(beam_hkl.tolist())}

    def hkl_set(op: OrientationPlanLike) -> set[tuple[int, ...]]:
        return {tuple(h) for h in op.alignment.hkl.tolist()}

    # Every observed reflection whose Bragg condition one of the two frames crosses, up to the
    # largest |g| the plan scores anywhere, plus everything the two frames score.
    cell = np.asarray(grid.cell)
    reciprocal = np.asarray(grid.reciprocal_basis)
    g_cap = max(
        (float(np.linalg.norm(np.asarray(h) @ reciprocal)) for op in scored for h in hkl_set(op)),
        default=0.0,
    )
    crossed: set[tuple[int, ...]] = set()
    for op in frames:
        observed = op.pattern.hkl.cpu().numpy().astype(np.int64)
        in_cap = np.linalg.norm(observed @ reciprocal, axis=1) <= g_cap
        g_obs = g_vectors(observed, orientation_basis(cell, np.asarray(op.orientation)))
        crosses, _ = integrated_mask(
            g_obs, energy=op.energy, u0=op.u0, integration=rocking.integration
        )
        crossed |= {tuple(h) for h in observed[crosses & in_cap].tolist()}
    crossed |= set().union(*(hkl_set(op) for op in scored[start : start + 2]))
    crossed -= {(0, 0, 0)}
    # Report-only: a reflection is drawn only if a measurement of it in these two frames is above
    # 3 sigma. The weak ones stay out of the curves; scoring is unchanged.
    strong = {
        tuple(int(v) for v in h)
        for op in frames
        for h, i, sig in zip(
            op.pattern.hkl.cpu().tolist(),
            op.pattern.intensities.cpu().tolist(),
            op.pattern.sigmas.cpu().tolist(),
            strict=True,
        )
        if i > 3.0 * sig
    }
    hkl = sorted(h for h in crossed if h in column and h in strong)
    owners_of: dict[tuple[int, ...], list[int]] = {}
    for op in scored:
        for h in hkl_set(op):
            owners_of.setdefault(h, []).append(op.pattern.rotation_index)
    # Sg of every reflection along the shared sweep.
    g_hkl = g_vectors(np.asarray(hkl, dtype=np.int64), basis)
    tilted = np.einsum("tij,nj->tni", tilts, g_hkl)
    sg = excitation_errors(tilted.reshape(-1, 3), middle.energy, u0=middle.u0).reshape(
        len(angles), -1
    )
    per_hkl = np.stack([curves[:, column[h]] for h in hkl], axis=1)
    width = _median_width(per_hkl, sg, spec.fraction)
    lever = np.linalg.norm(g_hkl[:, 1:], axis=1)
    with np.errstate(divide="ignore"):
        half_width = np.rad2deg((width / 2.0) / lever)
    # Every Sg = 0 crossing along the shared sweep, by linear interpolation between tilts.
    zeros: list[tuple[float, ...]] = []
    for n in range(len(hkl)):
        a, b = sg[:-1, n], sg[1:, n]
        steps = np.flatnonzero(np.sign(a) != np.sign(b))
        zeros.append(
            tuple(
                float(angles[k] + a[k] / (a[k] - b[k]) * (angles[k + 1] - angles[k])) for k in steps
            )
        )

    shown = {op.pattern.rotation_index for op in frames}
    # Colour by a shown frame that scores it when there is one; list any other scoring frames.
    scored_in, also_scored_in = [], []
    for h in hkl:
        owners = sorted(owners_of.get(h, []), key=lambda r: (r not in shown, r))
        scored_in.append(owners[0] if owners else -1)
        also_scored_in.append(tuple(owners[1:]))
    return IntegratedRockingCurves(
        dataset=middle.pattern.dataset,
        rotation_indices=tuple(op.pattern.rotation_index for op in frames),
        frame_centres_deg=tuple(centres),
        semiangle=semi,
        width_sg=width,
        angles_deg=tuple(float(a) for a in angles),
        hkl=tuple((int(h[0]), int(h[1]), int(h[2])) for h in hkl),
        scored_in=tuple(scored_in),
        scored_elsewhere=tuple(o != -1 and o not in shown for o in scored_in),
        also_scored_in=tuple(also_scored_in),
        half_width_deg=tuple(float(w) for w in half_width),
        curves=tuple(tuple(float(v) for v in curves[:, column[h]]) for h in hkl),
        sg_zero_deg=tuple(zeros),
        report_mode=report_mode,
        klar_window_deg=() if klar is None else _klar_window_deg(lever, semi, klar),
    )


def _klar_window_deg(
    lever: NDArray[np.float64], semiangle: float, klar: BeamSelection
) -> tuple[float, ...]:
    """Each reflection's Klar acceptance half-width in tilt (degrees) about a frame centre.

    With ``Sg ~ lever * (alpha - crossing)`` over a frame and ``sg_sweep = lever * semiangle``,
    ``|Sg| / sg_sweep < rsg`` reads ``|crossing - centre| < rsg * semiangle`` and
    ``sg_sweep - |Sg| > dsg`` reads ``|crossing - centre| < semiangle - dsg / lever``. The window is
    the tighter of the two, floored at 0 (a reflection on the rock axis is never accepted).
    """
    with np.errstate(divide="ignore"):
        dsg_limit = semiangle - np.rad2deg(klar.dsg / lever)
    window = np.clip(np.minimum(klar.rsg * semiangle, dsg_limit), 0.0, None)
    return tuple(float(w) for w in window)


def _tilt_intensities(
    plan: OrientationPlan,
    fgb: Tensor,
    thickness: Tensor,
    method: SolverMethod,
    absorption: Absorption,
    max_batch: int | None,
) -> NDArray[np.float64]:
    """Per-tilt intensities ``(tilts, beams)`` of ``plan`` at its first thickness, unreduced.

    Assembles and solves the tilts in blocks: a wide sweep has many tilts over many beams, and
    building every tilt's ``(N, N)`` Bloch matrix at once would not fit in memory. The block size is
    the engine's own memory-safe bound for this beam count.
    """
    n_beams = int(plan.beam_hkl.shape[0])
    block = max_batch if max_batch is not None else memory_safe_max_batch(n_beams)
    thicknesses = thickness.to(device=fgb.device, dtype=torch.float32)
    parts = []
    with torch.no_grad():
        for start in range(0, len(plan.beam_plans), block):
            batch = stack_beam_plans(plan.beam_plans[start : start + block])
            amplitudes = propagate(
                build_bloch_systems(batch, fgb, absorption),
                thicknesses,
                method=method,
                max_batch=block,
            )
            parts.append((amplitudes.abs() ** 2)[:, 0, :].cpu().numpy().astype(np.float64))
    return np.concatenate(parts, axis=0)


def _median_width(curves: NDArray[np.float64], sg: NDArray[np.float64], fraction: float) -> float:
    """Median ``Sg`` span holding ``fraction`` of each fully resolved rocking curve.

    ``curves`` and ``sg`` are ``(tilts, reflections)`` along one sweep. A reflection counts only
    when its intensity has died away at both ends of the sweep (under 5% of its peak), so its whole
    curve is in view. Raises when none qualifies.
    """
    lo, hi = (1.0 - fraction) / 2.0, (1.0 + fraction) / 2.0
    widths = []
    for n in range(curves.shape[1]):
        curve = curves[:, n]
        total = curve.sum()
        if total <= 0.0 or max(curve[0], curve[-1]) > 0.05 * curve.max():
            continue
        cumulative = np.cumsum(curve) / total
        a = int(np.searchsorted(cumulative, lo))
        b = int(np.searchsorted(cumulative, hi))
        widths.append(abs(float(sg[b, n] - sg[a, n])))
    if not widths:
        raise ValueError("no reflection's whole rocking curve was resolved; cannot measure a width")
    return float(np.median(widths))


# Above this share of scored reflections dropped for spilling out of their frame, the run suggests
# wider virtual frames.
_DROP_WARNING_FRACTION = 0.05

# Above this fraction of scored reflections with a rocking curve wider than a whole virtual
# frame, the frames are too narrow to integrate the data and the run says so.
_BROAD_WARNING_FRACTION = 0.10


def _warn_if_broad(
    scored: tuple[OrientationPlanLike, ...],
    cell: NDArray[np.float64],
    width: float,
    rocking: RockingCurve,
) -> None:
    """Warn when many scored reflections' rocking curves are wider than a virtual frame.

    A reflection's curve spans ``width`` in ``Sg``, which is ``width / lever`` radians of tilt
    (:func:`~diffBloch.preprocess.integrated.sg_per_radian`); wider than the frame's ``2 *
    semiangle`` means no single frame can integrate it.
    """
    frame = 2.0 * rocking.integration.semiangle
    n_scored = n_broad = 0
    for op in scored:
        hkl = op.alignment.hkl.cpu().numpy().astype(np.int64)
        if hkl.shape[0] == 0:
            continue
        g = g_vectors(hkl, orientation_basis(cell, np.asarray(op.orientation)))
        lever = sg_per_radian(g, rocking.integration)
        with np.errstate(divide="ignore"):
            angular = np.rad2deg(width / lever)
        n_scored += hkl.shape[0]
        n_broad += int(np.count_nonzero(angular > frame))
    if n_scored and n_broad / n_scored > _BROAD_WARNING_FRACTION:
        _log.warning(
            "%d of %d scored reflections (%.0f%%) have rocking curves wider than a virtual frame "
            "(%.3g deg); no single frame integrates them. Consider bigger virtual frames.",
            n_broad,
            n_scored,
            100.0 * n_broad / n_scored,
            frame,
        )


def _drop_spilling(
    plan: Plan,
    op: OrientationPlanLike,
    fgb: Tensor,
    *,
    engine: RefinementEngine,
    rocking: RockingCurve,
    solve_g_max: float,
    sg_max: float,
    method: SolverMethod,
    absorption: Absorption,
    max_batch: int | None,
    fraction: float,
) -> tuple[OrientationPlanLike, int]:
    """``op`` keeping only scored reflections with at least ``fraction`` of their curve inside it,
    and how many it dropped.

    Simulates the frame over its own sweep plus one semiangle more on each side, at its own tilt
    spacing, and measures how much of each scored reflection's intensity falls inside the frame.
    """
    hkl = op.alignment.hkl.cpu().numpy().astype(np.int64)
    if hkl.shape[0] == 0:
        return op, 0
    semi = rocking.integration.semiangle
    step = 2.0 * semi / max(rocking.sampling - 1, 1)
    angles = np.linspace(-2.0 * semi, 2.0 * semi, int(round(4.0 * semi / step)) + 1)
    tilts = _rx(angles)
    grid = plan.structure_factor_grid
    # The refinement's own coupled solve (per-segment beam sets over the sweep), with beams coupled
    # only up to the largest |g| this frame scores, whatever the configured g_max: a quick check.
    largest = float(np.max(np.linalg.norm(hkl @ np.asarray(grid.reciprocal_basis), axis=1)))
    segments = build_coupling_segments(
        UnionCoupling(g_max=largest * 1.0001, sg_max=sg_max),
        np.asarray(grid.structure_factor_hkl, dtype=np.int64),
        cell=np.asarray(grid.cell),
        orientation=np.asarray(op.orientation),
        tilts=tilts,
        energy=op.energy,
        u0=op.u0,
    )
    wide = CoupledOrientationPlan.build(
        grid,
        [(segment.union_hkl, segment.covered_tilt_indices) for segment in segments],
        op.pattern,
        energy=op.energy,
        thickness=op.thickness,
        u0=op.u0,
        orientation=np.asarray(op.orientation),
        tilts=tilts,
    )
    with torch.no_grad():
        curves = (
            engine.segmented_rocking_curve(wide, fgb, op.thickness)[:, 0, :]
            .cpu()
            .numpy()
            .astype(np.float64)
        )
    column = {tuple(h): i for i, h in enumerate(wide.beam_hkl.tolist())}
    inside = np.abs(angles) <= semi + 1e-9
    keep = []
    for row, h in enumerate(hkl.tolist()):
        if tuple(h) not in column or (np.asarray(h) == 0).all():
            keep.append(row)
            continue
        curve = curves[:, column[tuple(h)]]
        total = curve.sum()
        if total <= 0.0 or curve[inside].sum() / total >= fraction:
            keep.append(row)
    return _keep_rows(op, keep), int(hkl.shape[0] - len(keep))


def _inside_outer_limit(
    ops: tuple[OrientationPlanLike, ...],
    index: int,
    g: NDArray[np.float64],
    op: OrientationPlanLike,
    rocking: RockingCurve,
) -> NDArray[np.bool_]:
    """Keep-mask limiting a dataset's first and last frame on their open (outer) side.

    An interior frame only wins crossings within about half a frame step of its centre: past that,
    the neighbour is more central. The first and last frame of a dataset have no neighbour on their
    outer side, so without this they would keep crossings out to the frame edge, where a camera can
    record the end of the sweep twice. Holding the outer side to the same half-step as the inner side
    treats every frame alike. Interior frames, and a dataset of one frame, keep everything.

    The frame's local tilt runs in the same direction as the PETS alpha axis, so the neighbour lies
    on the side of ``sign(alpha_neighbour - alpha_frame)``.
    """
    dataset = op.pattern.dataset
    before = index > 0 and ops[index - 1].pattern.dataset == dataset
    after = index + 1 < len(ops) and ops[index + 1].pattern.dataset == dataset
    if before == after:
        return np.ones(g.shape[0], dtype=np.bool_)
    neighbour = ops[index - 1] if before else ops[index + 1]
    step = float(neighbour.pattern.alpha_degrees) - float(op.pattern.alpha_degrees)
    if step == 0.0:
        return np.ones(g.shape[0], dtype=np.bool_)
    crossing = bragg_crossing_angles(
        g, energy=op.energy, u0=op.u0, semiangle=rocking.integration.semiangle
    )
    inward = np.sign(step) * crossing  # positive toward the neighbour
    with np.errstate(invalid="ignore"):
        too_far_out: NDArray[np.bool_] = inward < -abs(step) / 2.0
    return ~too_far_out


def _keep_rows(op: OrientationPlanLike, rows: list[int]) -> OrientationPlanLike:
    alignment = op.alignment
    index = torch.as_tensor(rows, dtype=torch.int64, device=alignment.hkl.device)
    return replace(
        op,
        alignment=AlignmentPlan(
            hkl=alignment.hkl[index],
            solution_index=alignment.solution_index[index],
            pattern_index=alignment.pattern_index[index],
        ),
    )


def rocking_curve_width(
    plan: Plan,
    fgb: Tensor,
    spec: IntegratedScoring,
    *,
    rocking: RockingCurve,
    solve_g_max: float,
    sg_max: float,
    method: SolverMethod = "matrix_exp",
    absorption: Absorption = NO_ABSORPTION,
    max_batch: int | None = None,
) -> float:
    """The median ``Sg`` span (A^-1) holding ``spec.fraction`` of a reflection's rocking curve.

    ``spec.width_rotations`` evenly spaced rotations are solved (no mosaic samples) over a sweep
    ``spec.sweep_factor`` times the frame's semiangle at the frame's own tilt spacing. Every scored
    reflection whose Bragg crossing lies inside the frame -- so its whole curve is inside the wide
    sweep -- and whose intensity has died away at both ends of that sweep contributes the ``Sg``
    distance between the points where its cumulative intensity reaches ``(1 - fraction) / 2`` and
    ``(1 + fraction) / 2``. Raises when no reflection qualifies.
    """
    ops = require_built_plans(plan)
    picks = np.unique(np.linspace(0, len(ops) - 1, min(spec.width_rotations, len(ops))).round())
    widths: list[float] = []
    for pick in picks.astype(int):
        widths.extend(
            _widths_for(
                plan,
                ops[pick],
                fgb,
                spec,
                rocking=rocking,
                solve_g_max=solve_g_max,
                sg_max=sg_max,
                method=method,
                absorption=absorption,
                max_batch=max_batch,
            )
        )
    if not widths:
        raise ValueError("no reflection's whole rocking curve was resolved; cannot measure a width")
    return float(np.median(widths))


def _widths_for(
    plan: Plan,
    op: OrientationPlanLike,
    fgb: Tensor,
    spec: IntegratedScoring,
    *,
    rocking: RockingCurve,
    solve_g_max: float,
    sg_max: float,
    method: SolverMethod,
    absorption: Absorption,
    max_batch: int | None,
) -> list[float]:
    grid = plan.structure_factor_grid
    cell = np.asarray(grid.cell)
    semiangle = rocking.integration.semiangle
    wide_semiangle = spec.sweep_factor * semiangle
    n_wide = int(round((rocking.sampling - 1) * spec.sweep_factor)) + 1
    tilts = rocking_curve_tilts(wide_semiangle, n_wide)
    basis = orientation_basis(cell, np.asarray(op.orientation))

    beam_hkl = _wide_sweep_beams(grid, basis, tilts, op, solve_g_max, sg_max)
    wide = OrientationPlan.build(
        grid,
        beam_hkl,
        op.pattern,
        energy=op.energy,
        thickness=op.thickness,
        u0=op.u0,
        orientation=np.asarray(op.orientation),
        tilts=tilts,
        build_alignment=False,
    )
    curves = _tilt_intensities(wide, fgb, op.thickness, method, absorption, max_batch)

    beam_row = {tuple(b): i for i, b in enumerate(beam_hkl.tolist())}
    scored = [h for h in op.alignment.hkl.cpu().numpy().tolist() if tuple(h) in beam_row]
    if not scored:
        return []
    hkl = np.asarray(scored, dtype=np.int64)
    g = g_vectors(hkl, basis)
    crossing = bragg_crossing_angles(g, energy=op.energy, u0=op.u0, semiangle=semiangle)
    tilted = np.einsum("tij,nj->tni", tilts, g)
    sg = excitation_errors(tilted.reshape(-1, 3), op.energy, u0=op.u0).reshape(n_wide, -1)

    lo, hi = (1.0 - spec.fraction) / 2.0, (1.0 + spec.fraction) / 2.0
    widths: list[float] = []
    for n, h in enumerate(hkl.tolist()):
        if np.isnan(crossing[n]):
            continue
        curve = curves[:, beam_row[tuple(h)]]
        total = curve.sum()
        if total <= 0.0 or max(curve[0], curve[-1]) > 0.05 * curve.max():
            continue  # the curve has not died away inside the wide sweep
        cumulative = np.cumsum(curve) / total
        a = int(np.searchsorted(cumulative, lo))
        b = int(np.searchsorted(cumulative, hi))
        widths.append(abs(float(sg[b, n] - sg[a, n])))
    return widths


def _wide_sweep_beams(
    grid: StructureFactorGrid,
    basis: NDArray[np.float64],
    tilts: NDArray[np.float64],
    op: OrientationPlanLike,
    solve_g_max: float,
    sg_max: float,
) -> NDArray[np.int64]:
    """Beams excited (``|Sg| < sg_max``, ``|g| < solve_g_max``) anywhere in the wide sweep."""
    hkl = np.asarray(grid.structure_factor_hkl, dtype=np.int64)
    g = g_vectors(hkl, basis)
    pool = np.linalg.norm(g, axis=1) < solve_g_max
    hkl, g = hkl[pool], g[pool]
    excited = np.zeros(len(hkl), dtype=bool)
    for tilt in tilts:
        excited |= np.abs(excitation_errors(g @ tilt.T, op.energy, u0=op.u0)) < sg_max
    excited |= (hkl == 0).all(axis=1)
    return hkl[excited]

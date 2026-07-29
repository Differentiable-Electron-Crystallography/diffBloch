"""Simulation-convergence testing over ``g_max``, ``sg_max``, and tilt steps."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from torch import Tensor

from diffBloch.core.solver import SolverMethod
from diffBloch.engine.plan import CoupledOrientationPlan, OrientationPlan, StructureFactorGrid
from diffBloch.observability import (
    NULL_LOGGER,
    ConvergencePassStarted,
    ConvergenceSweepStarted,
    ConvergenceTrial,
    Logger,
)
from diffBloch.preprocess.experiment import RefinementSetup
from diffBloch.preprocess.pipeline import PlanStep, as_step
from diffBloch.preprocess.plan import Plan, require_built_plans, require_orientation_plans
from diffBloch.preprocess.steps.convergence import converge_scalar, simulation_rfactor
from diffBloch.preprocess.steps.coupling import couple_beams
from diffBloch.preprocess.steps.rocking_curve import integrate_rocking_curve
from diffBloch.specs import (
    ConvergenceTest,
    ConvergenceTolerance,
    RockingCurve,
    SegmentedUnionCoupling,
)

__all__ = ["ConvergenceState", "converge_numerics", "run_convergence"]


@dataclass(frozen=True)
class ConvergenceState:
    """The three numerical controls varied by a convergence test."""

    g_max: float
    sg_max: float
    tilt_steps: int


def converge_numerics(
    test: ConvergenceTest,
    rocking: RockingCurve,
    simulation: SegmentedUnionCoupling,
    refinement: RefinementSetup,
    tolerance: ConvergenceTolerance,
    *,
    method: SolverMethod = "matrix_exp",
    logger: Logger = NULL_LOGGER,
) -> PlanStep:
    """Return a step that converges ``g_max``, ``sg_max``, and rocking-curve tilt steps."""

    def run(plan: Plan) -> Plan:
        converged, _state = run_convergence(
            plan,
            ConvergenceState(
                g_max=simulation.g_max,
                sg_max=simulation.sg_max,
                tilt_steps=rocking.sampling,
            ),
            test,
            rocking,
            simulation,
            refinement,
            tolerance,
            method=method,
            logger=logger,
        )
        return converged

    return as_step(
        "converge_numerics",
        {
            "test": test,
            "rocking": rocking,
            "simulation": simulation,
            "tolerance": tolerance,
            "method": method,
        },
        run,
    )


def run_convergence(
    plan: Plan,
    state: ConvergenceState,
    test: ConvergenceTest,
    rocking: RockingCurve,
    simulation: SegmentedUnionCoupling,
    refinement: RefinementSetup,
    tolerance: ConvergenceTolerance,
    *,
    method: SolverMethod = "matrix_exp",
    logger: Logger = NULL_LOGGER,
) -> tuple[Plan, ConvergenceState]:
    """Run coordinate sweeps until all three simulation controls are self-stable."""

    steps = {
        "g_max": test.g_max_step,
        "sg_max": test.sg_max_step,
        "tilt_steps": float(test.tilt_steps_step),
    }
    reciprocal_basis = np.asarray(plan.structure_factor_grid.reciprocal_basis, dtype=np.float64)
    scored_g_max = max(
        float(
            np.linalg.norm(
                np.asarray(op.beam_hkl, dtype=np.float64) @ reciprocal_basis,
                axis=1,
            ).max()
        )
        for op in require_orientation_plans(plan)
    )
    comparison_hkl: tuple[Tensor, ...] | None = None

    def build(g_max: float, sg_max: float, tilt_steps: float) -> Plan:
        grid = StructureFactorGrid.from_cell_for_beam_cutoff(
            np.asarray(plan.structure_factor_grid.cell, dtype=np.float64),
            max(g_max, scored_g_max),
        )
        expanded = replace(
            plan,
            structure_factor_grid=grid,
            orientations=tuple(
                OrientationPlan.build(
                    grid,
                    np.asarray(op.beam_hkl, dtype=np.int64),
                    op.pattern,
                    energy=op.energy,
                    thickness=op.thickness,
                    u0=op.u0,
                    orientation=op.orientation,
                    tilt_reduction=op.tilt_reduction,
                )
                for op in require_orientation_plans(plan)
            ),
        )
        integrated = integrate_rocking_curve(
            replace(rocking, sampling=int(round(tilt_steps)))
        )(expanded)
        coupled = couple_beams(
            replace(simulation, g_max=g_max, sg_max=sg_max)
        )(integrated)
        if comparison_hkl is None:
            return coupled
        orientations = tuple(
            _include_fixed_hkl(coupled.structure_factor_grid, op, fixed)
            for op, fixed in zip(
                require_built_plans(coupled), comparison_hkl, strict=True
            )
        )
        return replace(coupled, orientations=orientations)

    values = {
        "g_max": state.g_max,
        "sg_max": state.sg_max,
        "tilt_steps": float(state.tilt_steps),
    }
    initial_plan = build(**values)
    comparison_hkl = tuple(op.alignment.hkl for op in require_built_plans(initial_plan))
    n_compared_hkl = sum(int(hkl.shape[0]) for hkl in comparison_hkl)
    measure = simulation_rfactor(
        refinement,
        method=method,
        comparison_hkl=comparison_hkl,
    )

    for pass_index in range(test.num_passes):
        logger.report(
            ConvergencePassStarted(
                pass_index=pass_index + 1,
                g_max=values["g_max"],
                sg_max=values["sg_max"],
                tilt_steps=int(round(values["tilt_steps"])),
                r_factor_threshold=tolerance.r_factor_threshold,
                n_orientations=len(comparison_hkl),
            )
        )
        order = (
            ("g_max", "tilt_steps", "sg_max")
            if pass_index == 0
            else ("tilt_steps", "g_max", "sg_max")
        )
        for control in order:
            logger.report(
                ConvergenceSweepStarted(control=control, pass_index=pass_index + 1)
            )
            start = values[control]
            trial_index = 0

            def compare(
                previous: float,
                candidate: float,
                *,
                name: str = control,
                pass_number: int = pass_index + 1,
            ) -> float:
                nonlocal trial_index
                previous_values = {**values, name: previous}
                candidate_values = {**values, name: candidate}
                r_factor = measure(
                    build(**previous_values),
                    build(**candidate_values),
                )
                logger.report(
                    ConvergenceTrial(
                        control=name,
                        trial_index=trial_index,
                        pass_index=pass_number,
                        previous=previous,
                        candidate=candidate,
                        r_factor=r_factor,
                        n_compared_hkl=n_compared_hkl,
                    )
                )
                trial_index += 1
                return r_factor

            values[control] = converge_scalar(
                lambda value: value,
                compare,
                tolerance,
                start=start,
                step=steps[control],
                accept_converged_candidate=False,
            )

    settled = ConvergenceState(
        g_max=values["g_max"],
        sg_max=values["sg_max"],
        tilt_steps=int(round(values["tilt_steps"])),
    )
    return build(settled.g_max, settled.sg_max, settled.tilt_steps), settled


def _include_fixed_hkl(
    grid: StructureFactorGrid,
    orientation: OrientationPlan | CoupledOrientationPlan,
    fixed_hkl: Tensor,
) -> CoupledOrientationPlan:
    """Keep the initial scored reflections in every adaptive-union segment."""
    if not isinstance(orientation, CoupledOrientationPlan):
        raise TypeError("convergence with segmented unions requires coupled orientation plans")
    fixed = np.asarray(fixed_hkl, dtype=np.int64)
    segments: list[tuple[NDArray[np.int64], list[int]]] = [
        (
            np.asarray(
                np.unique(
                    np.concatenate(
                        [np.asarray(segment.plan.beam_hkl, dtype=np.int64), fixed],
                        axis=0,
                    ),
                    axis=0,
                ),
                dtype=np.int64,
            ),
            np.asarray(segment.cover, dtype=np.int64).tolist(),
        )
        for segment in orientation.segments
    ]
    return CoupledOrientationPlan.build(
        grid,
        segments,
        orientation.pattern,
        energy=orientation.energy,
        thickness=orientation.thickness,
        u0=orientation.u0,
        orientation=orientation.orientation,
        tilts=np.asarray(orientation.tilts, dtype=np.float64),
        tilt_reduction=orientation.tilt_reduction,
        scored_hkl=fixed,
    )

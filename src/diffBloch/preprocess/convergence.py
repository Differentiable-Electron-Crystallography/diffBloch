"""Convergence testing: stop growing a simulation-accuracy knob once the pattern stops changing.

A convergence sweep is *self-referential* -- unlike ``fit_orientation`` / ``fit_thickness`` (which
match the simulation to *observed* data), it asks whether two *consecutive* simulations still
differ.
:func:`simulation_converged` is that comparison expressed as a
:data:`~diffBloch.preprocess.pipeline.ConvergenceCheck` (a ``(previous, current) -> bool`` that
:func:`~diffBloch.preprocess.pipeline.iterate_until` drives to a fixpoint): it simulates both Plans
and returns whether their mean per-orientation R-factor has dropped below a tolerance.

Faithful to ``diffBloch_private`` ``convergence_testing._compute_step_rfactor`` (per-orientation
``rbragg_abs`` between two simulation tables, averaged). See
``design/decisions/stage11-convergence.md``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from diffBloch.core.losses import optimal_scale, rbragg
from diffBloch.core.products import BlochSolution
from diffBloch.core.solver import Method
from diffBloch.preprocess.experiment import RefinementSetup
from diffBloch.preprocess.pipeline import ConvergenceCheck
from diffBloch.preprocess.plan import Plan
from diffBloch.preprocess.scoring import build_engine
from diffBloch.specs import ConvergenceTolerance

__all__ = ["simulation_converged"]

# The R-factor compares two simulations, not simulation-vs-data, so there is no measurement noise to
# weight by; a near-zero sigma makes ``rbragg`` effectively unweighted while keeping its
# ``I > 3*sigma`` mask inclusive (the private uses sigmas = 1e-10 for the same reason).
_UNWEIGHTED_SIGMA = 1e-10


def simulation_converged(
    refinement: RefinementSetup,
    tolerance: ConvergenceTolerance,
    *,
    method: Method = "matrix_exp",
) -> ConvergenceCheck:
    """Return a ``(previous, current) -> bool`` check: have consecutive simulations stabilised?

    ``refinement`` (the read-only structure context) is captured and rejoined to each Plan via
    :func:`build_engine`; ``method`` configures the solver. The returned check simulates both Plans,
    computes the scale-optimised ``rbragg`` R-factor between the two on each orientation's shared
    reflections, averages over orientations, and returns whether that mean is below
    ``tolerance.r_factor_threshold``. The comparison is a control-flow decision, not a gradient
    path, so the simulated intensities are detached.

    The two Plans must describe the same orientations in the same order (a convergence step rebuilds
    each orientation, changing only its beam set), and each pair must share at least one reflection
    (the retained 000 guarantees this in practice).
    """

    def check(previous: Plan, current: Plan) -> bool:
        previous_solutions = _simulate(previous, refinement, method)
        current_solutions = _simulate(current, refinement, method)
        if len(previous_solutions) != len(current_solutions):
            raise ValueError("convergence check requires the two Plans to share their orientations")
        r_factors = [
            _orientation_rfactor(prev, curr)
            for prev, curr in zip(previous_solutions, current_solutions, strict=True)
        ]
        return float(np.mean(r_factors)) < tolerance.r_factor_threshold

    return check


def _simulate(plan: Plan, refinement: RefinementSetup, method: Method) -> tuple[BlochSolution, ...]:
    return build_engine(plan, refinement, method=method).simulate(refinement.params)


def _orientation_rfactor(previous: BlochSolution, current: BlochSolution) -> float:
    """Scale-optimised ``rbragg`` between two simulations on their shared reflections.

    Each table is ``(T, N)`` over its own beam set; the beam sets differ between the two
    simulations, so the comparison is restricted to the reflections both contain. A single intensity
    scale (shared across thicknesses, matching ``optimal_scale``) maps ``current`` onto ``previous``
    before the R-factor, since the two simulations have no common normalization.
    """
    previous_index, current_index = _shared_reflections(previous.beam_hkl, current.beam_hkl)
    previous_intensity = previous.intensities.detach().cpu()[:, previous_index].reshape(-1)
    current_intensity = current.intensities.detach().cpu()[:, current_index].reshape(-1)
    sigmas = torch.full_like(previous_intensity, _UNWEIGHTED_SIGMA)
    _, r_value = optimal_scale(current_intensity, previous_intensity, sigmas, metric=rbragg)
    return float(r_value)


def _shared_reflections(previous_hkl: Tensor, current_hkl: Tensor) -> tuple[Tensor, Tensor]:
    """Indices into each beam set selecting the reflections present in both, in a shared order."""
    previous_rows = previous_hkl.detach().cpu().numpy()
    current_rows = current_hkl.detach().cpu().numpy()
    previous_position = {tuple(row): i for i, row in enumerate(previous_rows)}
    previous_index: list[int] = []
    current_index: list[int] = []
    for j, row in enumerate(current_rows):
        i = previous_position.get(tuple(row))
        if i is not None:
            previous_index.append(i)
            current_index.append(j)
    if not previous_index:
        raise ValueError("convergence check found no reflections shared by the two simulations")
    return torch.tensor(previous_index), torch.tensor(current_index)

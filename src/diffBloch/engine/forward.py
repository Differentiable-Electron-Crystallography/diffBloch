"""The forward composition spine: raw parameters -> simulated diffraction -> scalar objective.

A :class:`RefinementEngine` holds the refinement-invariant plans (constraint spec, ASU-expansion
plan, the shared scattering grid, and one :class:`~diffBloch.engine.plan.OrientationPlan` per
rotation) and maps :class:`~diffBloch.params.RefinableParams` to a differentiable objective:

    constrain -> expand ASU -> structure_factors (Fgb on the shared grid)
              -> per orientation: build_bloch_system -> propagate -> intensities -> align -> loss

``objective`` / ``simulate`` are pure and differentiable; ``refine`` delegates to the quarantined
imperative loop in :mod:`diffBloch.engine.refine`. ``from_config`` / ``from_experiment``
construction is deferred until beam selection (stage 11) exists; engines are assembled from explicit
per-orientation beam sets.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from diffBloch.core.constraints import positive
from diffBloch.core.dynamical import build_bloch_system, build_bloch_systems, stack_beam_plans
from diffBloch.core.losses import optimal_scale
from diffBloch.core.products import (
    AlignedIntensities,
    BlochSolution,
    align,
    intensities,
    reduce_tilts,
)
from diffBloch.core.scattering import structure_factors
from diffBloch.core.solver import FloatFormat, Method, precision_dtypes, propagate
from diffBloch.core.symmetry import AsuExpansionPlan, expand_asu
from diffBloch.engine.plan import (
    OrientationPlanLike,
    ScatteringGrid,
    SegmentedOrientationPlan,
)
from diffBloch.engine.refine import (
    ObjectiveComponent,
    ObjectiveValue,
    OptimizerName,
    RefinementResult,
    run_refinement,
)
from diffBloch.observability import NULL_LOGGER, Logger
from diffBloch.params import ConstraintSpec, RefinableParams, constrain

__all__ = [
    "LossFn",
    "RefinementEngine",
]

# A loss reduces one orientation's aligned intensities to a scalar term (calculated vs observed).
# The engine sums these per-orientation terms into the scalar ``objective`` ``refine`` minimises.
type LossFn = Callable[[AlignedIntensities], Tensor]


@dataclass(frozen=True)
class RefinementEngine:
    """Forward from raw parameters to a differentiable scalar objective (plus a refinement driver).

    Holds the refinement-invariant context: the constraint ``spec``, the ASU-expansion ``asu_plan``,
    the ASU atomic ``numbers``, the shared ``grid``, the per-rotation ``orientations`` (each
    carrying its own frozen ``thickness``), the per-orientation ``loss``, and the propagation
    ``method``.
    """

    spec: ConstraintSpec
    asu_plan: AsuExpansionPlan
    numbers: Tensor
    grid: ScatteringGrid
    orientations: tuple[OrientationPlanLike, ...]
    loss: LossFn
    method: Method = "matrix_exp"
    # Solve numeric field. "fp64" everywhere by default (byte-identical to complex128). "fp32"
    # (complex64) is a search-time knob, set only on the transient scoring engines the preprocess
    # fits build; the terminal estimators (objective/refine here, run_inference via build_engine)
    # never enable it, so the reproducible pinned result stays fp64. See core.solver.propagate.
    precision: FloatFormat = "fp64"

    def simulate(self, params: RefinableParams) -> tuple[BlochSolution, ...]:
        """Return the calculated :class:`BlochSolution` for every orientation (no loss)."""
        if not self.orientations:
            raise ValueError("engine has no orientations to evaluate")
        fgb = self._structure_factors(params)
        return tuple(self._solve(o, fgb, self._thickness_for(o, params)) for o in self.orientations)

    def fgb(self, params: RefinableParams) -> Tensor:
        """The calculated structure factors ``F_gb`` on the shared grid.

        The orientation-invariant part of the forward model: compute once and reuse across
        orientations (e.g. when scoring many trial orientations of one structure).
        """
        return self._structure_factors(params)

    def score_orientation(self, orientation: OrientationPlanLike, fgb: Tensor) -> Tensor:
        """Scaling-optimised weighted-R2 (wR2) for one orientation against its observed pattern.

        Runs the forward Bloch simulation for ``orientation`` from a precomputed ``fgb``
        (:meth:`fgb`), aligns calculated vs observed intensities, and grid-searches the intensity
        scale minimising wR2 (:func:`diffBloch.core.losses.optimal_scale`). With multiple
        thicknesses the best-fitting thickness's score is returned -- thickness is a nuisance when
        scoring orientation (the private preprocess scored the first thickness). This is the
        objective ``fit_orientation`` minimises.
        """
        return self.score_orientation_per_thickness(orientation, fgb).min()

    def score_orientation_per_thickness(
        self, orientation: OrientationPlanLike, fgb: Tensor
    ) -> Tensor:
        """Scaling-optimised wR2 for each of the orientation's thicknesses (shape ``(T,)``).

        One forward Bloch simulation from a precomputed ``fgb`` (:meth:`fgb`) covers all ``T``
        thicknesses at once: the expensive eigendecomposition depends only on the orientation and
        ``fgb``, while thickness enters only the cheap propagation tail. The calculated and observed
        intensities are aligned once (alignment is thickness-independent), then for each thickness
        the intensity scale minimising wR2 is found (:func:`diffBloch.core.losses.optimal_scale`).

        ``fit_thickness`` grid-searches this vector and bakes the lowest-wR2 thickness;
        :meth:`score_orientation` collapses it with ``.min()`` (thickness is a nuisance there).
        """
        aligned = align(
            self._solve(orientation, fgb, orientation.thickness),
            orientation.pattern,
            orientation.alignment,
        )
        return torch.stack(
            [
                optimal_scale(aligned.calculated[t], aligned.observed[t], aligned.sigmas[t])[1]
                for t in range(aligned.calculated.shape[0])
            ]
        )

    def objective_value(self, params: RefinableParams) -> ObjectiveValue:
        """Return the objective as a scalar total plus named scalar components.

        Differentiable in ``params``. The only component today is ``"diffraction"``; this shape is
        the public seam for future geometric/restraint terms without making the optimizer know their
        details.
        """
        if not self.orientations:
            raise ValueError("engine has no orientations to evaluate")
        fgb = self._structure_factors(params)
        total = params.asu_positions.new_zeros(())
        for orientation in self.orientations:
            solution = self._solve(orientation, fgb, self._thickness_for(orientation, params))
            aligned = align(solution, orientation.pattern, orientation.alignment)
            term = self.loss(aligned)
            # Catch a non-reducing loss here, where the mistake is, rather than letting a
            # non-scalar surface much later as an opaque ``backward()`` failure.
            if term.ndim != 0:
                raise ValueError(f"loss must return a scalar, got shape {tuple(term.shape)}")
            total = total + term
        return ObjectiveValue({"diffraction": ObjectiveComponent(raw=total)})

    def objective(self, params: RefinableParams) -> Tensor:
        """Return the scalar objective: the per-orientation ``loss`` summed over orientations.

        Differentiable in ``params``; this is the quantity ``refine`` minimises.
        """
        return self.objective_value(params).total

    def refine(
        self,
        params: RefinableParams,
        *,
        steps: int,
        targets: Sequence[str] = ("positions", "adp"),
        optimizer: OptimizerName = "lbfgs",
        lr: float = 1e-3,
        logger: Logger = NULL_LOGGER,
    ) -> RefinementResult:
        """Optimize the selected ``targets`` to minimise the objective; return a result snapshot.

        Delegates to :func:`diffBloch.engine.refine.run_refinement` over this engine's pure
        ``objective``: the caller's ``params`` are never mutated. Single shared ``lr`` for now
        (per-group rates deferred); ``least_squares`` and component ``activate`` are likewise
        deferred. ``logger`` streams per-step and
        completion events (default :data:`NULL_LOGGER` = no-op, result unchanged).
        """
        return run_refinement(
            self.objective,
            params,
            steps=steps,
            targets=targets,
            optimizer=optimizer,
            lr=lr,
            logger=logger,
        )

    def _thickness_for(self, orientation: OrientationPlanLike, params: RefinableParams) -> Tensor:
        """The thickness ``(T,)`` the forward model uses for one orientation.

        There are two sources of thickness, and this picks between them:

        - **Normally**, each orientation carries its own fixed thickness
          (``orientation.thickness``), because the specimen presents a different path length at each
          tilt. It is seeded from the
          sample thickness and later replaced by the best-fitting value found by ``fit_thickness``.
        - **When the caller is also refining thickness**, ``params.thickness_raw`` is set, and it
          overrides the per-orientation value for every orientation (a single refined thickness
          shared across all of them). It is an unconstrained real number mapped to a positive
          thickness by ``positive`` -- the same mapping :func:`diffBloch.params.constrain` uses --
          so selecting the ``"thickness"`` refine target actually changes the simulation.

        ``params.thickness_raw is None`` means "not refining thickness", so the per-orientation
        value is used. Letting thickness vary per orientation while being refined (a learned
        ``theta -> thickness`` model) is deliberately deferred future work.
        """
        if params.thickness_raw is None:
            return orientation.thickness
        return positive(params.thickness_raw)

    def _structure_factors(self, params: RefinableParams) -> Tensor:
        state = constrain(params, self.spec)
        device = state.positions.device  # the active (params) device; co-locate invariants here
        expanded = expand_asu(
            self.asu_plan,
            state.positions,
            numbers=self.numbers.to(device),
            uij=state.uij_star,
            occupancies=state.occupancies,
        )
        assert expanded.numbers is not None and expanded.uij is not None
        assert expanded.occupancies is not None
        return structure_factors(
            expanded.positions,
            expanded.numbers,
            expanded.occupancies,
            expanded.uij,
            hkl=self.grid.grid_hkl.to(device),
            reciprocal_basis=self.grid.reciprocal_basis.to(device),
            cell_volume=self.grid.cell_volume,
            g_max=self.grid.g_max,
        )

    def _solve(
        self, orientation: OrientationPlanLike, fgb: Tensor, thicknesses: Tensor
    ) -> BlochSolution:
        if isinstance(orientation, SegmentedOrientationPlan):
            return self._solve_segmented(orientation, fgb, thicknesses)
        device = fgb.device  # fgb is param-derived; thicknesses/beam_hkl must co-locate with it
        real_dtype, _ = precision_dtypes(self.precision)
        thicknesses = thicknesses.to(device=device, dtype=real_dtype)
        beam_hkl = orientation.beam_hkl.to(device)
        # Untilted (length 1): the static solve, byte-identical to the pre-integration path.
        if len(orientation.beam_plans) == 1:
            amplitudes = propagate(
                build_bloch_system(orientation.beam_plans[0], fgb),
                thicknesses,
                method=self.method,
                precision=self.precision,
            )
            return BlochSolution.from_propagation(amplitudes, beam_hkl, thicknesses)
        # Rocking-curve integration: the tilts share this orientation's beam set, so ONE batched
        # solve over (B, N, N) replaces a Python loop of B single solves (the tilt-batching perf
        # path). The engine then reduces |psi|^2 over the tilt axis (incoherent rotation-frame
        # integration; PlainSum or a mosaicity broadening) via BlochSolution.integrate_batched.
        batch = stack_beam_plans(orientation.beam_plans)
        amplitudes = propagate(
            build_bloch_systems(batch, fgb),
            thicknesses,
            method=self.method,
            precision=self.precision,
        )  # (B, T, N)
        return BlochSolution.integrate_batched(
            amplitudes, beam_hkl, thicknesses, reduction=orientation.tilt_reduction
        )

    def _solve_segmented(
        self, plan: SegmentedOrientationPlan, fgb: Tensor, thicknesses: Tensor
    ) -> BlochSolution:
        """Solve each tilt-chunk on its own beam set, reassemble the union curve, then reduce.

        The tilt-dependent coupling path (Option A): each :class:`SegmentPlan` is solved over its
        covered tilts with the batched propagator, and its per-tilt intensities are scattered onto
        the shared ``(N_tilts, T, N_union)`` rocking curve (each tilt belongs to exactly one
        segment; a beam absent from a chunk stays 0 at that chunk's tilts). Only once the whole
        curve is reassembled is the tilt reduction applied -- the mosaicity window spans more tilts
        than any single chunk holds -- and the result is returned as an ordinary
        :class:`BlochSolution` over the union beam set, so ``align`` / scoring are unchanged.
        """
        device = fgb.device
        real_dtype, _ = precision_dtypes(self.precision)
        thicknesses = thicknesses.to(device=device, dtype=real_dtype)
        n_tilts = int(plan.tilts.shape[0])
        n_union = int(plan.beam_hkl.shape[0])
        n_thick = int(thicknesses.shape[0])
        curve = fgb.new_zeros((n_tilts, n_thick, n_union), dtype=thicknesses.dtype)
        for segment in plan.segments:
            batch = stack_beam_plans(segment.plan.beam_plans)
            amplitudes = propagate(
                build_bloch_systems(batch, fgb),
                thicknesses,
                method=self.method,
                precision=self.precision,
            )  # (C, T, n_seg)
            cover = segment.cover.to(device)
            union_index = segment.union_index.to(device)
            block = curve[cover]  # (C, T, n_union) gathered copy
            block[:, :, union_index] = intensities(amplitudes)
            curve[cover] = block
        total = reduce_tilts(curve, plan.tilt_reduction)  # (T, n_union)
        # The reassembled curve is an intensity sum, so its per-tilt amplitudes were never coherent;
        # store the magnitude sqrt(total) in the solve's complex format (complex128 under fp64 ->
        # byte-identical to today; complex64 under fp32, matching the static/batched fp32 paths).
        _, complex_dtype = precision_dtypes(self.precision)
        return BlochSolution(
            total.sqrt().to(complex_dtype), total, plan.beam_hkl.to(device), thicknesses
        )

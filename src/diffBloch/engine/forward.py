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
from diffBloch.core.dynamical import build_bloch_system
from diffBloch.core.losses import optimal_scale
from diffBloch.core.products import AlignedIntensities, BlochSolution, align
from diffBloch.core.scattering import structure_factors
from diffBloch.core.solver import Method, propagate
from diffBloch.core.symmetry import AsuExpansionPlan, expand_asu
from diffBloch.engine.plan import OrientationPlan, ScatteringGrid
from diffBloch.engine.refine import OptimizerName, RefinementResult, run_refinement
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
    orientations: tuple[OrientationPlan, ...]
    loss: LossFn
    method: Method = "matrix_exp"

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

    def score_orientation(self, orientation: OrientationPlan, fgb: Tensor) -> Tensor:
        """Scaling-optimised weighted-R2 (wR2) for one orientation against its observed pattern.

        Runs the forward Bloch simulation for ``orientation`` from a precomputed ``fgb``
        (:meth:`fgb`), aligns calculated vs observed intensities, and grid-searches the intensity
        scale minimising wR2 (:func:`diffBloch.core.losses.optimal_scale`). With multiple
        thicknesses the best-fitting thickness's score is returned -- thickness is a nuisance when
        scoring orientation (the private preprocess scored the first thickness). This is the
        objective ``fit_orientation`` minimises (``design/decisions/stage11-fit-orientation.md``).
        """
        return self.score_orientation_per_thickness(orientation, fgb).min()

    def score_orientation_per_thickness(self, orientation: OrientationPlan, fgb: Tensor) -> Tensor:
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

    def objective(self, params: RefinableParams) -> Tensor:
        """Return the scalar objective: the per-orientation ``loss`` summed over orientations.

        Differentiable in ``params``; this is the quantity ``refine`` minimises.
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
        return total

    def refine(
        self,
        params: RefinableParams,
        *,
        steps: int,
        targets: Sequence[str] = ("positions", "adp"),
        optimizer: OptimizerName = "lbfgs",
        lr: float = 1e-3,
    ) -> RefinementResult:
        """Optimize the selected ``targets`` to minimise the objective; return a result snapshot.

        Delegates to :func:`diffBloch.engine.refine.run_refinement` over this engine's pure
        ``objective``: the caller's ``params`` are never mutated. Single shared ``lr`` for now
        (per-group rates deferred); ``least_squares`` and component ``activate`` are deferred --
        see ``design/decisions/stage10-refinement-loop.md``.
        """
        return run_refinement(
            self.objective,
            params,
            steps=steps,
            targets=targets,
            optimizer=optimizer,
            lr=lr,
        )

    def _thickness_for(self, orientation: OrientationPlan, params: RefinableParams) -> Tensor:
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
        ``theta -> thickness`` model) is future work; see ROADMAP / KNOWN_ISSUES.md.
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
        self, orientation: OrientationPlan, fgb: Tensor, thicknesses: Tensor
    ) -> BlochSolution:
        device = fgb.device  # fgb is param-derived; thicknesses/beam_hkl must co-locate with it
        thicknesses = thicknesses.to(device)
        beam_hkl = orientation.beam_hkl.to(device)
        # One sub-solution per rocking-curve tilt (length 1 = the untilted static solve). The tilts
        # share this orientation's beam set; the engine sums |psi|^2 over them (BlochSolution.
        # integrate) -- an incoherent rotation-frame integration. N=1 returns the sub-solution
        # directly, byte-identical to the pre-integration path.
        sub = [
            BlochSolution.from_propagation(
                propagate(build_bloch_system(beam_plan, fgb), thicknesses, method=self.method),
                beam_hkl,
                thicknesses,
            )
            for beam_plan in orientation.beam_plans
        ]
        return sub[0] if len(sub) == 1 else BlochSolution.integrate(sub)

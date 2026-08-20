"""Forward composition spine: raw structure parameters -> diffraction -> scalar objective.

A :class:`RefinementEngine` holds the refinement-invariant plans (constraint spec, ASU-expansion
plan, the shared scattering grid, and one :class:`~diffBloch.engine.plan.OrientationPlan` per
rotation) and maps :class:`~diffBloch.params.RefinableParams` to a differentiable objective:

    constrain -> expand ASU -> structure_factors (Fgb on the shared grid)
              -> per orientation: build_bloch_system -> propagate -> intensities -> align -> loss

``objective_value`` / ``simulate`` are pure and differentiable;
:func:`run_refinement_model` delegates to the quarantined imperative loop in
:mod:`diffBloch.engine.refine`. Engines are assembled from explicit per-orientation beam sets that
preprocessing has already selected.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from diffBloch.core.dynamical import (
    BeamPlanBatch,
    build_bloch_system,
    build_bloch_systems,
    stack_beam_plans,
)
from diffBloch.core.losses import optimal_scale, rbragg, w_rbragg
from diffBloch.core.products import (
    AlignedIntensities,
    BlochSolution,
    align,
    intensities,
    reduce_tilts,
)
from diffBloch.core.scattering import structure_factors
from diffBloch.core.solver import (
    SolverMethod,
    memory_safe_max_batch,
    propagate,
)
from diffBloch.core.symmetry import AsuExpansionPlan, expand_asu
from diffBloch.engine.constraints import ConstraintTransform
from diffBloch.engine.losses import wr2_scores
from diffBloch.engine.plan import (
    CoupledOrientationPlan,
    OrientationPlanLike,
    StructureFactorGrid,
)
from diffBloch.engine.refine import (
    ObjectiveComponent,
    ObjectiveValue,
    OptimizerName,
    PenaltyTerm,
    TrainableSpec,
    _build_optimizer,
    _component_measurements,
    _detach_params,
    _resolve_trainable,
    _scalar_float,
    _to_trainable_params,
)
from diffBloch.observability import (
    NULL_LOGGER,
    Logger,
    ObjectiveManifest,
    ObjectiveTerm,
    RefinementCompleted,
    RefinementOrientationStep,
    RefinementStarted,
    RefinementStep,
)
from diffBloch.params import ConstraintSpec, PhysicalState, RefinableParams, constrain
from diffBloch.specs import Absorption

__all__ = [
    "ForwardContext",
    "LossFn",
    "ModelComponent",
    "RotationMetrics",
    "ScoresFn",
    "StructureComponent",
    "RefinementEngine",
    "RefinementModel",
    "ModelRefinementResult",
    "RefinementProblem",
    "build_refinement_model",
    "build_refinement_problem",
    "run_refinement_model",
]

# A loss reduces one orientation's aligned intensities to a scalar term (calculated vs observed).
# The engine sums these per-orientation terms into the scalar objective refinement minimises.
type LossFn = Callable[[AlignedIntensities], Tensor]

# A scores fn reduces one orientation's aligned intensities to a per-thickness (T,) vector -- the
# same metric a LossFn sums to a scalar, kept unreduced for a search that argmins over thickness
# (score_orientation_per_thickness) or over trial orientations (score_orientation). ExperimentConfig
# .objective supplies matching loss/scores pairs so the gradient refinement and the
# orientation/thickness preprocessing search share one metric.
type ScoresFn = Callable[[AlignedIntensities], Tensor]

_log = logging.getLogger(__name__)


class _Timer:
    """A no-op-unless-``enabled`` wall-clock timer, logged via stdlib diagnostics.

    CUDA kernels queue asynchronously, so an accurate boundary needs ``torch.cuda.synchronize()``
    around the measured block -- itself real overhead, so it (and the ``perf_counter`` call) only
    runs when profiling is explicitly requested. Never on by default; this is diagnostics
    (``logging``), not a domain-observation :class:`~diffBloch.observability.Event`.
    """

    __slots__ = ("label", "device", "enabled", "_start")

    def __init__(self, label: str, device: torch.device, enabled: bool) -> None:
        self.label = label
        self.device = device
        self.enabled = enabled
        self._start = 0.0

    def __enter__(self) -> _Timer:
        if self.enabled:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self.enabled:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            _log.info(
                "profile: %-28s %8.1f ms", self.label, (time.perf_counter() - self._start) * 1e3
            )


@dataclass(frozen=True)
class RotationMetrics:
    """One rotation's scaling-optimised wR2/R_obs for a settled model snapshot.

    Report/plot use (:meth:`RefinementEngine.per_rotation_metrics`), not the objective: each metric
    independently re-optimises its own intensity scale (:func:`diffBloch.core.losses.optimal_scale`),
    exactly as ``refinement_metrics``/the training objective do, so ``wr2``/``r_obs`` here match what
    those report elsewhere. ``rotation_index`` is the original zero-based PETS rotation index.
    """

    rotation_index: int
    wr2: float
    r_obs: float
    n_matched: int


@dataclass(frozen=True)
class RefinementEngine:
    """Forward from raw structure parameters to a differentiable scalar objective.

    Holds the refinement-invariant context: the constraint ``spec``, the ASU-expansion ``asu_plan``,
    the ASU atomic ``numbers``, the shared ``grid``, the per-rotation ``orientations`` (each
    carrying its own frozen ``thickness``), the per-orientation ``loss``, and the propagation
    ``method``.
    """

    spec: ConstraintSpec
    asu_plan: AsuExpansionPlan
    numbers: Tensor
    grid: StructureFactorGrid
    orientations: tuple[OrientationPlanLike, ...]
    loss: LossFn
    # The per-thickness counterpart of loss (see ScoresFn): what score_orientation /
    # score_orientation_per_thickness search over. Defaults to wr2_scores, reproducing this
    # engine's long-standing hardcoded search criterion; build_engine derives both loss and scores
    # from the same ExperimentConfig.objective so the two stay in lockstep.
    scores: ScoresFn = wr2_scores
    method: SolverMethod = "matrix_exp"
    # matrix_exp propagator block cap (memory only; matches unbounded to machine precision, a
    # rounding-level ~1 ulp shift, never accuracy). None (default) lets each
    # solve pick a memory-safe block from its beam count (memory_safe_max_batch), which bounds the
    # (B, T, N, N) propagator that a wide coupled segment x a thickness grid would otherwise
    # materialize all at once. A positive int pins the block
    # for a specific device budget. Execution-only, like method.
    max_batch: int | None = None
    absorption: Absorption = Absorption()
    active_structure_factor_indices: Tensor | None = None
    # Execution-only diagnostics switch (like method): logs per-phase wall time
    # (structure factors, each rotation's solve) via stdlib logging. Off by default -- zero cost.
    profile: bool = False
    # Execution-only memory/compute tradeoff (like max_batch): checkpoint each per-orientation /
    # per-segment solve so its intermediates (including every matrix_exp block) are freed after
    # forward and recomputed on backward, bounding peak memory at the cost of a second forward pass.
    # On by default (the memory-safe choice for the wide coupled-union solves this was added for);
    # disabling it trades that memory headroom for one fewer full recompute per step. Gradients are
    # identical either way -- checkpointing changes only what is retained, never any value.
    checkpoint_activations: bool = True

    def _max_batch_for(self, n_beams: int) -> int:
        """The matrix_exp block cap for a solve over ``n_beams`` beams (explicit pin, else safe)."""
        if self.max_batch is not None:
            return self.max_batch
        return memory_safe_max_batch(n_beams)

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

    def structure_factor_values(self, params: RefinableParams, indices: Tensor) -> Tensor:
        """Calculate ``Fgb`` only for selected rows of the shared support grid."""
        state = self.physical_state(params)
        return self._structure_factor_values_from_state(state, indices)

    def score_orientation(self, orientation: OrientationPlanLike, fgb: Tensor) -> Tensor:
        """This engine's configured ``scores`` for one orientation against its observed pattern.

        Runs the forward Bloch simulation for ``orientation`` from a precomputed ``fgb``
        (:meth:`fgb`), aligns calculated vs observed intensities, and reduces by ``self.scores``
        (default wR2, grid-searching the intensity scale via
        :func:`diffBloch.core.losses.optimal_scale`). With multiple thicknesses the best-fitting
        thickness's score is returned -- thickness is a nuisance when scoring orientation. This is
        the objective ``optimize_orientation`` minimises.
        """
        return self.score_orientation_per_thickness(orientation, fgb).min()

    def score_orientation_per_thickness(
        self, orientation: OrientationPlanLike, fgb: Tensor
    ) -> Tensor:
        """This engine's configured ``scores`` for each of the orientation's thicknesses (``(T,)``).

        One forward Bloch simulation from a precomputed ``fgb`` (:meth:`fgb`) covers all ``T``
        thicknesses at once: the expensive eigendecomposition depends only on the orientation and
        ``fgb``, while thickness enters only the cheap propagation tail. The calculated and observed
        intensities are aligned once (alignment is thickness-independent), then reduced by
        ``self.scores`` (default :func:`~diffBloch.engine.losses.wr2_scores`) -- the same per-metric
        function ``self.loss`` sums for the gradient objective, so this search and the refinement
        stage share one ``ExperimentConfig.objective``.

        ``optimize_thickness`` grid-searches this vector and bakes the lowest-scoring thickness;
        :meth:`score_orientation` collapses it with ``.min()`` (thickness is a nuisance there).

        Runs under ``torch.no_grad()``: this is search scoring (``optimize_orientation`` /
        ``optimize_thickness`` grid search + argmin), never backpropagated -- every caller consumes a
        detached scalar. Without it the ``T``-thickness solve builds an autograd graph whose
        retained ``matrix_exp`` intermediates accumulate across every propagator block, defeating
        the ``max_batch`` memory bound (:func:`~diffBloch.core.solver.propagate`) and OOMing a wide
        coupled segment on the GPU. Grad-off does not change the returned scores.
        """
        with torch.no_grad():
            aligned = align(
                self._solve(orientation, fgb, orientation.thickness),
                orientation.pattern,
                orientation.alignment,
            )
            return self.scores(aligned)

    def objective_value(
        self,
        params: RefinableParams,
        penalties: tuple[PenaltyTerm, ...] = (),
        constraints: tuple[ConstraintTransform, ...] = (),
    ) -> ObjectiveValue:
        """Return the objective as a scalar total plus named scalar components.

        Differentiable in ``params``. The objective is composed in a fixed order that separates hard
        *constraints* (enforced transforms) from soft *penalties* (additive terms)::

            raw RefinableParams
              -> crystallographic constraints (constrain / ConstraintSpec):
                   site-symmetry position projector, ADP equalities, positivity/bounded transforms
              -> PhysicalState
              -> molecular hard constraints  (the `constraints` ConstraintTransform layer)
              -> diffraction term + soft penalties
              -> scalar objective

        ``self.physical_state(params)`` applies the crystallographic constraints; each
        ``ConstraintTransform`` in ``constraints`` then reparameterizes that state in tuple order
        (duplicate names rejected), so both the diffraction term and the ``penalties`` see the
        transformed state. The ``"diffraction"`` component is always present; ``penalties`` add
        weighted soft-penalty components (bond-length, etc.) without making the optimizer know their
        details. Hydrogen riding (:class:`~diffBloch.engine.constraints.HydrogenRiding`) is one such
        ``ConstraintTransform``; a soft penalty and a hard constraint are distinct -- a constraint
        reparameterizes, it is not a cost term.
        """
        return self._objective_value(
            params,
            penalties=penalties,
            constraints=constraints,
            forward_context_for=None,
        )

    def objective_value_model(
        self,
        model: RefinementModel,
        *,
        penalties: tuple[PenaltyTerm, ...] = (),
    ) -> ObjectiveValue:
        """Return the objective for a :class:`RefinementModel`.

        Structure-only models delegate to :meth:`objective_value` for exact behavior parity. Models
        with components use the same objective order, but let components supply forward-context
        values such as per-orientation thickness before falling back to the built-in thickness path.
        """
        return self._objective_value(
            model.structure.initial,
            penalties=penalties,
            constraints=model.structure.constraints,
            forward_context_for=_component_forward_context_for(model) if model.components else None,
        )

    def refinement_metrics(self, model: RefinementModel) -> tuple[float, int, int, int, int]:
        """Return mean R_obs and reflection counts for one refinement-model snapshot.

        Counts are over PETS rows in the selected rotations: matched rows enter the diffraction
        alignment, unmatched rows do not; strong/weak split matched rows at ``I > 3 sigma``.
        """
        with torch.no_grad():
            state = self.physical_state(model.structure.initial)
            for constraint in model.structure.constraints:
                state = constraint.apply(state)
            fgb = self._structure_factors_from_state(state)
            r_values: list[float] = []
            n_matched = 0
            n_strong = 0
            n_weak = 0
            n_unmatched = 0
            for rotation_index, orientation in enumerate(self.orientations):
                context = _forward_context(
                    model, rotation_index=rotation_index, orientation=orientation
                )
                thickness = context.thickness
                if thickness is None:
                    thickness = self._thickness_for(orientation, model.structure.initial)
                aligned = align(
                    self._solve(orientation, fgb, thickness),
                    orientation.pattern,
                    orientation.alignment,
                )
                scores = torch.stack(
                    [
                        optimal_scale(
                            aligned.calculated[t],
                            aligned.observed[t],
                            aligned.sigmas[t],
                            metric=rbragg,
                        )[1]
                        for t in range(aligned.calculated.shape[0])
                    ]
                )
                finite = scores[torch.isfinite(scores)]
                if finite.numel():
                    r_values.append(float(finite.min()))
                strong = aligned.observed[0] > 3.0 * aligned.sigmas[0]
                matched = int(strong.numel())
                n_matched += matched
                n_strong += int(strong.sum())
                n_weak += matched - int(strong.sum())
                n_unmatched += int(orientation.pattern.hkl.shape[0]) - matched
        mean_r_obs = sum(r_values) / len(r_values) if r_values else float("nan")
        return mean_r_obs, n_matched, n_strong, n_weak, n_unmatched

    def per_rotation_metrics(self, model: RefinementModel) -> tuple[RotationMetrics, ...]:
        """Per-rotation wR2/R_obs for one refinement-model snapshot (report/plot use).

        Same loop shape as :meth:`refinement_metrics` (component-aware thickness, one thickness per
        rotation picked by the wR2 that is actually minimised during training -- not by R_obs, which
        is reported but never optimised), returning every rotation's pair instead of an aggregate.
        """
        with torch.no_grad():
            state = self.physical_state(model.structure.initial)
            for constraint in model.structure.constraints:
                state = constraint.apply(state)
            fgb = self._structure_factors_from_state(state)
            rows = []
            for rotation_index, orientation in enumerate(self.orientations):
                context = _forward_context(
                    model, rotation_index=rotation_index, orientation=orientation
                )
                thickness = context.thickness
                if thickness is None:
                    thickness = self._thickness_for(orientation, model.structure.initial)
                aligned = align(
                    self._solve(orientation, fgb, thickness),
                    orientation.pattern,
                    orientation.alignment,
                )
                wr2_scores = torch.stack(
                    [
                        optimal_scale(
                            aligned.calculated[t],
                            aligned.observed[t],
                            aligned.sigmas[t],
                            metric=w_rbragg,
                        )[1]
                        for t in range(aligned.calculated.shape[0])
                    ]
                )
                r_obs_scores = torch.stack(
                    [
                        optimal_scale(
                            aligned.calculated[t],
                            aligned.observed[t],
                            aligned.sigmas[t],
                            metric=rbragg,
                        )[1]
                        for t in range(aligned.calculated.shape[0])
                    ]
                )
                best_t = int(torch.argmin(wr2_scores))
                rows.append(
                    RotationMetrics(
                        rotation_index=orientation.pattern.rotation_index,
                        wr2=float(wr2_scores[best_t]),
                        r_obs=float(r_obs_scores[best_t]),
                        n_matched=int(aligned.observed.shape[-1]),
                    )
                )
        return tuple(rows)

    def _objective_value(
        self,
        params: RefinableParams,
        *,
        penalties: tuple[PenaltyTerm, ...],
        constraints: tuple[ConstraintTransform, ...],
        forward_context_for: Callable[[int, OrientationPlanLike], ForwardContext] | None,
    ) -> ObjectiveValue:
        """Shared objective implementation for structure-only and component model paths."""
        if not self.orientations:
            raise ValueError("engine has no orientations to evaluate")
        names = [constraint.name for constraint in constraints]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate constraint name among {names!r}")
        state = self.physical_state(params)
        for constraint in constraints:
            state = constraint.apply(state)
        fgb = self._structure_factors_from_state(state)
        total = params.asu_positions.new_zeros(())
        wr2_values: list[float] = []
        r_obs_values: list[float] = []
        per_rotation: list[dict[str, float]] = []
        for rotation_index, orientation in enumerate(self.orientations):
            context = (
                ForwardContext()
                if forward_context_for is None
                else forward_context_for(rotation_index, orientation)
            )
            thickness = context.thickness
            if thickness is None:
                thickness = self._thickness_for(orientation, params)
            with _Timer(f"solve[rotation={rotation_index}]", fgb.device, self.profile):
                solution = self._solve(orientation, fgb, thickness)
            aligned = align(solution, orientation.pattern, orientation.alignment)
            with torch.no_grad():
                wr2_scores = torch.stack(
                    [
                        optimal_scale(
                            aligned.calculated[t].detach(),
                            aligned.observed[t],
                            aligned.sigmas[t],
                        )[1]
                        for t in range(aligned.calculated.shape[0])
                    ]
                )
                finite_wr2 = wr2_scores[torch.isfinite(wr2_scores)]
                rotation_wr2 = float(finite_wr2.min()) if finite_wr2.numel() else float("nan")
                if finite_wr2.numel():
                    wr2_values.append(rotation_wr2)
                scores = torch.stack(
                    [
                        optimal_scale(
                            aligned.calculated[t].detach(),
                            aligned.observed[t],
                            aligned.sigmas[t],
                            metric=rbragg,
                        )[1]
                        for t in range(aligned.calculated.shape[0])
                    ]
                )
                finite = scores[torch.isfinite(scores)]
                rotation_r_obs = float(finite.min()) if finite.numel() else float("nan")
                if finite.numel():
                    r_obs_values.append(rotation_r_obs)
            term = self.loss(aligned)
            if term.ndim != 0:
                raise ValueError(f"loss must return a scalar, got shape {tuple(term.shape)}")
            total = total + term
            per_rotation.append(
                {
                    "rotation_index": float(rotation_index),
                    "wr2": rotation_wr2,
                    "r_obs": rotation_r_obs,
                    "diff_loss": float(term.detach()),
                }
            )
        components = {"diffraction": ObjectiveComponent(raw=total)}
        for penalty in penalties:
            if penalty.name in components:
                raise ValueError(f"duplicate objective component name {penalty.name!r}")
            components[penalty.name] = ObjectiveComponent(
                raw=penalty.value(state), weight=penalty.weight
            )
        return ObjectiveValue(
            components,
            diagnostics={
                "wr2": (sum(wr2_values) / len(wr2_values) if wr2_values else float("nan")),
                "r_obs": (sum(r_obs_values) / len(r_obs_values) if r_obs_values else float("nan")),
                # Each mean's own denominator, reported beside it. wR2 and R_obs are NaN-filtered
                # independently, so a rotation can contribute to one and not the other -- one
                # shared count would misstate at least one of the two means.
                "n_rotations": float(len(self.orientations)),
                "n_wr2_evaluated": float(len(wr2_values)),
                "n_r_obs_evaluated": float(len(r_obs_values)),
            },
            per_rotation=per_rotation,
        )

    def _thickness_for(self, orientation: OrientationPlanLike, params: RefinableParams) -> Tensor:
        """The thickness ``(T,)`` the forward model uses for one orientation.

        There are two sources of thickness, and this picks between them:

        - **Normally**, each orientation carries its own fixed thickness
          (``orientation.thickness``), because the specimen presents a different path length at each
          tilt. It is seeded from the
          sample thickness and later replaced by the best-fitting value found by ``optimize_thickness``.
        Trainable thickness refinement is represented by model components that provide
        ``ForwardContext.thickness``. Without such a component, the settled per-orientation Plan
        thickness is the sole source used by the Bloch solve.
        """
        return orientation.thickness

    def physical_state(self, params: RefinableParams) -> PhysicalState:
        """Return bounded physical ASU quantities for ``params``."""
        return constrain(params, self.spec)

    def _structure_factors(self, params: RefinableParams) -> Tensor:
        state = self.physical_state(params)
        return self._structure_factors_from_state(state)

    def _structure_factors_from_state(self, state: PhysicalState) -> Tensor:
        with _Timer("structure_factors", state.positions.device, self.profile):
            active = self.active_structure_factor_indices
            if active is None:
                return self._structure_factor_values_from_state(state, None)
            values = self._structure_factor_values_from_state(state, active)
            full = values.new_zeros(self.grid.structure_factor_hkl.shape[0])
            return full.index_copy(0, active.to(device=values.device), values)

    def _structure_factor_values_from_state(
        self, state: PhysicalState, indices: Tensor | None
    ) -> Tensor:
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
        hkl = self.grid.structure_factor_hkl.to(device)
        if indices is not None:
            indices = indices.to(device=device)
            hkl = hkl.index_select(0, indices)
        return structure_factors(
            expanded.positions,
            expanded.numbers,
            expanded.occupancies,
            expanded.uij,
            hkl=hkl,
            reciprocal_basis=self.grid.reciprocal_basis.to(device),
            cell_volume=self.grid.cell_volume,
            g_max=self.grid.g_max,
            absorption=self.absorption,
            energy=self.orientations[0].energy,
        )

    def _solve(
        self, orientation: OrientationPlanLike, fgb: Tensor, thicknesses: Tensor
    ) -> BlochSolution:
        if isinstance(orientation, CoupledOrientationPlan):
            return self._solve_segmented(orientation, fgb, thicknesses)
        device = fgb.device  # fgb is param-derived; thicknesses/beam_hkl must co-locate with it
        thicknesses = thicknesses.to(device=device, dtype=torch.float32)
        beam_hkl = orientation.beam_hkl.to(device)
        # Untilted (length 1): the static single solve.
        if len(orientation.beam_plans) == 1:
            amplitudes = propagate(
                build_bloch_system(orientation.beam_plans[0], fgb, self.absorption),
                thicknesses,
                method=self.method,
                max_batch=self._max_batch_for(beam_hkl.shape[0]),
            )
            return BlochSolution.from_propagation(amplitudes, beam_hkl, thicknesses)
        # Rocking-curve integration: the tilts share this orientation's beam set, so ONE batched
        # solve over (B, N, N) replaces a Python loop of B single solves (the tilt-batching perf
        # path). The engine then reduces |psi|^2 over the tilt axis (incoherent rotation-frame
        # integration; PlainSum or a mosaicity broadening) via BlochSolution.integrate_batched.
        batch = stack_beam_plans(orientation.beam_plans)

        def solve_batch(fgb_: Tensor, thicknesses_: Tensor) -> Tensor:
            return propagate(
                build_bloch_systems(batch, fgb_, self.absorption),
                thicknesses_,
                method=self.method,
                max_batch=self._max_batch_for(beam_hkl.shape[0]),
            )

        amplitudes = (
            checkpoint(solve_batch, fgb, thicknesses, use_reentrant=False)
            if self.checkpoint_activations
            and torch.is_grad_enabled()
            and (fgb.requires_grad or thicknesses.requires_grad)
            else solve_batch(fgb, thicknesses)
        )  # (B, T, N)
        return BlochSolution.integrate_batched(
            amplitudes, beam_hkl, thicknesses, reduction=orientation.tilt_reduction
        )

    def _solve_segmented(
        self, plan: CoupledOrientationPlan, fgb: Tensor, thicknesses: Tensor
    ) -> BlochSolution:
        """Solve each tilt-chunk on its own beam set, reassemble the union curve, then reduce.

        The tilt-dependent coupling path: each :class:`SegmentPlan` is solved over its
        covered tilts with the batched propagator, and its per-tilt intensities are scattered onto
        the shared ``(N_tilts, T, N_union)`` rocking curve (each tilt belongs to exactly one
        segment; a beam absent from a chunk stays 0 at that chunk's tilts). Only once the whole
        curve is reassembled is the tilt reduction applied -- the mosaicity sample span can cover
        more tilts than any single chunk holds -- and the result is returned as an ordinary
        :class:`BlochSolution` over the union beam set, so ``align`` / scoring are unchanged.
        """
        device = fgb.device
        thicknesses = thicknesses.to(device=device, dtype=torch.float32)
        n_tilts = int(plan.tilts.shape[0])
        n_union = int(plan.beam_hkl.shape[0])
        n_thick = int(thicknesses.shape[0])
        curve = fgb.new_zeros((n_tilts, n_thick, n_union), dtype=thicknesses.dtype)
        for segment in plan.segments:
            batch = stack_beam_plans(segment.plan.beam_plans)
            n_segment_beams = int(segment.union_beam_index.shape[0])

            def solve_segment(
                fgb_: Tensor,
                thicknesses_: Tensor,
                *,
                batch_: BeamPlanBatch = batch,
                n_segment_beams_: int = n_segment_beams,
            ) -> Tensor:
                return propagate(
                    build_bloch_systems(batch_, fgb_, self.absorption),
                    thicknesses_,
                    method=self.method,
                    max_batch=self._max_batch_for(n_segment_beams_),
                )

            amplitudes = (
                checkpoint(solve_segment, fgb, thicknesses, use_reentrant=False)
                if self.checkpoint_activations
                and torch.is_grad_enabled()
                and (fgb.requires_grad or thicknesses.requires_grad)
                else solve_segment(fgb, thicknesses)
            )  # (C, T, n_seg)
            cover = segment.cover.to(device)
            union_beam_index = segment.union_beam_index.to(device)
            block = curve[cover]  # (C, T, n_union) gathered copy
            block[:, :, union_beam_index] = intensities(amplitudes)
            curve[cover] = block
        total = reduce_tilts(curve, plan.tilt_reduction)  # (T, n_union)
        # The reassembled curve is an intensity sum, so its per-tilt amplitudes were never coherent;
        # store the magnitude sqrt(total) in the solve's complex64 format (matching the
        # static/batched paths).
        return BlochSolution(
            total.sqrt().to(torch.complex64), total, plan.beam_hkl.to(device), thicknesses
        )


@dataclass(frozen=True)
class ForwardContext:
    """Forward-model values supplied by refinement model components.

    Deliberately narrow: apparent thickness is the only value admitted, since it is the only value a
    component currently supplies. Scale/background/damage fields should be added only when consumed.
    """

    thickness: Tensor | None = None


class ModelComponent(Protocol):
    """A trainable/differentiable model component that can feed the forward simulation."""

    @property
    def key(self) -> str:
        """Stable component parameter-tree key for validation and optimizer grouping."""
        ...

    def initial_params(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Mapping[str, Tensor]:
        """Return initial parameter tensors for this component."""
        ...

    def forward_context(
        self,
        params: Mapping[str, Tensor],
        *,
        rotation_index: int,
        orientation: OrientationPlanLike,
    ) -> ForwardContext:
        """Return this component's values for one orientation."""
        ...


@dataclass(frozen=True)
class StructureComponent:
    """The physical-structure component of a refinement model.

    A thin wrapper around the structure refinement inputs: ``initial`` is the
    :class:`~diffBloch.params.RefinableParams`, and ``constraints`` is the tuple of hard molecular
    transforms applied after the crystallographic ``constrain``. Structure-local hard
    parameterizations belong here.
    """

    initial: RefinableParams
    constraints: tuple[ConstraintTransform, ...] = ()


@dataclass(frozen=True)
class RefinementModel:
    """Trainable refinement model value, currently structure-only.

    The model is the value optimized against a static :class:`RefinementEngine`. Non-structure
    components provide forward-context values such as apparent thickness.
    """

    structure: StructureComponent
    components: tuple[ModelComponent, ...] = ()
    component_params: Mapping[str, Mapping[str, Tensor]] = MappingProxyType({})

    def __post_init__(self) -> None:
        keys = [component.key for component in self.components]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate refinement component key among {keys!r}")
        params = {key: MappingProxyType(dict(self.component_params.get(key, {}))) for key in keys}
        unexpected = set(self.component_params) - set(keys)
        if unexpected:
            raise ValueError(
                f"component_params has no matching component for {sorted(unexpected)!r}"
            )
        object.__setattr__(self, "component_params", MappingProxyType(params))


def _component_forward_context_for(
    model: RefinementModel,
) -> Callable[[int, OrientationPlanLike], ForwardContext]:
    def forward_context_for(
        rotation_index: int, orientation: OrientationPlanLike
    ) -> ForwardContext:
        return _forward_context(model, rotation_index=rotation_index, orientation=orientation)

    return forward_context_for


def _forward_context(
    model: RefinementModel,
    *,
    rotation_index: int,
    orientation: OrientationPlanLike,
) -> ForwardContext:
    thickness: Tensor | None = None
    for component in model.components:
        params = model.component_params[component.key]
        context = component.forward_context(
            params, rotation_index=rotation_index, orientation=orientation
        )
        if context.thickness is not None:
            if thickness is not None:
                raise ValueError("multiple refinement components provided thickness")
            thickness = context.thickness
    return ForwardContext(thickness=thickness)


def build_refinement_model(
    *,
    initial: RefinableParams,
    constraints: tuple[ConstraintTransform, ...] = (),
    components: tuple[ModelComponent, ...] = (),
    component_params: Mapping[str, Mapping[str, Tensor]] = MappingProxyType({}),
) -> RefinementModel:
    """Construct a :class:`RefinementModel`.

    With no components supplied it is behavior-equivalent to structure-only refinement; supplied
    components contribute forward-context values (e.g. apparent thickness) to that structure core.
    """
    return RefinementModel(
        structure=StructureComponent(initial=initial, constraints=constraints),
        components=components,
        component_params=component_params,
    )


@dataclass(frozen=True)
class ModelRefinementResult:
    """The outcome of optimizing a refinement model.

    Returns the optimized model value. ``params``/``best_params`` properties
    preserve the structure-only convenience used by the default app path and existing tests.
    """

    model: RefinementModel
    losses: Tensor
    best_model: RefinementModel
    best_step: int
    selection_losses: Tensor | None = None
    history: tuple[RefinementStep, ...] = ()
    reflection_counts: Mapping[str, int] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)
    # The same value emitted as the run's opening event, carried here so a report can state the
    # composed objective without the caller re-threading the problem/model it was built from.
    objective_manifest: ObjectiveManifest | None = None

    @property
    def params(self) -> RefinableParams:
        """Final refined structure parameters."""
        return self.model.structure.initial

    @property
    def best_params(self) -> RefinableParams:
        """Best recorded structure parameters."""
        return self.best_model.structure.initial

    @property
    def best_loss(self) -> float:
        """The loss used to select ``best_step``.

        By default this is the training objective. When ``run_refinement_model`` is given a
        selection engine (the app's held-out validation split), it is the corresponding selection
        objective instead.
        """
        losses = self.selection_losses if self.selection_losses is not None else self.losses
        return float(losses[self.best_step])


@dataclass(frozen=True)
class RefinementProblem:
    """Objective-side scientific composition for one refinement run.

    The trainable model value is passed separately to ``run_refinement_model``. The problem records
    only additive objective terms such as bond, angle, and planarity penalties.
    """

    penalties: tuple[PenaltyTerm, ...] = ()


def build_refinement_problem(
    *,
    penalties: tuple[PenaltyTerm, ...] = (),
) -> RefinementProblem:
    """Construct objective-side refinement composition data."""
    return RefinementProblem(penalties=penalties)


@dataclass(frozen=True)
class _TrainableComponentParams:
    params_by_component: Mapping[str, Mapping[str, Tensor]]
    leaves: tuple[Tensor, ...]

    def params(self) -> Mapping[str, Mapping[str, Tensor]]:
        return self.params_by_component


def _to_trainable_component_params(
    component_params: Mapping[str, Mapping[str, Tensor]],
) -> _TrainableComponentParams:
    params_by_component: dict[str, Mapping[str, Tensor]] = {}
    leaves: list[Tensor] = []
    for component_key, params in component_params.items():
        cloned: dict[str, Tensor] = {}
        for tensor_name, tensor in params.items():
            leaf = tensor.detach().clone().requires_grad_(True)
            cloned[tensor_name] = leaf
            leaves.append(leaf)
        params_by_component[component_key] = MappingProxyType(cloned)
    return _TrainableComponentParams(
        params_by_component=MappingProxyType(params_by_component), leaves=tuple(leaves)
    )


def _detach_component_params(
    component_params: Mapping[str, Mapping[str, Tensor]],
) -> Mapping[str, Mapping[str, Tensor]]:
    params_by_component: dict[str, Mapping[str, Tensor]] = {}
    for component_key, params in component_params.items():
        params_by_component[component_key] = MappingProxyType(
            {name: tensor.detach().clone() for name, tensor in params.items()}
        )
    return MappingProxyType(params_by_component)


def _detach_model(model: RefinementModel) -> RefinementModel:
    return dataclasses.replace(
        model,
        structure=dataclasses.replace(
            model.structure, initial=_detach_params(model.structure.initial)
        ),
        component_params=_detach_component_params(model.component_params),
    )


def run_refinement_model(
    engine: RefinementEngine,
    model: RefinementModel,
    problem: RefinementProblem,
    *,
    trainable: TrainableSpec,
    steps: int,
    optimizer: OptimizerName = "adam",
    lr: float = 1e-3,
    logger: Logger = NULL_LOGGER,
    verbose: bool = False,
    profile: bool = False,
    selection_engine: RefinementEngine | None = None,
) -> ModelRefinementResult:
    """Optimize a refinement model against the supplied engine/static context and problem terms.

    Before the first step this emits an :class:`~diffBloch.observability.ObjectiveManifest` naming
    the penalties (with weights), constraints, and components the run actually composed, and returns
    the same value on the result. This is the one place that holds the problem and the model
    together, so it is the only place that can state the objective's composition; the structure-only
    :func:`~diffBloch.engine.refine.run_refinement` receives a bare objective callable and therefore
    declares nothing.

    ``verbose`` ("verbose refinement") additionally reports one
    :class:`~diffBloch.observability.RefinementOrientationStep` per rotation per step (wr2/r_obs/
    diff_loss), for diagnosing which orientations drive the epoch mean reported by the ordinary
    :class:`~diffBloch.observability.RefinementStep`. Off by default: the per-rotation stream is
    ``n_orientations``x louder and is a diagnosis tool, not the default reporting shape. It is
    execution-only, like ``logger`` itself -- it changes what gets reported, never the objective or
    the optimizer trajectory, so it is a function argument / CLI flag, not an ``experiment.yaml``
    field (a config field would enter the preprocess/refinement identity for no scientific reason).

    ``profile`` logs per-phase wall time (structure factors, each rotation's solve, backward,
    optimizer step) via stdlib diagnostics logging (``logging.getLogger(__name__)``, level INFO,
    the ``"profile: "``-prefixed lines) -- see ``diffBloch.engine.forward._Timer``. Execution-only
    and off by default: it forces a CUDA sync around every measured block, which is itself real
    overhead, so it is a diagnosis tool for one run, not something to leave on. ``engine.profile``
    must also be set (:func:`~diffBloch.preprocess.scoring.build_engine` threads it through) for the
    structure-factor/solve breakdown; this flag alone only times backward/optimizer-step.

    ``selection_engine`` is an optional held-out objective used only to choose ``best_model`` /
    ``best_step``. It does not contribute gradients or alter the optimizer trajectory. The app uses
    it for ``refinement.split.train_test`` validation selection; without it, best selection remains
    the training objective. It costs one extra no-grad forward pass per step over the held-out
    rotations -- roughly ``val_frac`` of a training forward, paid every epoch with no opt-out --
    and it changes which objective :class:`~diffBloch.observability.RefinementCompleted` reports
    (see that event's ``selection`` field).
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    trainable_params = _to_trainable_params(
        model.structure.initial,
        _resolve_trainable(model.structure.initial, trainable, atomic_numbers=engine.numbers),
    )
    component_params = _to_trainable_component_params(model.component_params)
    opt = _build_optimizer(
        optimizer,
        [*trainable_params.leaves, *component_params.leaves],
        lr,
    )
    reported_objective: ObjectiveValue | None = None
    profile_device = model.structure.initial.asu_positions.device

    def current_model() -> RefinementModel:
        return dataclasses.replace(
            model,
            structure=dataclasses.replace(model.structure, initial=trainable_params.params()),
            component_params=component_params.params(),
        )

    def closure() -> float:
        nonlocal reported_objective
        opt.zero_grad()
        with _Timer("forward (total)", profile_device, profile):
            current_objective = engine.objective_value_model(
                current_model(), penalties=problem.penalties
            )
        if reported_objective is None:
            reported_objective = current_objective
        loss = current_objective.total
        with _Timer("backward", profile_device, profile):
            loss.backward()  # type: ignore[no-untyped-call]
        for leaf in [*trainable_params.leaves, *component_params.leaves]:
            if leaf.grad is not None and not leaf.grad.is_contiguous():
                leaf.grad = leaf.grad.contiguous()
        return float(loss.detach())

    losses: list[float] = []
    selection_losses: list[float] = []
    history: list[RefinementStep] = []
    best_loss = float("inf")
    best_step = 0
    best_model = _detach_model(current_model())
    # Declared before the first step, so the composed objective is legible from the run's opening
    # lines rather than inferred later from which per-term measurements happened to appear.
    manifest = ObjectiveManifest(
        penalties=tuple(
            ObjectiveTerm(name=penalty.name, weight=penalty.weight) for penalty in problem.penalties
        ),
        constraints=tuple(constraint.name for constraint in model.structure.constraints),
        components=tuple(component.key for component in model.components),
    )
    logger.report(manifest)
    logger.report(RefinementStarted(total_steps=steps))
    for step in range(steps):
        snapshot = _detach_model(current_model())
        reported_objective = None
        with _Timer(f"epoch {step} (total)", profile_device, profile):
            loss_value = opt.step(closure)
        assert loss_value is not None
        loss_value = float(loss_value)
        if reported_objective is None:
            raise RuntimeError("optimizer did not evaluate the refinement objective")
        losses.append(loss_value)
        diffraction_loss = _scalar_float(reported_objective.components["diffraction"].raw)
        # wR2/R_obs diagnostics are always computed (see _objective_value) regardless of
        # ExperimentConfig.loss_metrics, so refinement always reports both -- free, unlike the
        # preprocessing search, which reports only the metric it actually spent a solve computing.
        diagnostics = reported_objective.diagnostics
        selection_loss = loss_value
        selection_diagnostics: Mapping[str, float] | None = None
        if selection_engine is not None:
            with torch.no_grad():
                selection_objective = selection_engine.objective_value_model(
                    snapshot, penalties=problem.penalties
                )
            selection_loss = _scalar_float(selection_objective.total)
            selection_losses.append(selection_loss)
            selection_diagnostics = selection_objective.diagnostics
        event = RefinementStep(
            iteration=step,
            loss=loss_value,
            wr2=diagnostics["wr2"],
            r_obs=diagnostics["r_obs"],
            diff_loss=diffraction_loss,
            objective_total=_scalar_float(reported_objective.total),
            components=_component_measurements(reported_objective),
            n_rotations=int(diagnostics["n_rotations"]),
            n_wr2_evaluated=int(diagnostics["n_wr2_evaluated"]),
            n_r_obs_evaluated=int(diagnostics["n_r_obs_evaluated"]),
            val_wr2=(None if selection_diagnostics is None else selection_diagnostics["wr2"]),
            val_r_obs=(None if selection_diagnostics is None else selection_diagnostics["r_obs"]),
            val_n_rotations=(
                None if selection_diagnostics is None else int(selection_diagnostics["n_rotations"])
            ),
            val_n_wr2_evaluated=(
                None
                if selection_diagnostics is None
                else int(selection_diagnostics["n_wr2_evaluated"])
            ),
            val_n_r_obs_evaluated=(
                None
                if selection_diagnostics is None
                else int(selection_diagnostics["n_r_obs_evaluated"])
            ),
        )
        history.append(event)
        logger.report(event)
        if verbose:
            for entry in reported_objective.per_rotation:
                logger.report(
                    RefinementOrientationStep(
                        iteration=step,
                        rotation_index=int(entry["rotation_index"]),
                        wr2=entry["wr2"],
                        r_obs=entry["r_obs"],
                        diff_loss=entry["diff_loss"],
                    )
                )
        if selection_loss < best_loss:
            best_loss, best_step, best_model = selection_loss, step, snapshot
    _, n_matched, n_strong, n_weak, n_unmatched = engine.refinement_metrics(best_model)
    reflection_counts = MappingProxyType(
        {
            "matched": n_matched,
            "matched_i_gt_3sigma": n_strong,
            "matched_i_le_3sigma": n_weak,
            "unmatched_observed": n_unmatched,
        }
    )
    logger.report(
        RefinementCompleted(
            n_steps=steps,
            best_step=best_step,
            best_loss=best_loss,
            selection="validation" if selection_engine is not None else "training",
            reflection_counts=reflection_counts,
        )
    )
    return ModelRefinementResult(
        model=_detach_model(current_model()),
        losses=torch.tensor(losses, dtype=torch.float64),
        best_model=best_model,
        best_step=best_step,
        selection_losses=(
            torch.tensor(selection_losses, dtype=torch.float64)
            if selection_engine is not None
            else None
        ),
        history=tuple(history),
        reflection_counts=reflection_counts,
        objective_manifest=manifest,
    )

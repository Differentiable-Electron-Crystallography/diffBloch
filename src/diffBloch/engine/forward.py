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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

import torch
from torch import Tensor

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
from diffBloch.core.solver import (
    FloatFormat,
    SolverMethod,
    memory_safe_max_batch,
    precision_dtypes,
    propagate,
)
from diffBloch.core.symmetry import AsuExpansionPlan, expand_asu
from diffBloch.engine.constraints import ConstraintTransform
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
from diffBloch.observability import NULL_LOGGER, Logger, RefinementCompleted, RefinementStep
from diffBloch.params import ConstraintSpec, PhysicalState, RefinableParams, constrain

__all__ = [
    "ForwardContext",
    "LossFn",
    "ModelComponent",
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
    method: SolverMethod = "matrix_exp"
    # Solve numeric field. "fp64" (complex128) everywhere by default. "fp32"
    # (complex64) is a speed/precision knob: preprocess uses it only for transient coarse-search
    # engines, and the default refine path stays fp64 unless config opts in. See
    # core.solver.propagate.
    precision: FloatFormat = "fp64"
    # matrix_exp propagator block cap (memory only; matches unbounded to machine precision, a
    # rounding-level ~1 ulp shift, never accuracy). None (default) lets each
    # solve pick a memory-safe block from its beam count (memory_safe_max_batch), which bounds the
    # (B, T, N, N) propagator that a wide coupled segment x a thickness grid would otherwise
    # materialize all at once. A positive int pins the block
    # for a specific device budget. Execution-only, like precision/method.
    max_batch: int | None = None

    def _max_batch_for(self, n_beams: int) -> int:
        """The matrix_exp block cap for a solve over ``n_beams`` beams (explicit pin, else safe)."""
        if self.max_batch is not None:
            return self.max_batch
        return memory_safe_max_batch(n_beams, self.precision)

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
        scoring orientation. This is the objective ``fit_orientation`` minimises.
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

        Runs under ``torch.no_grad()``: this is search scoring (``fit_orientation`` /
        ``fit_thickness`` grid search + argmin), never backpropagated -- every caller consumes a
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
            return torch.stack(
                [
                    optimal_scale(aligned.calculated[t], aligned.observed[t], aligned.sigmas[t])[1]
                    for t in range(aligned.calculated.shape[0])
                ]
            )

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
        for rotation_index, orientation in enumerate(self.orientations):
            context = (
                ForwardContext()
                if forward_context_for is None
                else forward_context_for(rotation_index, orientation)
            )
            thickness = context.thickness
            if thickness is None:
                thickness = self._thickness_for(orientation, params)
            solution = self._solve(orientation, fgb, thickness)
            aligned = align(solution, orientation.pattern, orientation.alignment)
            term = self.loss(aligned)
            if term.ndim != 0:
                raise ValueError(f"loss must return a scalar, got shape {tuple(term.shape)}")
            total = total + term
        components = {"diffraction": ObjectiveComponent(raw=total)}
        for penalty in penalties:
            if penalty.name in components:
                raise ValueError(f"duplicate objective component name {penalty.name!r}")
            components[penalty.name] = ObjectiveComponent(
                raw=penalty.value(state), weight=penalty.weight
            )
        return ObjectiveValue(components)

    def _thickness_for(self, orientation: OrientationPlanLike, params: RefinableParams) -> Tensor:
        """The thickness ``(T,)`` the forward model uses for one orientation.

        There are two sources of thickness, and this picks between them:

        - **Normally**, each orientation carries its own fixed thickness
          (``orientation.thickness``), because the specimen presents a different path length at each
          tilt. It is seeded from the
          sample thickness and later replaced by the best-fitting value found by ``fit_thickness``.
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
            hkl=self.grid.structure_factor_hkl.to(device),
            reciprocal_basis=self.grid.reciprocal_basis.to(device),
            cell_volume=self.grid.cell_volume,
            g_max=self.grid.g_max,
        )

    def _solve(
        self, orientation: OrientationPlanLike, fgb: Tensor, thicknesses: Tensor
    ) -> BlochSolution:
        if isinstance(orientation, CoupledOrientationPlan):
            return self._solve_segmented(orientation, fgb, thicknesses)
        device = fgb.device  # fgb is param-derived; thicknesses/beam_hkl must co-locate with it
        real_dtype, _ = precision_dtypes(self.precision)
        thicknesses = thicknesses.to(device=device, dtype=real_dtype)
        beam_hkl = orientation.beam_hkl.to(device)
        # Untilted (length 1): the static single solve.
        if len(orientation.beam_plans) == 1:
            amplitudes = propagate(
                build_bloch_system(orientation.beam_plans[0], fgb),
                thicknesses,
                method=self.method,
                precision=self.precision,
                max_batch=self._max_batch_for(beam_hkl.shape[0]),
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
            max_batch=self._max_batch_for(beam_hkl.shape[0]),
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
                max_batch=self._max_batch_for(segment.union_beam_index.shape[0]),
            )  # (C, T, n_seg)
            cover = segment.cover.to(device)
            union_beam_index = segment.union_beam_index.to(device)
            block = curve[cover]  # (C, T, n_union) gathered copy
            block[:, :, union_beam_index] = intensities(amplitudes)
            curve[cover] = block
        total = reduce_tilts(curve, plan.tilt_reduction)  # (T, n_union)
        # The reassembled curve is an intensity sum, so its per-tilt amplitudes were never coherent;
        # store the magnitude sqrt(total) in the solve's complex format (complex128 under fp64,
        # complex64 under fp32, matching the static/batched paths).
        _, complex_dtype = precision_dtypes(self.precision)
        return BlochSolution(
            total.sqrt().to(complex_dtype), total, plan.beam_hkl.to(device), thicknesses
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
        """The lowest recorded (pre-update) loss."""
        return float(self.losses[self.best_step])


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
    optimizer: OptimizerName = "lbfgs",
    lr: float = 1e-3,
    logger: Logger = NULL_LOGGER,
) -> ModelRefinementResult:
    """Optimize a refinement model against the supplied engine/static context and problem terms."""
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

    def current_model() -> RefinementModel:
        return dataclasses.replace(
            model,
            structure=dataclasses.replace(model.structure, initial=trainable_params.params()),
            component_params=component_params.params(),
        )

    def closure() -> float:
        nonlocal reported_objective
        opt.zero_grad()
        current_objective = engine.objective_value_model(
            current_model(), penalties=problem.penalties
        )
        if reported_objective is None:
            reported_objective = current_objective
        loss = current_objective.total
        loss.backward()  # type: ignore[no-untyped-call]
        for leaf in [*trainable_params.leaves, *component_params.leaves]:
            if leaf.grad is not None and not leaf.grad.is_contiguous():
                leaf.grad = leaf.grad.contiguous()
        return float(loss.detach())

    losses: list[float] = []
    best_loss = float("inf")
    best_step = 0
    best_model = _detach_model(current_model())
    for step in range(steps):
        snapshot = _detach_model(current_model())
        reported_objective = None
        loss_value = opt.step(closure)
        assert loss_value is not None
        loss_value = float(loss_value)
        if reported_objective is None:
            raise RuntimeError("optimizer did not evaluate the refinement objective")
        losses.append(loss_value)
        logger.report(
            RefinementStep(
                iteration=step,
                loss=loss_value,
                objective_total=_scalar_float(reported_objective.total),
                components=_component_measurements(reported_objective),
            )
        )
        if loss_value < best_loss:
            best_loss, best_step, best_model = loss_value, step, snapshot
    logger.report(RefinementCompleted(n_steps=steps, best_step=best_step, best_loss=best_loss))
    return ModelRefinementResult(
        model=_detach_model(current_model()),
        losses=torch.tensor(losses, dtype=torch.float64),
        best_model=best_model,
        best_step=best_step,
    )

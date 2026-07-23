"""The imperative refinement loop -- the one deliberately stateful corner of an otherwise pure core.

``torch.optim`` optimizers mutate ``.grad`` and leaf tensors in place and carry internal state, so a
training loop cannot be a pure function. This module quarantines that imperativeness behind a
functional contract: :func:`run_refinement` takes the engine's pure ``objective_value`` callable
and the caller's parameters, clones the *target* fields into fresh ``requires_grad`` leaves (the
rest become detached constants), steps a chosen backend, and returns a new detached
:class:`RefinementResult`.
The caller's parameters are never touched. ``core/`` stays free of ``torch.optim`` entirely.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Protocol

import gemmi
import torch
from torch import Tensor

from diffBloch.observability import (
    NULL_LOGGER,
    Logger,
    RefinementCompleted,
    RefinementStep,
)
from diffBloch.params import PhysicalState, RefinableParams

__all__ = [
    "AtomSelection",
    "ObjectiveComponent",
    "ObjectiveValue",
    "OptimizerName",
    "RefinementResult",
    "PenaltyTerm",
    "TrainableSpec",
    "run_refinement",
]

# The torch.optim backends wired here; least_squares (Gauss-Newton/LM) is deferred.
type OptimizerName = Literal["adam", "adamw", "lbfgs"]

# Trainable group name -> the RefinableParams field(s) it unlocks for optimization. "adp" maps to
# both raw ADP fields; only those actually present become leaves.
_TRAINABLE_FIELDS: dict[str, tuple[str, ...]] = {
    "positions": ("asu_positions",),
    "adp": ("uij_raw", "u_iso_raw"),
    "occupancy": ("occupancy_raw",),
    "fgb": ("Fgb",),
}


@dataclass(frozen=True)
class AtomSelection:
    """A coarse atom/parameter selection for one trainable group.

    The current modes distinguish whole-group ``all`` vs ``none``. The value can be extended with
    element or index filters without reintroducing stringly-typed refinement targets.
    """

    mode: Literal["all", "none"]
    element_include: tuple[str, ...] = ()
    element_exclude: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject invalid runtime construction, not just invalid static/config values."""
        if self.mode not in {"all", "none"}:
            raise ValueError(f"atom selection mode must be 'all' or 'none'; got {self.mode!r}")
        _element_numbers(self.element_include)
        _element_numbers(self.element_exclude)
        overlap = set(self.element_include) & set(self.element_exclude)
        if overlap:
            raise ValueError(f"elements cannot be both included and excluded: {sorted(overlap)}")
        if self.mode == "none" and (self.element_include or self.element_exclude):
            raise ValueError("element filters require selection mode 'all'")

    @classmethod
    def all(cls) -> AtomSelection:
        """Select every present parameter in the group."""
        return cls("all")

    @classmethod
    def include_elements(cls, *symbols: str) -> AtomSelection:
        """Select only atoms whose element symbol is listed."""
        return cls("all", element_include=tuple(symbols))

    @classmethod
    def exclude_elements(cls, *symbols: str) -> AtomSelection:
        """Select all atoms except those whose element symbol is listed."""
        return cls("all", element_exclude=tuple(symbols))

    @classmethod
    def none(cls) -> AtomSelection:
        """Select no parameters in the group."""
        return cls("none")

    @property
    def selects_any(self) -> bool:
        """Whether this selection unlocks the corresponding trainable group."""
        return self.mode == "all"

    @property
    def has_element_filter(self) -> bool:
        """Whether this selection needs ASU atomic numbers to resolve per-row leaves."""
        return bool(self.element_include or self.element_exclude)


def _element_numbers(symbols: tuple[str, ...]) -> frozenset[int]:
    """Resolve element symbols to atomic numbers at the API boundary."""
    numbers: set[int] = set()
    for symbol in symbols:
        element = gemmi.Element(symbol)
        number = int(element.atomic_number)
        if number <= 0 or element.name.lower() != symbol.lower():
            raise ValueError(f"unknown element symbol {symbol!r}")
        numbers.add(number)
    return frozenset(numbers)


@dataclass(frozen=True)
class TrainableSpec:
    """Explicit selection of which parameter groups are trainable in a refinement problem."""

    positions: AtomSelection = field(default_factory=AtomSelection.none)
    adp: AtomSelection = field(default_factory=AtomSelection.none)
    occupancy: AtomSelection = field(default_factory=AtomSelection.none)
    fgb: AtomSelection = field(default_factory=AtomSelection.none)

    @classmethod
    def positions_and_adp(cls) -> TrainableSpec:
        """The default: refine positions and ADPs when present."""
        return cls(positions=AtomSelection.all(), adp=AtomSelection.all())


@dataclass(frozen=True)
class ObjectiveComponent:
    """One named refinement objective term.

    ``raw`` is the scientifically meaningful scalar diagnostic (for example, a bond penalty before
    weighting). ``weight`` scales that diagnostic into the optimizer-facing ``contribution``.
    """

    raw: Tensor
    weight: float = 1.0

    @property
    def contribution(self) -> Tensor:
        """The weighted scalar contribution this component adds to the objective total."""
        return self.raw * self.weight


@dataclass(frozen=True, init=False)
class ObjectiveValue:
    """A scalar refinement objective plus named scalar components.

    ``total`` is computed from component contributions so reporting and optimization cannot silently
    drift. ``components`` is a read-only mapping whose values retain both raw diagnostics and
    weights for future penalty reporting.
    """

    total: Tensor
    components: Mapping[str, ObjectiveComponent]

    def __init__(self, components: Mapping[str, ObjectiveComponent]) -> None:
        if not components:
            raise ValueError("at least one objective component is required")
        copied = dict(components)
        contributions: list[Tensor] = []
        for name, component in copied.items():
            if component.raw.ndim != 0:
                raise ValueError(
                    f"objective component {name!r} must be scalar, got shape "
                    f"{tuple(component.raw.shape)}"
                )
            contributions.append(component.contribution)
        total = contributions[0].new_zeros(())
        for contribution in contributions:
            total = total + contribution
        object.__setattr__(self, "total", total)
        object.__setattr__(self, "components", MappingProxyType(copied))


class PenaltyTerm(Protocol):
    """A soft refinement penalty evaluated on the physical ASU state.

    Penalties are objective components, not hard constraints: ``loss`` returns the raw scientific
    diagnostic and ``weight`` scales it into the optimizer-facing contribution. Concrete terms own
    their invariant context (for example metric/cell, connectivity, targets, and sigmas) instead of
    bloating :class:`~diffBloch.params.PhysicalState` with every possible penalty input.
    """

    name: str
    weight: float

    def value(self, state: PhysicalState) -> Tensor:
        """Return this penalty's raw scalar loss for the current physical state."""
        ...


@dataclass(frozen=True)
class RefinementResult:
    """The outcome of a refinement run (all tensors detached).

    ``params`` are the final parameters after the last step; ``losses`` ``(steps,)`` is the
    per-step training curve (each entry is the objective *before* that step's update);
    ``best_params`` / ``best_step`` snapshot the lowest recorded loss, for early-stopping callers.
    """

    params: RefinableParams
    losses: Tensor
    best_params: RefinableParams
    best_step: int

    @property
    def best_loss(self) -> float:
        """The lowest recorded (pre-update) loss."""
        return float(self.losses[self.best_step])


def run_refinement(
    objective_value: Callable[[RefinableParams], ObjectiveValue],
    params: RefinableParams,
    *,
    steps: int,
    trainable: TrainableSpec,
    optimizer: OptimizerName,
    lr: float,
    atomic_numbers: Tensor | None = None,
    logger: Logger = NULL_LOGGER,
) -> RefinementResult:
    """Optimize the ``trainable`` parameter groups to minimise ``objective_value(params).total``.

    Functional contract over an unavoidably imperative core: the caller's ``params`` are never
    mutated. Selected fields become fresh ``requires_grad`` leaves (non-selected fields detached
    constants); a backend steps them for ``steps`` iterations via a closure (which unifies LBFGS'
    re-evaluation with Adam/AdamW). Trainable selections map through ``_TRAINABLE_FIELDS``; a
    selected group with no present parameter, or zero ``steps``, raises.

    ``logger`` receives a :class:`RefinementStep` per iteration and one
    :class:`RefinementCompleted` at the end; the default :data:`NULL_LOGGER` makes emission a no-op,
    so the returned result is unchanged. Step events include the structured ``ObjectiveValue``
    components as numeric diagnostics, making diffraction/penalty tradeoffs inspectable.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    trainable_params = _to_trainable_params(
        params, _resolve_trainable(params, trainable, atomic_numbers=atomic_numbers)
    )
    opt = _build_optimizer(optimizer, list(trainable_params.leaves), lr)
    reported_objective: ObjectiveValue | None = None

    def closure() -> float:
        nonlocal reported_objective
        opt.zero_grad()
        current_objective = objective_value(trainable_params.params())
        if reported_objective is None:
            # LBFGS may evaluate the closure multiple times during one outer step. ``step`` returns
            # the first/pre-update loss, so diagnostics must snapshot that same ObjectiveValue, not
            # a later line-search probe.
            reported_objective = current_objective
        loss = current_objective.total
        loss.backward()  # type: ignore[no-untyped-call]
        # Some objective ops (symmetry projection, ADP reparameterization) leave a leaf's grad
        # non-contiguous; LBFGS flattens grads with ``.view(-1)`` and would raise on that, so make
        # them contiguous here (a no-op for the grads that already are, and for optimizers that
        # don't flatten).
        for leaf in trainable_params.leaves:
            if leaf.grad is not None and not leaf.grad.is_contiguous():
                leaf.grad = leaf.grad.contiguous()
        return float(loss.detach())

    losses: list[float] = []
    best_loss = math.inf
    best_step = 0
    best_params = _detach_params(trainable_params.params())
    for step in range(steps):
        snapshot = _detach_params(trainable_params.params())  # params behind this step's loss
        reported_objective = None
        loss_value = opt.step(closure)
        assert loss_value is not None  # closure is always provided -> step returns the loss
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
            best_loss, best_step, best_params = loss_value, step, snapshot
    logger.report(RefinementCompleted(n_steps=steps, best_step=best_step, best_loss=best_loss))
    return RefinementResult(
        params=_detach_params(trainable_params.params()),
        losses=torch.tensor(losses, dtype=torch.float64),
        best_params=best_params,
        best_step=best_step,
    )


def _component_measurements(objective: ObjectiveValue) -> Mapping[str, Mapping[str, float]]:
    return MappingProxyType(
        {
            name: MappingProxyType(
                {
                    "raw": _scalar_float(component.raw),
                    "weight": float(component.weight),
                    "contribution": _scalar_float(component.contribution),
                }
            )
            for name, component in objective.components.items()
        }
    )


def _scalar_float(value: Tensor) -> float:
    return float(value.detach())


def _resolve_trainable(
    params: RefinableParams,
    trainable: TrainableSpec,
    *,
    atomic_numbers: Tensor | None,
) -> Mapping[str, Tensor | None]:
    """Map trainable selections to RefinableParams field names plus optional row masks."""
    fields: dict[str, Tensor | None] = {}
    selected_groups = [name for name in _TRAINABLE_FIELDS if getattr(trainable, name).selects_any]
    if not selected_groups:
        raise ValueError("at least one trainable parameter group is required")
    for group in selected_groups:
        selection = getattr(trainable, group)
        present = [name for name in _TRAINABLE_FIELDS[group] if getattr(params, name) is not None]
        if not present:
            raise ValueError(
                f"trainable group {group!r} selected but no matching parameter is present"
            )
        if selection.has_element_filter and group not in {"positions", "adp", "occupancy"}:
            raise ValueError(f"element filters are not meaningful for trainable group {group!r}")
        row_mask = _row_mask(selection, params=params, atomic_numbers=atomic_numbers)
        for field_name in present:
            fields[field_name] = row_mask
    return MappingProxyType(fields)


def _row_mask(
    selection: AtomSelection,
    *,
    params: RefinableParams,
    atomic_numbers: Tensor | None,
) -> Tensor | None:
    """Resolve an atom selection into a boolean ASU-row mask, or ``None`` for whole group."""
    if not selection.has_element_filter:
        return None
    if atomic_numbers is None:
        raise ValueError("element-filtered trainable selections require atomic numbers")
    if atomic_numbers.shape != (params.asu_positions.shape[0],):
        raise ValueError("atomic_numbers must have shape (N,) matching asu_positions")
    include = _element_numbers(selection.element_include)
    exclude = _element_numbers(selection.element_exclude)
    numbers = atomic_numbers.detach().cpu().to(dtype=torch.int64)
    mask = torch.ones_like(numbers, dtype=torch.bool)
    if include:
        mask = torch.zeros_like(numbers, dtype=torch.bool)
        for number in include:
            mask |= numbers == number
    for number in exclude:
        mask &= numbers != number
    if not bool(mask.any()):
        raise ValueError("trainable atom selection matched no atoms")
    return mask


@dataclass(frozen=True)
class _FieldOverride:
    """A trainable leaf plus optional row mask for reconstructing one parameter field."""

    leaf: Tensor
    row_mask: Tensor | None = None

    def apply(self, baseline: Tensor) -> Tensor:
        """Overlay this leaf onto a detached full-field baseline."""
        if self.row_mask is None:
            return self.leaf
        mask = self.row_mask.to(device=baseline.device)
        full = baseline.clone()
        full[mask] = self.leaf
        return full


@dataclass(frozen=True)
class _TrainableParams:
    """Frozen parameter context plus the trainable leaves that override it.

    This keeps optimizer leaves explicit and reconstructs a full :class:`RefinableParams` value for
    each objective evaluation. Under whole-group selections the overrides are full tensors;
    per-atom selections would reconstruct full fields from smaller selected leaves here.
    """

    frozen: RefinableParams
    overrides: Mapping[str, _FieldOverride]
    leaves: tuple[Tensor, ...]

    def params(self) -> RefinableParams:
        """Reconstruct the full parameter value for the current leaf tensors."""
        values = {
            name: override.apply(getattr(self.frozen, name))
            for name, override in self.overrides.items()
        }
        return dataclasses.replace(self.frozen, **values)


def _to_trainable_params(
    params: RefinableParams, trainable_fields: Mapping[str, Tensor | None]
) -> _TrainableParams:
    """Split ``params`` into detached frozen context and explicit trainable leaves."""
    frozen_values: dict[str, Any] = {}
    overrides: dict[str, _FieldOverride] = {}
    leaves: list[Tensor] = []
    for param_field in dataclasses.fields(RefinableParams):
        value = getattr(params, param_field.name)
        if value is None:
            frozen_values[param_field.name] = None
        elif param_field.name in trainable_fields:
            row_mask = trainable_fields[param_field.name]
            leaf_value = value[row_mask.to(device=value.device)] if row_mask is not None else value
            leaf = leaf_value.detach().clone().requires_grad_(True)
            # Keep the detached full-field baseline even when this whole field is overridden:
            # per-atom leaves reconstruct by scattering selected rows onto it.
            frozen_values[param_field.name] = value.detach().clone()
            overrides[param_field.name] = _FieldOverride(leaf=leaf, row_mask=row_mask)
            leaves.append(leaf)
        else:
            frozen_values[param_field.name] = value.detach().clone()
    return _TrainableParams(
        frozen=dataclasses.replace(params, **frozen_values),
        overrides=MappingProxyType(overrides),
        leaves=tuple(leaves),
    )


def _detach_params(params: RefinableParams) -> RefinableParams:
    """Return a fully-detached clone of ``params`` (no grad history)."""
    detached: dict[str, Any] = {}
    for param_field in dataclasses.fields(RefinableParams):
        value = getattr(params, param_field.name)
        detached[param_field.name] = None if value is None else value.detach().clone()
    return dataclasses.replace(params, **detached)


def _build_optimizer(name: OptimizerName, leaves: list[Tensor], lr: float) -> torch.optim.Optimizer:
    """Build the torch.optim backend over the leaf tensors (single shared learning rate)."""
    if name == "adam":
        return torch.optim.Adam(leaves, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(leaves, lr=lr)
    if name == "lbfgs":
        return torch.optim.LBFGS(leaves, lr=lr)
    raise ValueError(f"optimizer must be 'adam', 'adamw', or 'lbfgs'; got {name!r}")

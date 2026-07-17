"""The imperative refinement loop -- the one deliberately stateful corner of an otherwise pure core.

``torch.optim`` optimizers mutate ``.grad`` and leaf tensors in place and carry internal state, so a
training loop cannot be a pure function. This module quarantines that imperativeness behind a
functional contract: :func:`run_refinement` takes the engine's pure ``objective_value`` callable
and the caller's parameters, clones the *target* fields into fresh ``requires_grad`` leaves (the
rest become detached constants), steps a chosen backend, and returns a new detached
:class:`RefinementResult`.
The caller's parameters are never touched. ``core/`` stays free of ``torch.optim`` entirely.

Deliberately deferred surface: per-group learning rates, ``least_squares``, component ``activate``,
and ``OptimizerState``/history threading.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import torch
from torch import Tensor

from diffBloch.observability import (
    NULL_LOGGER,
    Logger,
    RefinementCompleted,
    RefinementStep,
)
from diffBloch.params import RefinableParams

__all__ = [
    "ObjectiveComponent",
    "AtomSelection",
    "ObjectiveValue",
    "OptimizerName",
    "RefinementResult",
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
    "thickness": ("thickness_raw",),
}


@dataclass(frozen=True)
class AtomSelection:
    """A coarse atom/parameter selection for one trainable group.

    Phase 5 only distinguishes whole-group ``all`` vs ``none``. Later phases can extend this value
    with element or index filters without reintroducing stringly-typed refinement targets.
    """

    mode: Literal["all", "none"]

    def __post_init__(self) -> None:
        """Reject invalid runtime construction, not just invalid static/config values."""
        if self.mode not in {"all", "none"}:
            raise ValueError(f"atom selection mode must be 'all' or 'none'; got {self.mode!r}")

    @classmethod
    def all(cls) -> AtomSelection:
        """Select every present parameter in the group."""
        return cls("all")

    @classmethod
    def none(cls) -> AtomSelection:
        """Select no parameters in the group."""
        return cls("none")

    @property
    def selects_any(self) -> bool:
        """Whether this selection unlocks the corresponding trainable group."""
        return self.mode == "all"


@dataclass(frozen=True)
class TrainableSpec:
    """Explicit selection of which parameter groups are trainable in a refinement problem."""

    positions: AtomSelection = field(default_factory=AtomSelection.none)
    adp: AtomSelection = field(default_factory=AtomSelection.none)
    occupancy: AtomSelection = field(default_factory=AtomSelection.none)
    fgb: AtomSelection = field(default_factory=AtomSelection.none)
    thickness: AtomSelection = field(default_factory=AtomSelection.none)

    @classmethod
    def positions_and_adp(cls) -> TrainableSpec:
        """The historical default: refine positions and ADPs when present."""
        return cls(positions=AtomSelection.all(), adp=AtomSelection.all())


@dataclass(frozen=True)
class ObjectiveComponent:
    """One named refinement objective term.

    ``raw`` is the scientifically meaningful scalar diagnostic (for example, a bond restraint before
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
    weights for future restraint reporting.
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
    so the returned result is unchanged. Measurements are the already-materialised per-step loss, so
    emission adds no extra device sync.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    leaf_params, leaves = _to_leaves(params, _resolve_trainable(params, trainable))
    opt = _build_optimizer(optimizer, leaves, lr)

    def closure() -> float:
        opt.zero_grad()
        loss = objective_value(leaf_params).total
        loss.backward()  # type: ignore[no-untyped-call]
        return float(loss.detach())

    losses: list[float] = []
    best_loss = math.inf
    best_step = 0
    best_params = _detach_params(leaf_params)
    for step in range(steps):
        snapshot = _detach_params(leaf_params)  # params behind this step's pre-update loss
        loss_value = opt.step(closure)
        assert loss_value is not None  # closure is always provided -> step returns the loss
        loss_value = float(loss_value)
        losses.append(loss_value)
        logger.report(RefinementStep(iteration=step, loss=loss_value))
        if loss_value < best_loss:
            best_loss, best_step, best_params = loss_value, step, snapshot
    logger.report(RefinementCompleted(n_steps=steps, best_step=best_step, best_loss=best_loss))
    return RefinementResult(
        params=_detach_params(leaf_params),
        losses=torch.tensor(losses, dtype=torch.float64),
        best_params=best_params,
        best_step=best_step,
    )


def _resolve_trainable(params: RefinableParams, trainable: TrainableSpec) -> frozenset[str]:
    """Map trainable selections to the present RefinableParams field names they unlock."""
    fields: set[str] = set()
    selected_groups = [name for name in _TRAINABLE_FIELDS if getattr(trainable, name).selects_any]
    if not selected_groups:
        raise ValueError("at least one trainable parameter group is required")
    for group in selected_groups:
        present = [name for name in _TRAINABLE_FIELDS[group] if getattr(params, name) is not None]
        if not present:
            raise ValueError(
                f"trainable group {group!r} selected but no matching parameter is present"
            )
        fields.update(present)
    return frozenset(fields)


def _to_leaves(
    params: RefinableParams, trainable_fields: frozenset[str]
) -> tuple[RefinableParams, list[Tensor]]:
    """Clone ``params`` so ``trainable_fields`` are fresh grad leaves, the rest constants."""
    values: dict[str, Any] = {}
    leaves: list[Tensor] = []
    for param_field in dataclasses.fields(RefinableParams):
        value = getattr(params, param_field.name)
        if value is None:
            values[param_field.name] = None
        elif param_field.name in trainable_fields:
            leaf = value.detach().clone().requires_grad_(True)
            values[param_field.name] = leaf
            leaves.append(leaf)
        else:
            values[param_field.name] = value.detach().clone()
    return dataclasses.replace(params, **values), leaves


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

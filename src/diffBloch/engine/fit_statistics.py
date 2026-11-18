"""Jana2020-convention fit statistics: pooled R/wR on ``|F|`` and goodness of fit (GoF).

Report-only. Nothing here enters the refinement objective, the orientation/thickness searches, or
any ``Plan``; it scores an already-settled model's matched reflections the way Jana2020 scores a
dynamical refinement, so diffBloch numbers can be put next to Jana numbers on the same scale.

The convention, pinned against Jana2020's own per-reflection output (quartz, borane, CsPbBr3 and
LuAG reproduce to every printed digit -- see ``tests/unit/test_fit_statistics.py``):

- refinement on ``|F|``: ``Fo = sqrt(max(I_obs, 0))``, ``Fc = k*sqrt(I_calc)`` with one amplitude
  scale ``k`` per frame (Jana's per-zone ``EDScale``), fitted here in closed form as the weighted
  least-squares optimum ``k = sum w Fo Fc' / sum w Fc'^2`` over every matched reflection of the
  frame (``w`` depends on ``Fo`` only, so this is exact);
- ``sigma(F)`` from :func:`diffBloch.core.losses.amplitude_sigma` (``sigma(I)/(2 Fo)``, weak
  reflections floored at ``5 sqrt(sigma(I))``);
- weight ``w = 1/(sigma(F)^2 + (u*Fo)^2)`` with instability factor ``u = 0.01``;
- "obs" are reflections with ``I > 3 sigma(I)``; "all" are every matched reflection, negative
  intensities included at ``Fo = 0``. ``N`` counts per-frame measurements, not unique ``hkl``;
- ``R_obs = sum|Fo - Fc| / sum Fo`` and ``wR = sqrt(sum w (Fo - Fc)^2 / sum w Fo^2)``, pooled over
  every frame (one sum, not a mean of per-frame values);
- ``GoF = sqrt(sum w (Fo - Fc)^2 / (N - p))``, where ``p`` excludes the per-frame scales. Jana
  lists those scales in its parameter count but does not charge them in GoF: with them counted,
  none of the reference refinements reproduces; without them, all do.

Weights follow Palatinus et al. (2015), Acta Cryst. A71, 235; the sigma floor and the R_sg/D_sg
reflection window follow Klar et al. (2023), Nature Chemistry, and its supplementary information.

``1/sigma^2`` variants (no instability term) are reported alongside: with ``u > 0`` the weights
cap the influence of strong reflections, so a Jana-weighted GoF near 1 is a weaker statement about
the noise level than a ``1/sigma^2`` GoF near 1.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import torch
from torch import Tensor

from diffBloch.core.losses import amplitude_sigma
from diffBloch.engine.constraints import ConstraintTransform
from diffBloch.engine.refine import TrainableSpec, _resolve_trainable
from diffBloch.params import ConstraintSpec, RefinableParams, constrain

__all__ = [
    "FitStatistics",
    "FitSums",
    "FrameFit",
    "ParameterCount",
    "fit_statistics",
    "frame_fit",
    "structural_parameter_count",
]

JANA_INSTABILITY = 0.01

type ReflectionSubset = Literal["obs", "all"]
type FitWeights = Literal["jana", "inverse_variance"]


@dataclass(frozen=True)
class FitSums:
    """Additive residual sums over a set of reflections, so frames pool by plain addition.

    ``weighted_residual`` is ``sum w (Fo - Fc)^2`` under Jana weights, ``inverse_variance_residual``
    the same with ``w = 1/sigma(F)^2``, ``absolute_residual`` ``sum |Fo - Fc|``, ``amplitude``
    ``sum Fo``, and ``weighted_amplitude`` ``sum w Fo^2`` (Jana weights).
    """

    n: int = 0
    weighted_residual: float = 0.0
    inverse_variance_residual: float = 0.0
    absolute_residual: float = 0.0
    amplitude: float = 0.0
    weighted_amplitude: float = 0.0

    def __add__(self, other: FitSums) -> FitSums:
        return FitSums(
            n=self.n + other.n,
            weighted_residual=self.weighted_residual + other.weighted_residual,
            inverse_variance_residual=self.inverse_variance_residual
            + other.inverse_variance_residual,
            absolute_residual=self.absolute_residual + other.absolute_residual,
            amplitude=self.amplitude + other.amplitude,
            weighted_amplitude=self.weighted_amplitude + other.weighted_amplitude,
        )


@dataclass(frozen=True)
class FrameFit:
    """One frame's residual sums for the ``obs`` and ``all`` subsets plus its fitted F scale."""

    observed_reflections: FitSums
    all_reflections: FitSums
    amplitude_scale: float


def _sums(fo: Tensor, fc: Tensor, sigma_f: Tensor, w: Tensor, mask: Tensor) -> FitSums:
    delta = fo[mask] - fc[mask]
    return FitSums(
        n=int(mask.sum()),
        weighted_residual=float((w[mask] * delta**2).sum()),
        inverse_variance_residual=float(((delta / sigma_f[mask]) ** 2).sum()),
        absolute_residual=float(delta.abs().sum()),
        amplitude=float(fo[mask].sum()),
        weighted_amplitude=float((w[mask] * fo[mask] ** 2).sum()),
    )


def frame_fit(
    calculated: Sequence[float] | Tensor,
    observed: Sequence[float] | Tensor,
    sigmas: Sequence[float] | Tensor,
    *,
    instability: float = JANA_INSTABILITY,
) -> FrameFit:
    """Score one frame's matched reflections under the Jana2020 convention.

    ``calculated`` may be on any intensity scale: the per-frame amplitude scale is refitted here
    (weighted least squares on ``|F|``, over all matched reflections), so the result does not
    depend on how the caller scaled it. Evaluated in float64 on the CPU; this is a report, not a
    differentiable term.
    """
    calc = torch.as_tensor(calculated, dtype=torch.float64).detach().cpu()
    obs = torch.as_tensor(observed, dtype=torch.float64).detach().cpu()
    sig = torch.as_tensor(sigmas, dtype=torch.float64).detach().cpu()
    if not (calc.shape == obs.shape == sig.shape) or calc.ndim != 1:
        raise ValueError("calculated, observed and sigmas must be equal-length 1-D sequences")
    if instability < 0:
        raise ValueError(f"instability must be >= 0, got {instability}")
    fo = obs.clamp(min=0.0).sqrt()
    fc_unscaled = calc.clamp(min=0.0).sqrt()
    sigma_f = amplitude_sigma(obs, sig)
    w = 1.0 / (sigma_f**2 + (instability * fo) ** 2)
    denominator = float((w * fc_unscaled**2).sum())
    scale = float((w * fo * fc_unscaled).sum()) / denominator if denominator > 0 else math.nan
    fc = scale * fc_unscaled
    strong = obs > 3.0 * sig
    every = torch.ones_like(strong)
    return FrameFit(
        observed_reflections=_sums(fo, fc, sigma_f, w, strong),
        all_reflections=_sums(fo, fc, sigma_f, w, every),
        amplitude_scale=scale,
    )


@dataclass(frozen=True)
class ParameterCount:
    """The parameter count ``p`` behind a GoF, itemised so it can be read and argued with.

    - ``structural``: independent structural parameters actually free in the refinement
      (:func:`structural_parameter_count`: special-position and ADP symmetry already removed).
    - ``components``: trainable component tensors (thickness network weights, mosaicity width).
    - ``per_frame_thickness``: thicknesses fitted per frame and used as such by the model.
    - ``per_frame_orientation``: orientation corrections fitted per frame in preprocessing.
    - ``per_frame_scales``: one intensity scale per frame. Reported, but never charged in ``p``,
      following Jana2020.

    :attr:`without_orientation` and :attr:`with_orientation` are the two ``p`` values GoF is
    reported at.
    """

    structural: int
    components: int = 0
    per_frame_thickness: int = 0
    per_frame_orientation: int = 0
    per_frame_scales: int = 0

    def __post_init__(self) -> None:
        for name in (
            "structural",
            "components",
            "per_frame_thickness",
            "per_frame_orientation",
            "per_frame_scales",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")

    @property
    def without_orientation(self) -> int:
        """``p`` with the per-frame orientation corrections left out."""
        return self.structural + self.components + self.per_frame_thickness

    @property
    def with_orientation(self) -> int:
        """``p`` with the per-frame orientation corrections charged."""
        return self.without_orientation + self.per_frame_orientation


@dataclass(frozen=True)
class FitStatistics:
    """Pooled Jana2020-convention fit statistics for a settled model.

    Fractions, not percentages. Every value is ``nan`` when undefined (no reflections, or
    ``N <= p`` for GoF).
    """

    observed_reflections: FitSums
    all_reflections: FitSums
    parameters: ParameterCount
    n_frames: int

    def _subset(self, subset: ReflectionSubset) -> FitSums:
        if subset == "obs":
            return self.observed_reflections
        if subset == "all":
            return self.all_reflections
        raise ValueError(f"subset must be 'obs' or 'all', got {subset!r}")

    def r_factor(self, subset: ReflectionSubset = "obs") -> float:
        """``sum |Fo - Fc| / sum Fo`` over the subset."""
        sums = self._subset(subset)
        return sums.absolute_residual / sums.amplitude if sums.amplitude > 0 else math.nan

    def weighted_r_factor(self, subset: ReflectionSubset) -> float:
        """Jana's ``wR = sqrt(sum w (Fo - Fc)^2 / sum w Fo^2)`` over the subset."""
        sums = self._subset(subset)
        if sums.weighted_amplitude <= 0:
            return math.nan
        return math.sqrt(sums.weighted_residual / sums.weighted_amplitude)

    def goodness_of_fit(
        self,
        subset: ReflectionSubset,
        *,
        include_orientation: bool,
        weights: FitWeights = "jana",
    ) -> float:
        """``sqrt(sum w (Fo - Fc)^2 / (N - p))`` over the subset.

        ``include_orientation`` picks :attr:`ParameterCount.with_orientation` over
        :attr:`ParameterCount.without_orientation`; ``weights`` picks Jana weights or ``1/sigma^2``.
        """
        sums = self._subset(subset)
        p = (
            self.parameters.with_orientation
            if include_orientation
            else self.parameters.without_orientation
        )
        dof = sums.n - p
        if dof <= 0:
            return math.nan
        if weights == "jana":
            residual = sums.weighted_residual
        elif weights == "inverse_variance":
            residual = sums.inverse_variance_residual
        else:
            raise ValueError(f"weights must be 'jana' or 'inverse_variance', got {weights!r}")
        return math.sqrt(residual / dof)


def fit_statistics(
    frames: Iterable[
        tuple[Sequence[float] | Tensor, Sequence[float] | Tensor, Sequence[float] | Tensor]
    ],
    parameters: ParameterCount,
    *,
    instability: float = JANA_INSTABILITY,
) -> FitStatistics:
    """Pool :func:`frame_fit` over frames given as ``(calculated, observed, sigmas)`` triples."""
    observed = FitSums()
    every = FitSums()
    n_frames = 0
    for calculated, measured, sigmas in frames:
        fit = frame_fit(calculated, measured, sigmas, instability=instability)
        observed = observed + fit.observed_reflections
        every = every + fit.all_reflections
        n_frames += 1
    return FitStatistics(
        observed_reflections=observed,
        all_reflections=every,
        parameters=parameters,
        n_frames=n_frames,
    )


def structural_parameter_count(
    params: RefinableParams,
    spec: ConstraintSpec,
    trainable: TrainableSpec,
    *,
    constraints: tuple[ConstraintTransform, ...] = (),
    atomic_numbers: Tensor | None = None,
) -> int:
    """Number of independent structural parameters the refinement actually varied.

    The optimizer's raw leaves over-count: a special-position atom carries three raw coordinates
    but fewer free ones after site-symmetry projection, an isotropic ADP fills six ``U*`` entries
    from one number, and a riding hydrogen has no freedom of its own. The honest count is the rank
    of the Jacobian of the physical state (positions, ``U*``, occupancies after :func:`constrain`
    and every ``constraints`` transform) with respect to the selected raw leaves, evaluated at
    ``params``. Leaves are selected exactly as the refinement selects them (``trainable`` plus
    ``atomic_numbers`` for element filters).
    """
    fields = _resolve_trainable(params, trainable, atomic_numbers=atomic_numbers)
    names = list(fields)
    baselines: dict[str, Tensor] = {}
    pieces: list[Tensor] = []
    for name in names:
        value = getattr(params, name)
        assert value is not None  # _resolve_trainable only returns present fields
        baseline = value.detach().to(dtype=torch.float64)
        baselines[name] = baseline
        mask = fields[name]
        selected = baseline if mask is None else baseline[mask.to(device=baseline.device)]
        pieces.append(selected.reshape(-1))
    sizes = [piece.numel() for piece in pieces]
    x0 = torch.cat(pieces)

    def as_double(value: Tensor | None) -> Tensor | None:
        return None if value is None else value.detach().to(dtype=torch.float64)

    base = replace(
        params,
        asu_positions=params.asu_positions.detach().to(dtype=torch.float64),
        uij_raw=as_double(params.uij_raw),
        u_iso_raw=as_double(params.u_iso_raw),
        occupancy_raw=as_double(params.occupancy_raw),
    )

    def physical(x: Tensor) -> Tensor:
        overrides: dict[str, Tensor] = {}
        for name, chunk in zip(names, torch.split(x, sizes), strict=True):
            mask = fields[name]
            baseline = baselines[name]
            if mask is None:
                overrides[name] = chunk.reshape(baseline.shape)
            else:
                row_mask = mask.to(device=baseline.device)
                full = baseline.clone()
                full[row_mask] = chunk.reshape(full[row_mask].shape)
                overrides[name] = full
        state = constrain(replace(base, **overrides), spec)
        for constraint in constraints:
            state = constraint.apply(state)
        return torch.cat(
            [
                state.positions.reshape(-1),
                state.uij_star.reshape(-1),
                state.occupancies.reshape(-1),
            ]
        )

    # torch ships jacobian without annotations; for one Tensor input/output it returns one Tensor.
    jacobian: Tensor = torch.autograd.functional.jacobian(physical, x0)  # type: ignore[no-untyped-call]
    return int(torch.linalg.matrix_rank(jacobian.detach().cpu()))

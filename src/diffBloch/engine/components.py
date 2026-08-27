"""Refinement model components that provide forward-model values."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor

from diffBloch.core.constraints import positive
from diffBloch.engine.forward import ForwardContext
from diffBloch.engine.plan import OrientationPlanLike
from diffBloch.observability import ThicknessProfile

__all__ = [
    "ApparentThicknessNN",
    "PerOrientationThickness",
    "QuadraticThicknessProfile",
    "ThicknessBounds",
    "TrainableIsotropicMosaicity",
]


@dataclass(frozen=True)
class ThicknessBounds:
    """Hard bounded parameterization for apparent thickness values."""

    min_angstrom: float
    max_angstrom: float

    def __post_init__(self) -> None:
        if self.min_angstrom <= 0:
            raise ValueError("min_angstrom must be positive")
        if self.max_angstrom <= self.min_angstrom:
            raise ValueError("max_angstrom must be greater than min_angstrom")

    def transform(self, unconstrained: Tensor) -> Tensor:
        """Map unconstrained parameter values into ``[min_angstrom, max_angstrom]``."""
        span = self.max_angstrom - self.min_angstrom
        return self.min_angstrom + torch.sigmoid(unconstrained) * span

    def inverse(self, thickness: Tensor) -> Tensor:
        """Map physical thickness values inside the bounds back to unconstrained values."""
        if bool((thickness <= self.min_angstrom).any()) or bool(
            (thickness >= self.max_angstrom).any()
        ):
            raise ValueError("initial thickness values must lie strictly inside ThicknessBounds")
        span = self.max_angstrom - self.min_angstrom
        fraction = (thickness - self.min_angstrom) / span
        return torch.logit(fraction)


@dataclass(frozen=True)
class PerOrientationThickness:
    """One trainable positive thickness vector per orientation.

    This is the simplest concrete component: it seeds params tensors from the prepared Plan's fixed
    orientation thicknesses and supplies ``positive(params[rotation_index])`` during the forward
    solve. Each orientation carries its own free thickness, with no sharing across orientations.
    """

    key: str = "per_orientation_thickness"

    def initial_params(
        self,
        *,
        plan: Sequence[OrientationPlanLike],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Mapping[str, Tensor]:
        if not plan:
            raise ValueError("plan has no orientations to seed thickness from")
        thicknesses = [orientation.thickness.to(device=device, dtype=dtype) for orientation in plan]
        shape = thicknesses[0].shape
        if any(thickness.shape != shape for thickness in thicknesses):
            raise ValueError("all orientation thickness vectors must have the same shape")
        stacked = torch.stack(thicknesses)
        return {"unconstrained": _positive_inverse(stacked)}

    def forward_context(
        self,
        params: Mapping[str, Tensor],
        *,
        rotation_index: int,
        orientation: OrientationPlanLike,
    ) -> ForwardContext:
        _ = orientation
        if "unconstrained" not in params:
            raise ValueError(
                "per-orientation thickness component requires an 'unconstrained' tensor"
            )
        values = params["unconstrained"]
        if values.ndim != 2:
            raise ValueError("per-orientation thickness params tensor must have shape (O, T)")
        if rotation_index < 0 or rotation_index >= values.shape[0]:
            raise ValueError("rotation_index is outside the per-orientation thickness tensor")
        return ForwardContext(thickness=positive(values[rotation_index]))


@dataclass(frozen=True)
class QuadraticThicknessProfile:
    """Bounded low-dimensional thickness profile over orientation angle.

    The component evaluates ``a0 + a1*x + a2*x^2`` where ``x`` is the rotation angle from identity
    normalized by pi, then applies :class:`ThicknessBounds`. It ties thickness to orientation with
    three shared coefficients rather than one free value per orientation, an interpretable
    middle ground between free per-orientation thicknesses and a learned apparent-thickness model.
    """

    bounds: ThicknessBounds
    key: str = "quadratic_thickness"

    def initial_params(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
        initial_thickness: Tensor | float | None = None,
    ) -> Mapping[str, Tensor]:
        coefficients = torch.zeros((3,), dtype=dtype, device=device)
        coefficients[0] = _initial_unconstrained_thickness(
            self.bounds, initial_thickness, dtype=dtype, device=device
        )
        return {"coefficients": coefficients}

    def forward_context(
        self,
        params: Mapping[str, Tensor],
        *,
        rotation_index: int,
        orientation: OrientationPlanLike,
    ) -> ForwardContext:
        _ = rotation_index
        if "coefficients" not in params:
            raise ValueError("quadratic thickness component requires a 'coefficients' tensor")
        coefficients = params["coefficients"]
        if coefficients.shape != (3,):
            raise ValueError("quadratic thickness coefficients must have shape (3,)")
        x = _orientation_angle_fraction(orientation.orientation.to(coefficients.device))
        raw_thickness = coefficients[0] + coefficients[1] * x + coefficients[2] * x.square()
        thickness = self.bounds.transform(raw_thickness).reshape(1)
        return ForwardContext(thickness=thickness)


def _initial_unconstrained_thickness(
    bounds: ThicknessBounds,
    initial_thickness: Tensor | float | None,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    if initial_thickness is None:
        return torch.zeros((), dtype=dtype, device=device)
    thickness = torch.as_tensor(initial_thickness, dtype=dtype, device=device)
    return bounds.inverse(thickness.reshape(()))


def _orientation_angle_fraction(orientation: Tensor) -> Tensor:
    trace = torch.trace(orientation)
    cosine = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.acos(cosine) / torch.pi


@dataclass(frozen=True)
class ApparentThicknessNN:
    """Per-dataset apparent-thickness neural network.

    The fixed ``1 -> 64 -> 64 -> 2`` tanh MLP consumes the PETS alpha angle after per-dataset
    min-max normalization to ``[-1, 1]``. Its first output is mapped affinely from that nominal
    interval into ``bounds``; as in the legacy implementation this is not a hard bound. In sampling
    mode the second output parameterizes a Gaussian width and the reparameterized, positive
    thickness samples are passed to the Bloch solve.

    Each network is scoped to one dataset's half-open ``rotation_range`` in pooled source-rotation
    indices, with ``normalized_alphas`` holding exactly that range's values. A rotation outside the
    range yields a context with no thickness: the model composition takes the one component that
    does claim it, so per-dataset networks partition the pooled index space without any shared
    routing table. ``label`` is the dataset's ``inputs.exp_data`` ref and names the component in
    reports. Pooled datasets deliberately share ``init_seed`` -- identical initialization is
    deterministic, and per-position seeds would make results depend on ``exp_data`` ordering.

    Legacy code drew new random samples on every forward call. Here the standard-normal draws are
    fixed by ``init_seed`` and source rotation index, retaining the same sampled model while keeping
    the objective deterministic.
    """

    bounds: ThicknessBounds
    normalized_alphas: tuple[float, ...]
    key: str = "apparent_thickness"
    form: Literal["min_thickness"] = "min_thickness"
    sample_thickness: bool = False
    num_samples: int = 40
    init_seed: int = 0
    rotation_range: tuple[int, int] = field(kw_only=True)
    label: str = field(kw_only=True)

    def __post_init__(self) -> None:
        if self.form != "min_thickness":
            raise ValueError("only form='min_thickness' is implemented")
        if not self.normalized_alphas:
            raise ValueError("normalized_alphas must contain every source PETS rotation")
        if any(not -1.0 <= alpha <= 1.0 for alpha in self.normalized_alphas):
            raise ValueError("normalized_alphas must lie in [-1, 1]")
        if self.num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        start, end = self.rotation_range
        if start < 0 or end <= start:
            raise ValueError("rotation_range must be a non-empty [start, end) with start >= 0")
        if end - start != len(self.normalized_alphas):
            raise ValueError(
                "rotation_range width must match len(normalized_alphas) "
                f"({end - start} != {len(self.normalized_alphas)})"
            )

    def initial_params(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Mapping[str, Tensor]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.init_seed)

        def linear(out_features: int, in_features: int) -> tuple[Tensor, Tensor]:
            # Match nn.Linear.reset_parameters exactly, including random-number consumption order.
            bound = in_features**-0.5
            weight = torch.empty((out_features, in_features), dtype=dtype)
            torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5), generator=generator)
            bias = torch.empty((out_features,), dtype=dtype).uniform_(
                -bound, bound, generator=generator
            )
            return weight.to(device), bias.to(device)

        w0, b0 = linear(64, 1)
        w1, b1 = linear(64, 64)
        w2, b2 = linear(2, 64)
        return {
            "layer0.weight": w0,
            "layer0.bias": b0,
            "layer1.weight": w1,
            "layer1.bias": b1,
            "layer2.weight": w2,
            "layer2.bias": b2,
        }

    def profile(
        self,
        params: Mapping[str, Tensor],
        orientations: Sequence[OrientationPlanLike],
        raw_alphas: Sequence[float] | NDArray[np.float64],
    ) -> ThicknessProfile:
        """Evaluate the trained curve at every in-range orientation's tilt angle, reportably.

        The component owns this because it owns the behaviour: sampling its own output is not
        something an orchestrator or a logger should reimplement, and a sink must never run the
        forward model itself. That ownership includes the scope -- callers pass the full pooled
        ``orientations`` and each network keeps its own ``rotation_range``. ``raw_alphas`` is
        indexed by pooled source PETS rotation index, so it too is passed unsliced.
        """
        start, end = self.rotation_range
        in_range = [
            orientation
            for orientation in orientations
            if start <= orientation.pattern.rotation_index < end
        ]
        indices = [orientation.pattern.rotation_index for orientation in in_range]
        thicknesses: list[float] = []
        for index, orientation in zip(indices, in_range, strict=True):
            context = self.forward_context(params, rotation_index=index, orientation=orientation)
            if context.thickness is None:
                raise ValueError("thickness component produced no thickness")
            thicknesses.append(float(context.thickness.reshape(-1)[0]))
        return ThicknessProfile(
            form=self.form,
            min_thickness=self.bounds.min_angstrom,
            max_thickness=self.bounds.max_angstrom,
            rotation_indices=tuple(indices),
            alphas=tuple(float(raw_alphas[index]) for index in indices),
            thicknesses=tuple(thicknesses),
            label=self.label,
        )

    def forward_context(
        self,
        params: Mapping[str, Tensor],
        *,
        rotation_index: int,
        orientation: OrientationPlanLike,
    ) -> ForwardContext:
        _ = rotation_index
        required = (
            "layer0.weight",
            "layer0.bias",
            "layer1.weight",
            "layer1.bias",
            "layer2.weight",
            "layer2.bias",
        )
        missing = [name for name in required if name not in params]
        if missing:
            raise ValueError(f"apparent thickness NN params tensors missing {missing!r}")
        w0 = params["layer0.weight"]
        source_index = orientation.pattern.rotation_index
        start, end = self.rotation_range
        if source_index < start or source_index >= end:
            # Another dataset's rotation: the network that does claim it supplies the thickness.
            return ForwardContext()
        local_index = source_index - start
        x = w0.new_tensor(self.normalized_alphas[local_index]).reshape(1, 1)
        x = torch.tanh(F.linear(x, w0, params["layer0.bias"]))
        x = torch.tanh(F.linear(x, params["layer1.weight"], params["layer1.bias"]))
        output = F.linear(x, params["layer2.weight"], params["layer2.bias"])
        mu = output[0, 0]
        span = self.bounds.max_angstrom - self.bounds.min_angstrom
        mu = self.bounds.min_angstrom + (mu + 1.0) * span / 2.0
        if not self.sample_thickness:
            return ForwardContext(thickness=mu.reshape(1))

        log_sigma = output[0, 1]
        sigma = torch.exp(log_sigma)
        sigma = 1.0 + 199.0 * torch.sigmoid(sigma)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.init_seed)
        epsilon = torch.randn(
            (len(self.normalized_alphas), self.num_samples),
            generator=generator,
            dtype=w0.dtype,
        )[local_index].to(w0.device)
        thickness = F.softplus(mu + sigma * epsilon)
        return ForwardContext(thickness=thickness)


@dataclass(frozen=True)
class TrainableIsotropicMosaicity:
    """One trainable isotropic-mosaicity sigma, shared across every rotation in the dataset.

    Mosaicity is a bulk property of the crystal, not of any one orientation, so unlike thickness
    there is a single free parameter here, not one per rotation. It refines by *reweighting* a fixed
    set of mosaic-tilt directions rather than moving them: the directions
    (:func:`~diffBloch.preprocess.orientation.isotropic_mosaic_tilts`'s Rayleigh-quantile placement)
    were baked into each orientation's structure-factor gather when the ``Plan`` was built, at
    ``init_sigma_degrees``, and moving them would mean rebuilding that gather every refinement step.
    Refining sigma instead smoothly redistributes weight across those same fixed directions --
    concentrated near the center as sigma shrinks, closer to uniform as it grows -- which captures
    the same qualitative "how much mosaic broadening" effect without the per-step rebuild cost. It
    cannot represent a spread genuinely wider than the fixed placement covers.

    ``polar_degrees`` is the per-mosaic-sample angular distance from center (one entry per mosaic
    sample, *not* tiled across the rocking-curve tilts -- :meth:`forward_context` tiles it to match
    each orientation's actual tilt count, which is a multiple of ``len(polar_degrees)``).

    ``rotation_range`` scopes this component to one dataset's half-open ``[start, end)`` pooled
    source-rotation indices, exactly like :class:`ApparentThicknessNN`: a pooled multi-dataset
    experiment gets one component per dataset (each with its own resolved sigma/samples), and a
    rotation outside a given component's range gets no override from it, falling through to
    whichever component does claim it.
    """

    polar_degrees: tuple[float, ...]
    init_sigma_degrees: float
    rotation_range: tuple[int, int]
    key: str = "isotropic_mosaicity"

    def __post_init__(self) -> None:
        if not self.polar_degrees:
            raise ValueError("polar_degrees must be non-empty")
        if self.init_sigma_degrees <= 0.0:
            raise ValueError("init_sigma_degrees must be positive")
        start, end = self.rotation_range
        if start < 0 or end <= start:
            raise ValueError("rotation_range must be a non-empty [start, end) with start >= 0")

    def initial_params(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Mapping[str, Tensor]:
        sigma = torch.tensor([self.init_sigma_degrees], dtype=dtype, device=device)
        return {"unconstrained_sigma": _positive_inverse(sigma)}

    def forward_context(
        self,
        params: Mapping[str, Tensor],
        *,
        rotation_index: int,
        orientation: OrientationPlanLike,
    ) -> ForwardContext:
        _ = rotation_index
        source_index = orientation.pattern.rotation_index
        start, end = self.rotation_range
        if source_index < start or source_index >= end:
            # Another dataset's rotation: the component that does claim it supplies the weights.
            return ForwardContext()
        if "unconstrained_sigma" not in params:
            raise ValueError(
                "isotropic mosaicity component requires an 'unconstrained_sigma' tensor"
            )
        raw = params["unconstrained_sigma"]
        sigma = positive(raw).reshape(())
        polar = raw.new_tensor(self.polar_degrees)
        weights = torch.exp(-(polar.square()) / (2.0 * sigma.square()))
        n_tilts = int(orientation.tilts.shape[0])
        n_mosaic = len(self.polar_degrees)
        if n_tilts % n_mosaic != 0:
            raise ValueError(
                f"orientation has {n_tilts} tilts, not a multiple of the {n_mosaic} mosaic "
                "samples this component was built from"
            )
        tiled = weights.repeat(n_tilts // n_mosaic)
        return ForwardContext(mosaic_weights=tiled)


def _positive_inverse(value: Tensor) -> Tensor:
    if bool((value <= 0).any()):
        raise ValueError("thickness values must be positive")
    return torch.log(torch.expm1(value))

"""Refinement model components that provide forward-model values."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
    """Legacy apparent-thickness neural network from checkpoint-preprocess.

    The fixed ``1 -> 64 -> 64 -> 2`` tanh MLP consumes the PETS alpha angle after dataset-wide
    min-max normalization to ``[-1, 1]``. Its first output is mapped affinely from that nominal
    interval into ``bounds``; as in the legacy implementation this is not a hard bound. In sampling
    mode the second output parameterizes a Gaussian width and the reparameterized, positive
    thickness samples are passed to the Bloch solve.

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

    def __post_init__(self) -> None:
        if self.form != "min_thickness":
            raise ValueError("only form='min_thickness' is implemented")
        if not self.normalized_alphas:
            raise ValueError("normalized_alphas must contain every source PETS rotation")
        if any(not -1.0 <= alpha <= 1.0 for alpha in self.normalized_alphas):
            raise ValueError("normalized_alphas must lie in [-1, 1]")
        if self.num_samples < 1:
            raise ValueError("num_samples must be >= 1")

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
        """Evaluate the trained curve at every orientation's tilt angle, as a reportable value.

        The component owns this because it owns the behaviour: sampling its own output is not
        something an orchestrator or a logger should reimplement, and a sink must never run the
        forward model itself. ``raw_alphas`` is indexed by source PETS rotation index.
        """
        indices = [orientation.pattern.rotation_index for orientation in orientations]
        thicknesses: list[float] = []
        for index, orientation in zip(indices, orientations, strict=True):
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
        if source_index < 0 or source_index >= len(self.normalized_alphas):
            raise ValueError("source rotation index is outside normalized_alphas")
        x = w0.new_tensor(self.normalized_alphas[source_index]).reshape(1, 1)
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
        )[source_index].to(w0.device)
        thickness = F.softplus(mu + sigma * epsilon)
        return ForwardContext(thickness=thickness)


def _positive_inverse(value: Tensor) -> Tensor:
    if bool((value <= 0).any()):
        raise ValueError("thickness values must be positive")
    return torch.log(torch.expm1(value))

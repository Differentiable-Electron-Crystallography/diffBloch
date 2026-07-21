"""Refinement model components that provide forward-model values."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from diffBloch.core.constraints import positive
from diffBloch.engine.forward import ForwardContext
from diffBloch.engine.plan import OrientationPlanLike

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
    orientation thicknesses and supplies ``positive(params[orientation_index])`` during the forward
    solve. It is an ablation/proof component before adding the apparent-thickness neural network.
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
        orientation_index: int,
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
        if orientation_index < 0 or orientation_index >= values.shape[0]:
            raise ValueError("orientation_index is outside the per-orientation thickness tensor")
        return ForwardContext(thickness=positive(values[orientation_index]))


@dataclass(frozen=True)
class QuadraticThicknessProfile:
    """Bounded low-dimensional thickness profile over orientation angle.

    The component evaluates ``a0 + a1*x + a2*x^2`` where ``x`` is the rotation angle from identity
    normalized by pi, then applies :class:`ThicknessBounds`. It is an interpretable bridge between
    per-orientation free thicknesses and a neural apparent-thickness model.
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
        orientation_index: int,
        orientation: OrientationPlanLike,
    ) -> ForwardContext:
        _ = orientation_index
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
    """Paper-style bounded apparent-thickness neural network component.

    The network maps orientation angle to two outputs ``(mu, sigma_raw)`` using the paper MLP
    architecture. The first implementation consumes only ``mu`` and rejects stochastic thickness
    sampling until the sigma likelihood/sampling path is wired.
    """

    bounds: ThicknessBounds
    key: str = "apparent_thickness"
    form: Literal["min_thickness"] = "min_thickness"
    hidden_width: int = 64
    sample_thickness: bool = False
    num_samples: int = 1
    init_seed: int = 0
    init_scale: float = 1.0e-3

    def __post_init__(self) -> None:
        if self.form != "min_thickness":
            raise ValueError("only form='min_thickness' is implemented")
        if self.hidden_width < 1:
            raise ValueError("hidden_width must be >= 1")
        if self.sample_thickness:
            raise ValueError("sample_thickness=True is not implemented")
        if self.num_samples != 1:
            raise ValueError("num_samples must be 1 until thickness sampling is implemented")
        if self.init_scale <= 0:
            raise ValueError("init_scale must be positive")

    def initial_params(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
        initial_thickness: Tensor | float | None = None,
    ) -> Mapping[str, Tensor]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.init_seed)

        def weight(shape: tuple[int, ...]) -> Tensor:
            value = torch.randn(shape, generator=generator, dtype=dtype) * self.init_scale
            return value.to(device)

        def bias(shape: tuple[int, ...]) -> Tensor:
            return torch.zeros(shape, dtype=dtype, device=device)

        width = self.hidden_width
        output_bias = bias((2,))
        output_bias[0] = _initial_unconstrained_thickness(
            self.bounds, initial_thickness, dtype=dtype, device=device
        )
        return {
            "layer0.weight": weight((width, 1)),
            "layer0.bias": bias((width,)),
            "layer1.weight": weight((width, width)),
            "layer1.bias": bias((width,)),
            "layer2.weight": weight((2, width)),
            "layer2.bias": output_bias,
        }

    def forward_context(
        self,
        params: Mapping[str, Tensor],
        *,
        orientation_index: int,
        orientation: OrientationPlanLike,
    ) -> ForwardContext:
        _ = orientation_index
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
        x = _orientation_angle_fraction(orientation.orientation.to(w0.device)).reshape(1, 1)
        x = x.to(dtype=w0.dtype)
        x = torch.tanh(F.linear(x, w0, params["layer0.bias"]))
        x = torch.tanh(F.linear(x, params["layer1.weight"], params["layer1.bias"]))
        output = F.linear(x, params["layer2.weight"], params["layer2.bias"])
        mu = output[0, 0]
        return ForwardContext(thickness=self.bounds.transform(mu).reshape(1))


def _positive_inverse(value: Tensor) -> Tensor:
    if bool((value <= 0).any()):
        raise ValueError("thickness values must be positive")
    return torch.log(torch.expm1(value))

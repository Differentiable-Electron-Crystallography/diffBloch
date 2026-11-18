"""Jana2020-convention fit statistics: oracle parity, invariances, and parameter counting."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from diffBloch.core.losses import amplitude_sigma
from diffBloch.engine import (
    AtomSelection,
    ParameterCount,
    TrainableSpec,
    fit_statistics,
    frame_fit,
    structural_parameter_count,
)
from diffBloch.io import read_structure
from diffBloch.preprocess import RefinementSetup

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
JANA = FIXTURES / "jana_fit_statistics"


def _jana_frames(name: str) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], int]:
    calc, meas, sigma, group_column = np.loadtxt(
        JANA / f"{name}.csv.gz", delimiter=",", skiprows=1, unpack=True
    )
    groups = group_column.astype(int)
    frames = [
        (calc[groups == group], meas[groups == group], sigma[groups == group])
        for group in np.unique(groups)
    ]
    return frames, len(frames)


@pytest.mark.parametrize("name", ["quartz", "borane"])
def test_reproduces_jana_r_wr_and_gof_from_janas_own_reflections(name: str) -> None:
    expected = json.loads((JANA / "expected.json").read_text())[name]
    frames, n_zones = _jana_frames(name)
    stats = fit_statistics(
        frames,
        ParameterCount(
            structural=int(expected["number_parameters"]) - n_zones,
            per_frame_scales=n_zones,
        ),
    )

    assert stats.all_reflections.n == expected["number_reflns"]
    assert stats.observed_reflections.n == expected["reflns_number_gt"]
    # Jana prints four decimals; agreement is to within that rounding.
    assert stats.r_factor("obs") == pytest.approx(expected["R_factor_gt"], abs=5e-5)
    assert stats.r_factor("all") == pytest.approx(expected["R_factor_all"], abs=5e-5)
    assert stats.weighted_r_factor("obs") == pytest.approx(expected["wR_factor_gt"], abs=5e-5)
    assert stats.weighted_r_factor("all") == pytest.approx(expected["wR_factor_ref"], abs=5e-5)
    gof_obs = stats.goodness_of_fit("obs", include_orientation=False)
    gof_all = stats.goodness_of_fit("all", include_orientation=False)
    assert gof_obs == pytest.approx(expected["goodness_of_fit_gt"], abs=5e-5)
    assert gof_all == pytest.approx(expected["goodness_of_fit_ref"], abs=5e-5)


def test_charging_the_per_frame_scales_does_not_reproduce_jana() -> None:
    """The scale exclusion is what the data says, not a free choice: counting them breaks parity."""
    expected = json.loads((JANA / "expected.json").read_text())["quartz"]
    frames, _ = _jana_frames("quartz")
    stats = fit_statistics(frames, ParameterCount(structural=int(expected["number_parameters"])))
    gof = stats.goodness_of_fit("obs", include_orientation=False)
    assert abs(gof - expected["goodness_of_fit_gt"]) > 1e-2


def test_frame_fit_is_independent_of_the_calculated_intensity_scale() -> None:
    generator = torch.Generator().manual_seed(0)
    calculated = torch.rand(50, generator=generator, dtype=torch.float64) * 100
    sigmas = torch.rand(50, generator=generator, dtype=torch.float64) * 5 + 1
    observed = 2.5 * calculated + sigmas * torch.randn(50, generator=generator, dtype=torch.float64)
    reference = frame_fit(calculated, observed, sigmas)
    rescaled = frame_fit(7.0 * calculated, observed, sigmas)
    assert rescaled.all_reflections.weighted_residual == pytest.approx(
        reference.all_reflections.weighted_residual, rel=1e-10
    )
    assert rescaled.amplitude_scale == pytest.approx(reference.amplitude_scale / math.sqrt(7.0))


def test_frames_pool_by_summing_not_by_averaging() -> None:
    frames, _ = _jana_frames("quartz")
    count = ParameterCount(structural=0)
    pooled = fit_statistics(frames[:2], count)
    first = frame_fit(*frames[0]).all_reflections
    second = frame_fit(*frames[1]).all_reflections
    assert pooled.all_reflections.n == first.n + second.n
    assert pooled.all_reflections.weighted_residual == pytest.approx(
        first.weighted_residual + second.weighted_residual
    )


def test_gof_is_nan_without_degrees_of_freedom() -> None:
    stats = fit_statistics([([1.0, 2.0], [1.0, 2.0], [0.1, 0.1])], ParameterCount(structural=2))
    assert math.isnan(stats.goodness_of_fit("all", include_orientation=False))


def test_parameter_count_charges_orientation_only_when_asked() -> None:
    count = ParameterCount(
        structural=14,
        components=5,
        per_frame_thickness=10,
        per_frame_orientation=30,
        per_frame_scales=10,
    )
    assert count.without_orientation == 29
    assert count.with_orientation == 59
    with pytest.raises(ValueError, match="structural"):
        ParameterCount(structural=-1)


def test_amplitude_sigma_floors_weak_reflections() -> None:
    """sigma(F) = sigma(I)/(2 sqrt(I)), floored at 5 sqrt(sigma(I)) for I < 0.01 sigma(I)."""
    observed = torch.tensor([400.0, 0.001, -3.0], dtype=torch.float64)
    sigmas = torch.tensor([20.0, 4.0, 4.0], dtype=torch.float64)
    expected = torch.tensor([20.0 / 40.0, 10.0, 10.0], dtype=torch.float64)
    assert torch.allclose(amplitude_sigma(observed, sigmas), expected)


def test_structural_count_removes_special_position_and_adp_symmetry() -> None:
    """Quartz (P3_221): Si on a 2-fold axis (1 xyz, 4 Uij) plus general O (3 xyz, 6 Uij) = 14.

    Jana2020 refines the same model with 15 = 14 + one thickness (its 126 parameters minus 111
    per-zone scales).
    """
    setup = RefinementSetup.from_structure(
        read_structure(FIXTURES / "quartz_anchor" / "enantiomer_1.cif")
    )
    both = structural_parameter_count(setup.params, setup.spec, TrainableSpec.positions_and_adp())
    positions = structural_parameter_count(
        setup.params, setup.spec, TrainableSpec(positions=AtomSelection.all())
    )
    assert (positions, both) == (4, 14)

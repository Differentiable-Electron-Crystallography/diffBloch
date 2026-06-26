"""Typed diffraction products and the calculated/observed alignment bridge (``core.products``)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from diffBloch.core.products import (
    BlochSolution,
    PatternBatch,
    align,
    build_alignment_plan,
    intensities,
)
from diffBloch.io.record import ObservationRecord


def _observation_record() -> ObservationRecord:
    # Two zone axes; reflections split across them so zone filtering is exercised.
    return ObservationRecord(
        unit_cell=np.eye(3),
        wavelength=0.0251,
        ub_matrix=np.eye(3),
        zone_axis_ids=np.asarray([1, 2]),
        zone_axes=np.zeros((2, 3)),
        precession_angles=np.asarray([1.0, 1.0]),
        alphas=np.zeros(2),
        betas=np.zeros(2),
        omegas=np.zeros(2),
        scales=np.ones(2),
        hkl=np.asarray([[1, 0, 0], [0, 1, 0], [2, 0, 0]], dtype=np.int64),
        intensities=np.asarray([10.0, 20.0, 30.0]),
        sigmas=np.asarray([1.0, 2.0, 3.0]),
        reflection_zone_axis_ids=np.asarray([1, 1, 2]),
    )


def test_bloch_solution_from_propagation() -> None:
    amplitudes = torch.tensor([[3 + 4j, 1 + 0j], [0 + 1j, 2 + 0j]], dtype=torch.complex128)
    beam_hkl = torch.tensor([[0, 0, 0], [1, 0, 0]])
    thicknesses = torch.tensor([1.0, 8.0])

    solution = BlochSolution.from_propagation(amplitudes, beam_hkl, thicknesses)
    assert torch.allclose(solution.intensities, intensities(amplitudes))
    assert solution.beam_hkl.dtype == torch.int64
    assert solution.intensities.shape == (2, 2)


@pytest.mark.parametrize(
    ("amplitudes", "beam_hkl", "thicknesses", "match"),
    [
        (torch.zeros(2, dtype=torch.complex128), [[0, 0, 0]], [1.0], "amplitudes must have shape"),
        (torch.zeros((1, 2), dtype=torch.complex128), [[0, 0, 0]], [1.0], "beam_hkl must have"),
        (torch.zeros((1, 1), dtype=torch.complex128), [[0, 0, 0]], [1.0, 2.0], "thicknesses must"),
    ],
)
def test_bloch_solution_rejects_bad_shapes(amplitudes, beam_hkl, thicknesses, match) -> None:
    with pytest.raises(ValueError, match=match):
        BlochSolution.from_propagation(
            amplitudes, torch.tensor(beam_hkl), torch.tensor(thicknesses)
        )


def test_pattern_batch_from_observation_record_full() -> None:
    pattern = PatternBatch.from_observation_record(_observation_record())
    assert pattern.hkl.shape == (3, 3)
    assert torch.equal(pattern.intensities, torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64))
    assert pattern.sigmas.dtype == torch.float64


def test_pattern_batch_zone_axis_filter() -> None:
    pattern = PatternBatch.from_observation_record(_observation_record(), zone_axis_id=1)
    assert pattern.hkl.shape == (2, 3)
    assert torch.equal(pattern.intensities, torch.tensor([10.0, 20.0], dtype=torch.float64))
    with pytest.raises(ValueError, match="no observed reflections for zone_axis_id 9"):
        PatternBatch.from_observation_record(_observation_record(), zone_axis_id=9)


def test_align_puts_calculated_and_observed_on_common_axis() -> None:
    # Calculated beams {000, 100, 010}; observed reflections {100, 010, 200}. Shared = {100, 010}
    # (in observed order); 000 (calc-only) and 200 (obs-only) are dropped.
    amplitudes = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=torch.complex128)
    beam_hkl = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    solution = BlochSolution.from_propagation(amplitudes, beam_hkl, torch.tensor([1.0, 8.0]))
    pattern = PatternBatch.from_observation_record(_observation_record())  # 100,010,200

    plan = build_alignment_plan(solution.beam_hkl, pattern.hkl)
    assert torch.equal(plan.hkl, torch.tensor([[1, 0, 0], [0, 1, 0]]))

    aligned = align(solution, pattern, plan)
    # calculated intensities are |2|^2=4 (100) and |3|^2=9 (010), broadcast over T=2 thicknesses
    f64 = torch.float64
    assert torch.allclose(aligned.calculated, torch.tensor([[4.0, 9.0], [4.0, 9.0]], dtype=f64))
    assert torch.allclose(aligned.observed, torch.tensor([[10.0, 20.0], [10.0, 20.0]], dtype=f64))
    assert torch.allclose(aligned.sigmas, torch.tensor([[1.0, 2.0], [1.0, 2.0]], dtype=f64))


def test_align_is_differentiable_back_to_amplitudes() -> None:
    amplitudes = torch.tensor([[1.0 + 0j, 2.0 + 0j]], dtype=torch.complex128, requires_grad=True)
    beam_hkl = torch.tensor([[1, 0, 0], [0, 1, 0]])
    solution = BlochSolution.from_propagation(amplitudes, beam_hkl, torch.tensor([1.0]))
    pattern = PatternBatch.from_observation_record(_observation_record())
    plan = build_alignment_plan(solution.beam_hkl, pattern.hkl)

    align(solution, pattern, plan).calculated.sum().backward()
    assert amplitudes.grad is not None and amplitudes.grad.abs().sum() > 0


def test_build_alignment_plan_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="solution_hkl must have shape"):
        build_alignment_plan(torch.tensor([1, 0, 0]), torch.tensor([[1, 0, 0]]))

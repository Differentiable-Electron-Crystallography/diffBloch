"""Typed diffraction products and the calculated/observed alignment bridge (``core.products``)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from diffBloch.core.products import (
    BlochSolution,
    MosaicSmoothed,
    PatternBatch,
    PlainSum,
    WeightedSum,
    align,
    build_alignment_plan,
    intensities,
)
from diffBloch.io.record import ExperimentalRecord


def _experimental_record() -> ExperimentalRecord:
    # Two zone axes; reflections split across them so zone filtering is exercised.
    return ExperimentalRecord(
        unit_cell=np.eye(3),
        cell_parameters=np.asarray([1.0, 1.0, 1.0, 90.0, 90.0, 90.0]),
        cell_parameters_su=np.full((6,), np.nan),
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


def test_integrate_sums_tilt_intensities_incoherently() -> None:
    beam_hkl = torch.tensor([[0, 0, 0], [1, 0, 0]])
    thicknesses = torch.tensor([1.0, 8.0])
    a = BlochSolution.from_propagation(
        torch.tensor([[3 + 4j, 1 + 0j], [0 + 1j, 2 + 0j]], dtype=torch.complex128),
        beam_hkl,
        thicknesses,
    )
    b = BlochSolution.from_propagation(
        torch.tensor([[0 + 2j, 2 + 0j], [1 + 0j, 0 + 1j]], dtype=torch.complex128),
        beam_hkl,
        thicknesses,
    )
    integrated = BlochSolution.integrate([a, b])
    # incoherent sum of |psi|^2, not of amplitudes.
    assert torch.allclose(integrated.intensities, a.intensities + b.intensities)
    # amplitudes is the real effective sqrt(total): the invariant |amplitudes|^2 == intensities.
    assert torch.allclose(integrated.amplitudes.abs().square(), integrated.intensities)
    assert torch.equal(integrated.beam_hkl, beam_hkl)
    assert torch.equal(integrated.thicknesses, thicknesses)


def test_integrate_rejects_empty_and_mismatched_beam_sets() -> None:
    hkl = torch.tensor([[0, 0, 0], [1, 0, 0]])
    thick = torch.tensor([1.0])
    a = BlochSolution.from_propagation(torch.ones((1, 2), dtype=torch.complex128), hkl, thick)
    other = BlochSolution.from_propagation(
        torch.ones((1, 2), dtype=torch.complex128), torch.tensor([[0, 0, 0], [2, 0, 0]]), thick
    )
    with pytest.raises(ValueError, match="at least one solution"):
        BlochSolution.integrate([])
    with pytest.raises(ValueError, match="share the same beam set"):
        BlochSolution.integrate([a, other])


def _ramp_tilts(values: list[float]) -> list[BlochSolution]:
    """One sub-solution per tilt, each a flat intensity ``v`` over a 1-thickness / 2-beam set."""
    hkl = torch.tensor([[0, 0, 0], [1, 0, 0]])
    thick = torch.tensor([100.0])
    return [
        BlochSolution.from_propagation(
            torch.full((1, 2), v, dtype=torch.float64).to(torch.complex128) ** 0.5, hkl, thick
        )
        for v in values
    ]


def test_integrate_default_reduction_is_the_plain_sum() -> None:
    sols = _ramp_tilts([1.0, 2.0, 3.0, 4.0, 5.0])
    # The default reduction (PlainSum) is the incoherent sum, identical to passing it explicitly.
    default = BlochSolution.integrate(sols)
    plain = BlochSolution.integrate(sols, reduction=PlainSum())
    assert torch.allclose(default.intensities, torch.full((1, 2), 15.0, dtype=torch.float64))
    assert torch.allclose(default.intensities, plain.intensities)


def test_integrate_mosaic_smoothed_is_moving_average_then_sum() -> None:
    sols = _ramp_tilts([1.0, 2.0, 3.0, 4.0, 5.0])
    # Three-sample moving average then sum == sum of sample means:
    # mean(1,2,3)+mean(2,3,4)+mean(3,4,5)
    # = 2 + 3 + 4 = 9 (zero-padding the smoothed curve does not change the sum).
    mosaic = BlochSolution.integrate(sols, reduction=MosaicSmoothed(3))
    assert torch.allclose(mosaic.intensities, torch.full((1, 2), 9.0, dtype=torch.float64))
    # A one-sample span is the identity: each mean is the tilt itself, so it equals the plain sum.
    identity = BlochSolution.integrate(sols, reduction=MosaicSmoothed(1))
    assert torch.allclose(identity.intensities, torch.full((1, 2), 15.0, dtype=torch.float64))
    assert torch.allclose(mosaic.amplitudes.abs().square(), mosaic.intensities)


def test_integrate_mosaic_sample_span_may_not_exceed_the_tilt_count() -> None:
    sols = _ramp_tilts([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="sample span 4 exceeds the 3 rocking-curve tilts"):
        BlochSolution.integrate(sols, reduction=MosaicSmoothed(4))


def test_weighted_sum_rejects_non_1d_weights() -> None:
    with pytest.raises(ValueError, match="weights must be 1-D"):
        WeightedSum(weights=torch.ones((2, 2)))


def test_integrate_weighted_sum_scales_each_tilt_before_summing() -> None:
    sols = _ramp_tilts([1.0, 2.0, 3.0])
    # Each tilt v is weighted by w, so the sum is dot([1,2,3], [1,0,2]) = 1 + 0 + 6 = 7.
    weighted = BlochSolution.integrate(
        sols, reduction=WeightedSum(weights=torch.tensor([1.0, 0.0, 2.0]))
    )
    assert torch.allclose(weighted.intensities, torch.full((1, 2), 7.0, dtype=torch.float64))
    assert torch.allclose(weighted.amplitudes.abs().square(), weighted.intensities)
    # Uniform unit weights reduce to the plain sum.
    uniform = BlochSolution.integrate(sols, reduction=WeightedSum(weights=torch.ones(3)))
    plain = BlochSolution.integrate(sols, reduction=PlainSum())
    assert torch.allclose(uniform.intensities, plain.intensities)


def test_integrate_weighted_sum_rejects_mismatched_weight_count() -> None:
    sols = _ramp_tilts([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="WeightedSum weights has 2 entries, expected 3"):
        BlochSolution.integrate(sols, reduction=WeightedSum(weights=torch.tensor([1.0, 2.0])))


def test_integrate_batched_matches_the_per_tilt_integrate() -> None:
    # integrate_batched takes the stacked (B, T, N) amplitudes a batched solve returns; it must
    # equal integrate over the corresponding per-tilt sub-solutions (identical stack + reduction).
    beam_hkl = torch.tensor([[0, 0, 0], [1, 0, 0]])
    thick = torch.tensor([10.0, 40.0])
    amps = [
        torch.tensor([[3 + 4j, 1 + 0j], [0 + 1j, 2 + 0j]], dtype=torch.complex128),
        torch.tensor([[0 + 2j, 2 + 0j], [1 + 0j, 0 + 1j]], dtype=torch.complex128),
        torch.tensor([[1 + 1j, 0 + 3j], [2 + 0j, 1 + 1j]], dtype=torch.complex128),
    ]
    sols = [BlochSolution.from_propagation(a, beam_hkl, thick) for a in amps]
    stacked = torch.stack(amps)  # (B, T, N)
    for reduction in (PlainSum(), MosaicSmoothed(2)):
        looped = BlochSolution.integrate(sols, reduction=reduction)
        batched = BlochSolution.integrate_batched(stacked, beam_hkl, thick, reduction=reduction)
        assert torch.equal(batched.intensities, looped.intensities)
        assert torch.equal(batched.amplitudes, looped.amplitudes)
        assert torch.equal(batched.beam_hkl, looped.beam_hkl)


def test_integrate_batched_rejects_non_3d_amplitudes() -> None:
    beam_hkl = torch.tensor([[0, 0, 0], [1, 0, 0]])
    thick = torch.tensor([10.0])
    with pytest.raises(ValueError, match=r"shape \(N_tilts, T, N\)"):
        BlochSolution.integrate_batched(torch.ones((2, 2), dtype=torch.complex128), beam_hkl, thick)


def test_pattern_batch_from_observation_record_full() -> None:
    pattern = PatternBatch.from_experimental_record(_experimental_record())
    assert pattern.hkl.shape == (3, 3)
    assert torch.equal(pattern.intensities, torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64))
    assert pattern.sigmas.dtype == torch.float64


def test_pattern_batch_zone_axis_filter() -> None:
    pattern = PatternBatch.from_experimental_record(_experimental_record(), zone_axis_id=1)
    assert pattern.hkl.shape == (2, 3)
    assert torch.equal(pattern.intensities, torch.tensor([10.0, 20.0], dtype=torch.float64))
    with pytest.raises(ValueError, match="no observed reflections for zone_axis_id 9"):
        PatternBatch.from_experimental_record(_experimental_record(), zone_axis_id=9)


def test_align_puts_calculated_and_observed_on_common_axis() -> None:
    # Calculated beams {000, 100, 010}; observed reflections {100, 010, 200}. Shared = {100, 010}
    # (in observed order); 000 (calc-only) and 200 (obs-only) are dropped.
    amplitudes = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=torch.complex128)
    beam_hkl = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    solution = BlochSolution.from_propagation(amplitudes, beam_hkl, torch.tensor([1.0, 8.0]))
    pattern = PatternBatch.from_experimental_record(_experimental_record())  # 100,010,200

    plan = build_alignment_plan(solution.beam_hkl, pattern.hkl)
    assert torch.equal(plan.hkl, torch.tensor([[1, 0, 0], [0, 1, 0]]))

    aligned = align(solution, pattern, plan)
    # all three outputs must be co-located (on calculated.device) for downstream losses
    assert aligned.observed.device == aligned.calculated.device
    assert aligned.sigmas.device == aligned.calculated.device
    # calculated intensities are |2|^2=4 (100) and |3|^2=9 (010), broadcast over T=2 thicknesses
    f64 = torch.float64
    assert torch.allclose(aligned.calculated, torch.tensor([[4.0, 9.0], [4.0, 9.0]], dtype=f64))
    assert torch.allclose(aligned.observed, torch.tensor([[10.0, 20.0], [10.0, 20.0]], dtype=f64))
    assert torch.allclose(aligned.sigmas, torch.tensor([[1.0, 2.0], [1.0, 2.0]], dtype=f64))


def test_align_is_differentiable_back_to_amplitudes() -> None:
    amplitudes = torch.tensor([[1.0 + 0j, 2.0 + 0j]], dtype=torch.complex128, requires_grad=True)
    beam_hkl = torch.tensor([[1, 0, 0], [0, 1, 0]])
    solution = BlochSolution.from_propagation(amplitudes, beam_hkl, torch.tensor([1.0]))
    pattern = PatternBatch.from_experimental_record(_experimental_record())
    plan = build_alignment_plan(solution.beam_hkl, pattern.hkl)

    align(solution, pattern, plan).calculated.sum().backward()
    assert amplitudes.grad is not None and amplitudes.grad.abs().sum() > 0


def test_build_alignment_plan_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="solution_hkl must have shape"):
        build_alignment_plan(torch.tensor([1, 0, 0]), torch.tensor([[1, 0, 0]]))

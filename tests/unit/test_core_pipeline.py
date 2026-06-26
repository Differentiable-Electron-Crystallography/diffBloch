"""End-to-end calculated pipeline on the synthetic-Friedel oracle.

Ties the stage-8/9 pieces into one chain through the *public* API --
``build_bloch_system -> propagate -> BlochSolution -> align -> losses`` -- and pins it against the
``diffBloch_private`` golden one step at a time, so a regression localises to the offending step
rather than only the endpoint:

    Fgb --structure_matrix--> A --propagate--> psi --|.|^2--> intensities --align--> loss

The golden ``Fgb`` is synthetic (Friedel-symmetric so ``A`` is Hermitian; see the fixture
``provenance.json``); this proves the *wiring, observable, and differentiability* of the chain. The
physically-real R-factor pin against a CIF dataset is the deferred e2e quartz anchor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from diffBloch.core.dynamical import build_beam_plan, build_bloch_system
from diffBloch.core.losses import mse
from diffBloch.core.products import (
    BlochSolution,
    PatternBatch,
    align,
    build_alignment_plan,
    intensities,
)
from diffBloch.core.solver import propagate

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "structure_matrix_oracle"
_ORACLE_ZONE = _FIXTURES / "structure_matrix_oracle.npz"  # zone axis: Mii == 1, methods coincide


def _load(npz: Path):
    data = np.load(npz)
    plan = build_beam_plan(
        data["beam_hkl"],
        data["grid_hkl"],
        data["reciprocal_basis"],
        energy=float(data["energy"]),
        gpts=tuple(int(point) for point in data["gpts"]),
        u0=float(data["u0"]),
    )
    return plan, data


def test_pipeline_steps_match_private_golden() -> None:
    # Composite-step breakdown: A, psi, then the intensity observable -- each vs the private golden.
    plan, data = _load(_ORACLE_ZONE)
    fgb = torch.tensor(data["structure_factor"])
    system = build_bloch_system(plan, fgb)
    thicknesses = torch.tensor(data["thicknesses"])

    # Step 1 -- structure matrix A.
    assert torch.allclose(system.a, torch.tensor(data["A"]), rtol=1e-10, atol=1e-12)

    # Step 2 -- propagated exit wavefunction psi.
    psi = propagate(system, thicknesses, method="matrix_exp")
    assert torch.allclose(psi, torch.tensor(data["psi_matrix_exp"]), rtol=1e-10, atol=1e-12)

    # Step 3 -- the intensity observable |psi|^2, via the public BlochSolution product.
    solution = BlochSolution.from_propagation(psi, torch.tensor(data["beam_hkl"]), thicknesses)
    golden_intensities = intensities(torch.tensor(data["psi_matrix_exp"]))
    assert torch.allclose(solution.intensities, golden_intensities, rtol=1e-10, atol=1e-12)


def test_pipeline_align_and_loss_compose() -> None:
    # The full public path: take the golden intensities at the first thickness as a self-consistent
    # observed pattern, align with the calculated solution, and check the loss behaves sensibly --
    # zero at the matching thickness, positive elsewhere.
    plan, data = _load(_ORACLE_ZONE)
    system = build_bloch_system(plan, torch.tensor(data["structure_factor"]))
    thicknesses = torch.tensor(data["thicknesses"])
    beam_hkl = torch.tensor(data["beam_hkl"])

    psi = propagate(system, thicknesses, method="matrix_exp")
    solution = BlochSolution.from_propagation(psi, beam_hkl, thicknesses)

    reference = 0  # observed == calculated intensities at the first thickness
    pattern = PatternBatch(
        hkl=beam_hkl,
        intensities=solution.intensities[reference].detach(),
        sigmas=torch.full((beam_hkl.shape[0],), 0.01, dtype=torch.float64),
    )
    plan_align = build_alignment_plan(solution.beam_hkl, pattern.hkl)
    assert torch.equal(plan_align.hkl, beam_hkl)  # zone fixture: every beam is observed

    aligned = align(solution, pattern, plan_align)
    per_thickness = mse(aligned.calculated, aligned.observed)  # (T,)
    zero = torch.zeros((), dtype=torch.float64)
    assert torch.allclose(per_thickness[reference], zero, atol=1e-12)
    others = torch.cat([per_thickness[:reference], per_thickness[reference + 1 :]])
    assert torch.all(others > 0)


def test_pipeline_is_differentiable_end_to_end_in_fgb() -> None:
    # The headline guarantee for a differentiable refiner: a gradient flows the whole way back to
    # the structure factors, Fgb -> A -> propagate -> |psi|^2 -> align -> loss.
    plan, data = _load(_ORACLE_ZONE)
    thicknesses = torch.tensor(data["thicknesses"])
    beam_hkl = torch.tensor(data["beam_hkl"])

    fgb = torch.tensor(data["structure_factor"], requires_grad=True)
    system = build_bloch_system(plan, fgb)
    psi = propagate(system, thicknesses, method="matrix_exp")
    solution = BlochSolution.from_propagation(psi, beam_hkl, thicknesses)

    # A fixed (detached) target perturbed from the calculated intensities, so the loss is non-zero.
    target = (solution.intensities[-1].detach() + 0.05).clamp(min=0.0)
    pattern = PatternBatch(
        hkl=beam_hkl,
        intensities=target,
        sigmas=torch.full((beam_hkl.shape[0],), 0.01, dtype=torch.float64),
    )
    plan_align = build_alignment_plan(solution.beam_hkl, pattern.hkl)
    aligned = align(solution, pattern, plan_align)

    mse(aligned.calculated, aligned.observed).sum().backward()
    assert fgb.grad is not None
    assert torch.isfinite(fgb.grad).all()
    assert fgb.grad.abs().sum() > 0

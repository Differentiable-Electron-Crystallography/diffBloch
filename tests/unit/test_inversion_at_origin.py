"""``has_inversion_at_origin`` and the engine's ``real_operator`` decision built on it."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from tests.unit.test_inference import _METHOD, _orientation, _params, _refinement, _silicon

from diffBloch.core import build_asu_expansion_plan, has_inversion_at_origin
from diffBloch.core.symmetry import AsuExpansionPlan
from diffBloch.preprocess.plan import Plan
from diffBloch.preprocess.scoring import build_engine
from diffBloch.specs import Absorption

_EYE = torch.eye(3)


def _plan(rotations: list[torch.Tensor], translations: list[list[float]]) -> AsuExpansionPlan:
    return AsuExpansionPlan(
        asu_indices=torch.zeros(1, dtype=torch.long),
        symop_indices=torch.zeros(1, dtype=torch.long),
        rotations=torch.stack(rotations).to(torch.float64),
        translations=torch.tensor(translations, dtype=torch.float64),
        n_asu_sites=1,
    )


def test_inversion_through_the_origin_counts_up_to_a_lattice_shift() -> None:
    assert has_inversion_at_origin(_plan([_EYE, -_EYE], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    assert has_inversion_at_origin(_plan([_EYE, -_EYE], [[0.0, 0.0, 0.0], [1.0, 0.0, -1.0]]))


def test_inversion_off_the_origin_or_absent_does_not_count() -> None:
    assert not has_inversion_at_origin(_plan([_EYE, -_EYE], [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]))
    assert not has_inversion_at_origin(_plan([_EYE], [[0.0, 0.0, 0.0]]))
    # A two-fold rotation is not an inversion, even through the origin.
    two_fold = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))
    assert not has_inversion_at_origin(_plan([_EYE, two_fold], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))


def _centrosymmetric_engines(**kwargs: object) -> tuple[object, object]:
    # One silicon-like site at a general position, expanded through {1, -1} about the origin.
    grid, _, spec, numbers = _silicon()
    centro = build_asu_expansion_plan(
        np.array([[0.11, 0.23, 0.37]]), np.stack([np.eye(3), -np.eye(3)]), np.zeros((2, 3))
    )
    refinement = replace(
        _refinement(centro, spec, numbers),
        params=replace(_params(), asu_positions=torch.tensor([[0.11, 0.23, 0.37]])),
    )
    plan = Plan(structure_factor_grid=grid, orientations=(_orientation(grid, torch.ones(3), 0.1),))
    return build_engine(plan, refinement, method=_METHOD, **kwargs), refinement  # type: ignore[arg-type]


def test_build_engine_takes_the_real_path_only_with_inversion_and_no_absorption() -> None:
    engine, _ = _centrosymmetric_engines()
    assert engine.real_operator is True  # type: ignore[attr-defined]
    absorbing, _ = _centrosymmetric_engines(absorption=Absorption(enabled=True))
    assert absorbing.real_operator is False  # type: ignore[attr-defined]

    grid, asu_plan, spec, numbers = _silicon()  # identity only: no inversion
    plan = Plan(structure_factor_grid=grid, orientations=(_orientation(grid, torch.ones(3), 0.1),))
    no_inversion = build_engine(plan, _refinement(asu_plan, spec, numbers), method=_METHOD)
    assert no_inversion.real_operator is False


def test_real_and_complex_engines_agree_on_a_centrosymmetric_structure() -> None:
    engine, refinement = _centrosymmetric_engines()
    complex_engine = replace(engine, real_operator=False)  # type: ignore[type-var]
    (real,) = engine.simulate(refinement.params)  # type: ignore[attr-defined]
    (direct,) = complex_engine.simulate(refinement.params)
    gap = (real.intensities - direct.intensities).abs().max() / direct.intensities.abs().max()
    assert float(gap) < 1e-5

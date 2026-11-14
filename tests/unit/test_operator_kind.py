"""``operator_kind``: what ``A`` is, decided once from the symmetry and absorption."""

from __future__ import annotations

import torch

from diffBloch.core.symmetry import AsuExpansionPlan
from diffBloch.preprocess.scoring import operator_kind
from diffBloch.specs import Absorption


def _plan(rotations: list[torch.Tensor], translations: list[list[float]]) -> AsuExpansionPlan:
    return AsuExpansionPlan(
        asu_indices=torch.zeros(1, dtype=torch.long),
        symop_indices=torch.zeros(1, dtype=torch.long),
        rotations=torch.stack(rotations).to(torch.float64),
        translations=torch.tensor(translations, dtype=torch.float64),
        n_asu_sites=1,
    )


_EYE = torch.eye(3)


def test_inversion_through_the_origin_is_real() -> None:
    plan = _plan([_EYE, -_EYE], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # a lattice shift is origin
    assert operator_kind(plan, Absorption()) == "real"


def test_inversion_off_the_origin_or_absent_is_hermitian() -> None:
    assert (
        operator_kind(_plan([_EYE, -_EYE], [[0, 0, 0], [0.5, 0, 0]]), Absorption()) == "hermitian"
    )
    assert operator_kind(_plan([_EYE], [[0, 0, 0]]), Absorption()) == "hermitian"


def test_absorption_is_general() -> None:
    plan = _plan([_EYE, -_EYE], [[0, 0, 0], [0, 0, 0]])
    assert operator_kind(plan, Absorption(enabled=True)) == "general"

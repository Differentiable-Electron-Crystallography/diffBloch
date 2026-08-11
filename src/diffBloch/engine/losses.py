"""Named ``LossFn``/``ScoresFn`` builders: the objective terms ``ExperimentConfig.objective`` picks.

Each adapts a pure :mod:`diffBloch.core.losses` intensity comparison (which reduces over the
reflection axis, yielding a ``(T,)`` per-thickness loss) into two shapes: a ``ScoresFn``
(``AlignedIntensities -> (T,) Tensor``, one score per thickness) and the corresponding ``LossFn``
(``AlignedIntensities -> scalar``, the ``ScoresFn`` summed over thickness). ``RefinementEngine``
uses both off the *same* underlying metric -- ``scores`` for the orientation/thickness
preprocessing search (``score_orientation``/``score_orientation_per_thickness``, which need a
per-thickness vector to argmin over) and ``loss`` for the differentiable gradient-refinement
objective -- so picking one ``ExperimentConfig.objective`` genuinely drives the whole pipeline, not
just the gradient stage. Saves every caller from rewriting
``lambda a: mse(a.calculated, a.observed).sum()`` and gives each objective a named, importable home
-- e.g. ``RefinementEngine(loss=rbragg_loss, scores=robs_scores, ...)``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from diffBloch.core.losses import l1, mse, optimal_scale, rbragg, w_rbragg
from diffBloch.core.products import AlignedIntensities

__all__ = [
    "l1_loss",
    "mse_loss",
    "rbragg_loss",
    "robs_scores",
    "wr2_loss",
    "wr2_scores",
    "w_rbragg_loss",
]


def mse_loss(aligned: AlignedIntensities) -> Tensor:
    """Per-orientation MSE loss term, summed over thicknesses to a scalar."""
    return mse(aligned.calculated, aligned.observed).sum()


def l1_loss(aligned: AlignedIntensities) -> Tensor:
    """Per-orientation L1 loss term, summed over thicknesses to a scalar."""
    return l1(aligned.calculated, aligned.observed).sum()


def w_rbragg_loss(aligned: AlignedIntensities) -> Tensor:
    """Per-orientation weighted-R2 term (default ``mu``), summed over thicknesses to a scalar.

    Raw: no calc<->obs scaling. Correct only where the caller has already put calculated on the
    observed scale; for the refinement objective use :func:`wr2_loss`, which is the
    :func:`~diffBloch.preprocess.scoring.build_engine` default.
    """
    return w_rbragg(aligned.calculated, aligned.observed, aligned.sigmas).sum()


def wr2_scores(aligned: AlignedIntensities) -> Tensor:
    """Per-thickness scaling-optimised weighted-R2 (shape ``(T,)``).

    The calculated intensities come off the dynamical solve on an arbitrary structure-factor scale,
    while the observed are PETS intensities on their own scale. Compared raw
    (:func:`w_rbragg_loss`), wR2 is denominator-dominated and parks near ~1 with a vanishing
    gradient, so a gradient refinement cannot descend it. Every call therefore re-fits the
    multiplicative intensity scale independently for every thickness through
    :func:`~diffBloch.core.losses.optimal_scale`. The selected grid branch remains differentiable
    in its calculated intensities (``torch.min`` routes the gradient through the winning
    candidate); only a boundary where the winning grid point changes is piecewise-smooth.
    """
    calc, obs = aligned.calculated, aligned.observed
    return torch.stack(
        [optimal_scale(calc[t], obs[t], aligned.sigmas[t])[1] for t in range(calc.shape[0])]
    )


def wr2_loss(aligned: AlignedIntensities) -> Tensor:
    """Scaling-optimised weighted-R2 -- the default refinement and orientation-search objective.

    Sums :func:`wr2_scores` over the thickness axis to a scalar; see there for the scale-fit.
    """
    return wr2_scores(aligned).sum()


def robs_scores(aligned: AlignedIntensities) -> Tensor:
    """Per-thickness scaling-optimised Bragg R(obs) (shape ``(T,)``).

    Refits the intensity scale per thickness exactly like :func:`wr2_scores`, but against the
    :func:`~diffBloch.core.losses.rbragg` metric instead of the default ``w_rbragg`` -- the same
    R_obs the app reports elsewhere (:meth:`~diffBloch.engine.forward.RefinementEngine.refinement_metrics`).
    """
    calc, obs = aligned.calculated, aligned.observed
    return torch.stack(
        [
            optimal_scale(calc[t], obs[t], aligned.sigmas[t], metric=rbragg)[1]
            for t in range(calc.shape[0])
        ]
    )


def rbragg_loss(aligned: AlignedIntensities) -> Tensor:
    """Scaling-optimised Bragg R(obs) objective. Sums :func:`robs_scores` over thickness."""
    return robs_scores(aligned).sum()

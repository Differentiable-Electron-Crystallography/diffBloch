"""Slice 11: the coverage sweep -- ``plan_coverage``, ``maximize_scalar``, ``cover_beams`` /
``cover_pool``.

The second convergence operation (match-count objective, distinct from ``convergence.py``'s
self-stability). Reuses the fast synthetic system from ``tests.unit.synthetic`` (an hkl cube whose
observed pattern is the full cube, so coverage tracks the selected-beam count as a beam knob
widens). ``plan_coverage`` is pinned on a partial-overlap plan; ``maximize_scalar`` on a scripted
objective; the adapters end-to-end.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from tests.unit.synthetic import beam_count, seed_system

from diffBloch.core.products import PatternBatch
from diffBloch.preprocess import (
    cover_beams,
    cover_pool,
    maximize_scalar,
    plan_coverage,
    select_beams,
)
from diffBloch.preprocess.plan import Plan
from diffBloch.specs import BeamSelection


def _coverage(plan: Plan, semiangle: float) -> int:
    return plan_coverage(select_beams(BeamSelection(integration_semiangle=semiangle))(plan))


# --- plan_coverage: the pure match-count objective ---


def test_plan_coverage_counts_the_beam_observed_intersection() -> None:
    _, seed = seed_system()
    op = seed.orientations[0]
    beam_hkls = {tuple(int(x) for x in hkl) for hkl in np.asarray(op.beam_hkl)}
    in_beams = [next(iter(beam_hkls)), *list(beam_hkls)[1:3]]  # 3 hkls that ARE beams
    not_beams = [(50, 50, 50), (60, 60, 60)]  # 2 hkls that are not
    observed = torch.tensor([*in_beams, *not_beams], dtype=torch.int64)
    partial = replace(
        op,
        pattern=PatternBatch(
            hkl=observed,
            intensities=torch.zeros(len(observed), dtype=torch.float64),
            sigmas=torch.ones(len(observed), dtype=torch.float64),
        ),
    )
    # Only the 3 in-beam observed reflections are matched; the 2 outsiders are not.
    assert plan_coverage(replace(seed, orientations=(partial,))) == 3


# --- maximize_scalar: the grow-while-improving driver, scripted ---


def test_maximize_scalar_stops_at_first_non_increase() -> None:
    # Score improves for two clicks then plateaus; the sweep returns the last improving build.
    scores = {1.0: 5, 1.1: 7, 1.2: 9, 1.3: 9}
    result = maximize_scalar(lambda v: v, lambda v: scores[round(v, 1)], start=1.0, step=0.1)
    assert result == pytest.approx(1.2)


def test_maximize_scalar_raises_when_score_never_plateaus() -> None:
    with pytest.raises(RuntimeError, match="did not plateau within 3 steps"):
        maximize_scalar(lambda v: v, lambda v: v, start=1.0, step=0.1, max_iterations=3)


def test_maximize_scalar_rejects_bad_max_iterations() -> None:
    with pytest.raises(ValueError, match="max_iterations must be >= 1"):
        maximize_scalar(lambda v: v, lambda v: v, start=1.0, step=0.1, max_iterations=0)


# --- cover_beams / cover_pool: the two beam levers, end-to-end ---


def test_cover_beams_widens_to_the_minimum_that_maximises_coverage() -> None:
    _, seed = seed_system()
    step = cover_beams(BeamSelection(integration_semiangle=0.68), step=0.2, max_iterations=30)
    converged = step(seed)

    started = _coverage(seed, 0.68)
    saturated = _coverage(seed, 5.0)
    # Widened past the start, stopping at the first window that buys no new match (short of the
    # fully-saturated set -- the minimal-sufficient coverage corner).
    assert started < plan_coverage(converged) < saturated


def test_cover_pool_widens_the_seed_to_maximise_coverage() -> None:
    from diffBloch.engine.plan import OrientationPlan
    from diffBloch.preprocess.experiment import seed_beam_hkl

    _, seed = seed_system()

    def reseed_coverage(g_max_refine: float) -> int:
        beam_hkl = seed_beam_hkl(seed.grid, g_max_refine=g_max_refine)
        reseeded = tuple(
            OrientationPlan.build(
                seed.grid,
                beam_hkl,
                op.pattern,
                energy=op.energy,
                thickness=op.thickness,
                u0=op.u0,
                orientation=op.orientation,
            )
            for op in seed.orientations
        )
        return _coverage(replace(seed, orientations=reseeded), 1.0)

    step = cover_pool(
        BeamSelection(integration_semiangle=1.0),
        start_g_max_refine=0.5,
        step=0.1,
        max_iterations=30,
    )
    converged = step(seed)
    # Widened the pool past the start; observed is a superset so coverage == active beam count.
    assert reseed_coverage(0.5) < plan_coverage(converged)
    assert plan_coverage(converged) == beam_count(converged)


def test_cover_pool_raises_past_the_grid_difference_support() -> None:
    _, seed = seed_system()
    step = cover_pool(
        BeamSelection(integration_semiangle=1.0),
        start_g_max_refine=1.0,
        step=0.2,
        max_iterations=50,
    )
    with pytest.raises(ValueError, match="exceeds the grid's beam-difference support"):
        step(seed)


def test_cover_beams_rejects_non_positive_step() -> None:
    with pytest.raises(ValueError, match="step must be positive"):
        cover_beams(BeamSelection(), step=0.0)


def test_cover_pool_rejects_non_positive_step() -> None:
    with pytest.raises(ValueError, match="step must be positive"):
        cover_pool(BeamSelection(), start_g_max_refine=0.5, step=0.0)

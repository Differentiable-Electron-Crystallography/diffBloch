"""``score_integrated_reflections`` and its report helpers on the first quartz-anchor frames."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import pytest

from diffBloch.config import load_config
from diffBloch.io import read_experimental_data, read_structure
from diffBloch.preprocess import from_experiment
from diffBloch.preprocess.plan import require_built_plans
from diffBloch.preprocess.serialize import read_plan
from diffBloch.preprocess.steps.integrated_scoring import (
    rocking_curve_report,
    score_integrated_reflections,
)
from diffBloch.specs import IntegratedScoring

QUARTZ = Path(__file__).parent.parent / "fixtures" / "quartz_anchor"
_N_FRAMES = 3


@lru_cache(maxsize=1)
def _quartz():
    config = load_config(QUARTZ / "experiment.yaml")
    setup = from_experiment(
        read_structure(QUARTZ / "enantiomer_1.cif"),
        read_experimental_data(QUARTZ / "exp_data.cif_pets"),
        config,
    )
    plan = read_plan(QUARTZ / "reproducibility" / "plan.exp_data.npz")
    plan = replace(plan, orientations=plan.orientations[:_N_FRAMES])
    simulation = config.blochwave.to_policy()
    common = {
        "rocking": config.blochwave.to_rocking_curve(setup.integration),
        "solve_g_max": simulation.g_max,
        "sg_max": simulation.sg_max,
    }
    return setup.refinement, plan, common


def _measurements(op):
    hkl = op.alignment.hkl.cpu().numpy()
    i_exp = op.pattern.intensities[op.alignment.pattern_index].cpu().numpy()
    return {(*map(int, h), float(i)) for h, i in zip(hkl, i_exp, strict=True)}


def test_scores_each_measurement_once_from_the_existing_scored_set() -> None:
    refinement, plan, common = _quartz()
    out = score_integrated_reflections(refinement, IntegratedScoring(), **common)(plan)

    before = require_built_plans(plan)
    after = require_built_plans(out)
    assert len(after) == len(before)
    seen: set = set()
    for old, new in zip(before, after, strict=True):
        rows = _measurements(new)
        assert rows <= _measurements(old)
        assert not rows & seen
        seen |= rows
    assert seen


def test_stricter_min_inside_scores_no_more() -> None:
    refinement, plan, common = _quartz()
    loose = score_integrated_reflections(refinement, IntegratedScoring(min_inside=0.0), **common)
    strict = score_integrated_reflections(refinement, IntegratedScoring(min_inside=1.0), **common)
    n = lambda p: sum(int(op.alignment.hkl.shape[0]) for op in require_built_plans(p))  # noqa: E731
    assert n(strict(plan)) <= n(loose(plan))


def test_rocking_curve_report_first_pair_and_all_pairs() -> None:
    refinement, plan, common = _quartz()
    (first,) = rocking_curve_report(plan, refinement, **common)
    assert len(first.rotation_indices) == 2
    events = rocking_curve_report(plan, refinement, which="all", **common)
    assert len(events) >= _N_FRAMES - 1
    with pytest.raises(ValueError, match="max_pairs"):
        rocking_curve_report(plan, refinement, which="all", max_pairs=0, **common)
    single = replace(plan, orientations=plan.orientations[:1])
    assert rocking_curve_report(single, refinement, **common) == ()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"fraction": 1.0}, "fraction"),
        ({"min_inside": 1.5}, "min_inside"),
        ({"width_rotations": 0}, "width_rotations"),
        ({"sweep_factor": 0.5}, "sweep_factor"),
    ],
)
def test_integrated_scoring_rejects_invalid_values(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        IntegratedScoring(**kwargs)

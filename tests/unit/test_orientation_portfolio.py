"""Unit coverage for the opt-in orientation-optimizer portfolio step."""

from __future__ import annotations

from math import nan
from types import SimpleNamespace

import pytest

from diffBloch.preprocess.pipeline import spec_to_params, step_records
from diffBloch.preprocess.plan import Plan
from diffBloch.preprocess.steps.orientation_portfolio import _select_by_wr2, _VariantResult
from diffBloch.specs import NelderMeadSearch, OrientationPortfolioSearch


def _plan(*tokens: object) -> Plan:
    return Plan(structure_factor_grid=None, orientations=tokens)  # type: ignore[arg-type]


def test_orientation_portfolio_search_defaults_are_labelled_ben_optimizer_variants() -> None:
    search = OrientationPortfolioSearch()

    assert [name for name, _variant in search.variants] == [
        "default_step_0.05_penalty",
        "fine_step_0.01_penalty",
        "finer_step_0.005_penalty",
    ]
    assert search.variants[0][1] == NelderMeadSearch()
    assert all(variant.penalize_fewer_reflections for _name, variant in search.variants)


def test_orientation_portfolio_search_rejects_ambiguous_variant_labels() -> None:
    duplicate = (
        ("same", NelderMeadSearch()),
        ("same", NelderMeadSearch(step_size=0.01)),
    )

    with pytest.raises(ValueError, match="variant names must be unique"):
        OrientationPortfolioSearch(variants=duplicate)


def test_orientation_portfolio_search_requires_matched_count_penalty() -> None:
    with pytest.raises(ValueError, match="penalize_fewer_reflections=True"):
        OrientationPortfolioSearch(
            variants=(("raw", NelderMeadSearch(penalize_fewer_reflections=False)),)
        )


def test_orientation_portfolio_provenance_records_every_variant() -> None:
    params = spec_to_params(
        {
            "search": OrientationPortfolioSearch(
                variants=(
                    ("default", NelderMeadSearch()),
                    ("fine", NelderMeadSearch(step_size=0.01)),
                )
            )
        }
    )

    assert params == {
        "search": {
            "__type__": "OrientationPortfolioSearch",
            "variants": [
                [
                    "default",
                    {
                        "__type__": "NelderMeadSearch",
                        "step_size": 0.05,
                        "max_iterations": 60,
                        "x_tolerance": 0.001,
                        "f_tolerance": 0.001,
                        "penalize_fewer_reflections": True,
                    },
                ],
                [
                    "fine",
                    {
                        "__type__": "NelderMeadSearch",
                        "step_size": 0.01,
                        "max_iterations": 60,
                        "x_tolerance": 0.001,
                        "f_tolerance": 0.001,
                        "penalize_fewer_reflections": True,
                    },
                ],
            ],
            "selector": "lowest_terminal_wr2",
        }
    }


def test_select_by_wr2_chooses_independently_per_orientation() -> None:
    a0 = SimpleNamespace(name="a0")
    a1 = SimpleNamespace(name="a1")
    b0 = SimpleNamespace(name="b0")
    b1 = SimpleNamespace(name="b1")
    c0 = SimpleNamespace(name="c0")
    c1 = SimpleNamespace(name="c1")

    selected = _select_by_wr2(
        (
            _VariantResult("a", _plan(a0, a1), (0.10, 0.40)),
            _VariantResult("b", _plan(b0, b1), (0.20, 0.20)),
            _VariantResult("c", _plan(c0, c1), (0.30, 0.30)),
        )
    )

    assert selected == (a0, b1)


def test_select_by_wr2_rejects_mismatched_plan_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        _select_by_wr2(
            (
                _VariantResult("a", _plan(object()), (0.1,)),
                _VariantResult("b", _plan(object(), object()), (0.1, 0.2)),
            )
        )


def test_select_by_wr2_prefers_finite_scores_and_falls_back_deterministically() -> None:
    finite = SimpleNamespace(name="finite")
    nan_first = SimpleNamespace(name="nan_first")
    nan_second = SimpleNamespace(name="nan_second")

    selected = _select_by_wr2(
        (
            _VariantResult("nan_first", _plan(nan_first, nan_first), (nan, nan)),
            _VariantResult("finite", _plan(finite, nan_second), (0.5, nan)),
            _VariantResult("nan_second", _plan(nan_second, nan_second), (nan, nan)),
        )
    )

    assert selected == (finite, nan_first)


def test_select_orientation_portfolio_step_record_names_the_selector() -> None:
    from tests.unit.synthetic import seed_system

    from diffBloch.preprocess import select_orientation_portfolio
    from diffBloch.specs import (
        BeamSelection,
        IntegrationGeometry,
        RockingCurve,
        ScoredHklSelection,
        TrialCoupling,
        UnionCoupling,
    )

    refinement, seed = seed_system()
    integration = IntegrationGeometry()
    step = select_orientation_portfolio(
        refinement,
        OrientationPortfolioSearch(variants=(("default", NelderMeadSearch(max_iterations=1)),)),
        rocking=RockingCurve(sampling=1, integration=integration),
        coupling=TrialCoupling(
            policy=UnionCoupling(g_max=1.0),
            scored=ScoredHklSelection(
                klar=BeamSelection(rsg=1.0, integration=integration),
                g_max=1.0,
            ),
        ),
    )

    (record,) = step_records([step])

    assert record.name == "select_orientation_portfolio"
    assert record.params is not None
    assert record.params["search"]["selector"] == "lowest_terminal_wr2"
    assert seed.orientations

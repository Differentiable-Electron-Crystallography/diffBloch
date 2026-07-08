"""The composition combinators + provenance stamping (``pipeline`` / ``identity`` / ``as_step``).

Provenance is the recipe identity a checkpoint locks against, so these pin the load-bearing
properties: a composed pipeline stamps one record per step in order; the serialized params are
deterministic and sensitive to any spec change; a bare (unrecorded) step stamps ``OPAQUE`` so it
can never yield a false cache hit.
"""

from __future__ import annotations

from dataclasses import replace

from diffBloch.preprocess.pipeline import (
    OPAQUE,
    Plan,
    Step,
    StepRecord,
    as_step,
    identity,
    pipeline,
    spec_to_params,
)
from diffBloch.specs import BeamSelection, IntegrationGeometry, TiltIndependent, TiltSegmentUnion


def _plan() -> Plan:
    # Provenance is grid/orientation-agnostic; a bare Plan with empty geometry exercises stamping.
    return Plan(grid=None, orientations=())  # type: ignore[arg-type]


def _tag(name: str) -> Step:
    """A trivial recorded step that leaves the plan otherwise untouched."""
    return as_step(name, None, lambda plan: plan)


def test_spec_to_params_tags_nested_specs_and_union_arms() -> None:
    params = spec_to_params(BeamSelection(rsg=0.9, integration=IntegrationGeometry(semiangle=1.0)))
    assert params == {
        "__type__": "BeamSelection",
        "rsg": 0.9,
        "dsg": 0.0015,
        "integration": {
            "__type__": "IntegrationGeometry",
            "semiangle": 1.0,
            "geometry": "continuous_rotation",
        },
    }
    # A fieldless union arm is distinguishable from any other empty spec by its type tag.
    assert spec_to_params(TiltIndependent()) == {"__type__": "TiltIndependent"}
    assert spec_to_params(TiltSegmentUnion())["__type__"] == "TiltSegmentUnion"
    assert spec_to_params(None) is None


def test_pipeline_stamps_one_record_per_step_in_order() -> None:
    out = pipeline([_tag("a"), _tag("b"), _tag("c")])(_plan())
    assert [r.name for r in out.provenance] == ["a", "b", "c"]


def test_identical_recipes_produce_equal_provenance() -> None:
    recipe = [as_step("select_beams", BeamSelection(), lambda p: p)]
    first = pipeline(recipe)(_plan())
    second = pipeline([as_step("select_beams", BeamSelection(), lambda p: p)])(_plan())
    assert first.provenance == second.provenance  # StepRecord equality = recipe identity


def test_a_changed_spec_changes_the_record() -> None:
    default = pipeline([as_step("select_beams", BeamSelection(), lambda p: p)])(_plan())
    changed = pipeline([as_step("select_beams", BeamSelection(rsg=0.5), lambda p: p)])(_plan())
    assert changed.provenance != default.provenance


def test_a_bare_step_stamps_opaque() -> None:
    # A caller's plain Plan->Plan closure (no record) must force a safe miss, never a false hit.
    out = pipeline([lambda plan: plan])(_plan())
    assert out.provenance == (OPAQUE,)


def test_identity_records_itself_and_is_a_noop() -> None:
    out = pipeline([identity])(_plan())
    assert out.provenance == (StepRecord(name="identity"),)


def test_empty_pipeline_leaves_provenance_untouched() -> None:
    seeded = replace(_plan(), provenance=(StepRecord(name="prior"),))
    assert pipeline([])(seeded).provenance == (StepRecord(name="prior"),)

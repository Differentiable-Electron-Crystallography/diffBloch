"""The composition combinators + provenance stamping (``pipeline`` / ``identity`` / ``as_step``).

Provenance is the recipe identity a checkpoint locks against, so these pin the load-bearing
properties: a composed pipeline stamps one record per step in order; the serialized params are
deterministic and sensitive to any spec change; a bare (unrecorded) step stamps ``OPAQUE`` so it
can never yield a false cache hit.
"""

from __future__ import annotations

from dataclasses import replace

from tests.unit.synthetic import seed_system

from diffBloch.preprocess import (
    build_orientation_plans,
    integrate_rocking_curve,
    mosaicity,
    select_beams,
)
from diffBloch.preprocess.pipeline import (
    OPAQUE,
    Plan,
    Step,
    StepRecord,
    as_step,
    fork,
    identity,
    pipeline,
    resolve_recipe,
    spec_to_params,
    step_records,
)
from diffBloch.specs import (
    BeamSelection,
    IntegrationGeometry,
    Mosaicity,
    RockingCurve,
    TiltIndependent,
    TiltSegmentUnion,
)


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


# --- fork (the choice combinator) -----------------------------------------------------------------
# These predicates ignore the grid (passed None) -- resolution mechanics are what's under test here;
# grid-driven branch selection + checkpoint reuse/stale is pinned in test_program_checkpoint.py.


def test_resolve_recipe_splices_the_chosen_branch() -> None:
    taken = resolve_recipe(
        [
            _tag("a"),
            fork(lambda g: True, when_true=[_tag("t1"), _tag("t2")], when_false=[_tag("f")]),
        ],
        None,  # predicate ignores the grid
    )
    assert [r.name for r in step_records(taken)] == ["a", "t1", "t2"]
    skipped = resolve_recipe(
        [fork(lambda g: False, when_true=[_tag("t")], when_false=[_tag("f1"), _tag("f2")])], None
    )
    assert [r.name for r in step_records(skipped)] == ["f1", "f2"]


def test_resolve_recipe_flattens_nested_forks() -> None:
    inner = fork(lambda g: True, when_true=[_tag("x"), _tag("y")], when_false=[_tag("no")])
    outer = fork(lambda g: True, when_true=[_tag("a"), inner, _tag("b")], when_false=[_tag("no")])
    assert [r.name for r in step_records(resolve_recipe([outer], None))] == ["a", "x", "y", "b"]


def test_resolved_fork_stamps_the_branch_step_records() -> None:
    resolved = resolve_recipe(
        [fork(lambda g: True, when_true=[_tag("t")], when_false=[_tag("f")])], None
    )
    assert [r.name for r in pipeline(resolved)(_plan()).provenance] == ["t"]


def test_raw_fork_in_pipeline_runs_the_branch_but_records_opaque() -> None:
    # An un-resolved Fork dropped into a raw pipeline still runs the right branch (Fork.__call__),
    # but records OPAQUE (a non-Step in the stamping loop) -- a safe miss, never a false reuse.
    ran: list[str] = []

    def marking(plan: Plan) -> Plan:
        ran.append("t")
        return plan

    branch = fork(lambda g: True, when_true=[as_step("t", None, marking)], when_false=[_tag("f")])
    out = pipeline([branch])(_plan())
    assert ran == ["t"]
    assert out.provenance == (OPAQUE,)


def test_grid_is_invariant_across_the_grid_shaping_steps() -> None:
    # fork's checkpointability rests on plan.grid being the *same value* at every step, so resolving
    # a fork against base.grid equals its branch at the fork's runtime position (see the fork
    # docstring). Enforce it as a tripwire rather than a convention: the grid-shaping steps must
    # thread the same grid object through, never rebuild it. (couple_beams likewise only
    # replace(orientations=...)s -- omitted here because it needs g_max >= 2*coupling-cap.)
    _, seed = seed_system()
    shaped = pipeline(
        [
            select_beams(BeamSelection()),
            build_orientation_plans(),
            integrate_rocking_curve(RockingCurve(sampling=3)),
            mosaicity(Mosaicity(window=2)),
        ]
    )(seed)
    assert shaped.grid is seed.grid

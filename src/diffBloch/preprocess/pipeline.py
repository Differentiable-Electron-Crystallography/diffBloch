"""The preprocess composition combinators: sequence and fixpoint over ``Plan -> Plan`` steps.

A *step* is a pure ``Plan -> Plan`` transformer (it *fits* something -- numerics, orientation,
thickness -- and returns a sharpened :class:`~diffBloch.preprocess.plan.Plan`). :func:`pipeline`
chains steps left to right; :func:`iterate_until` drives one step to a fixpoint (for convergence
testing or alternating fits). Both return a ``Plan -> Plan`` step, so they nest -- a fixpoint of
steps is itself a step. ``refine`` is deliberately *not* expressible here: it is the terminal
``Plan -> Result`` estimator, not a ``Plan -> Plan`` transform.

**Provenance.** A step is self-describing: it carries a :class:`StepRecord` (its name + serialized
params). As :func:`pipeline` applies each step it *stamps* that record onto the resulting
:class:`~diffBloch.preprocess.plan.Plan`'s ``provenance`` tuple, so the final Plan records the
ordered recipe that produced it -- the Writer-monad pattern (each step ``tell``\\ s its record).
This is what lets a checkpoint bind its identity to the recipe, not just the inputs. A step with no
record (a bare closure, a nested composite) stamps :data:`OPAQUE` -- a plan whose provenance
contains it can never be reused (safe: a miss, never a false hit).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import TYPE_CHECKING, Any

from diffBloch.observability import NULL_LOGGER, Logger, PlanStepCompleted
from diffBloch.preprocess.plan import Plan, summarize_plan

if TYPE_CHECKING:
    # Annotation-only (``from __future__ import annotations``): StructureFactorGrid is never
    # touched at
    # runtime here, so keep it a type-only import -- pipeline stays free of a runtime edge up into
    # ``engine`` (no cycle today; this keeps it that way).
    from diffBloch.engine.plan import StructureFactorGrid

__all__ = [
    "OPAQUE",
    "ConvergenceCheck",
    "Fork",
    "PlanStep",
    "Step",
    "StepRecord",
    "as_step",
    "fork",
    "identity",
    "iterate_until",
    "pipeline",
    "resolve_recipe",
    "spec_to_params",
    "step_records",
]

# A step sharpens the Plan; a convergence check compares a step's (previous, just-produced) Plans.
type PlanStep = Callable[[Plan], Plan]
type ConvergenceCheck = Callable[[Plan, Plan], bool]


def spec_to_params(spec: Any) -> dict[str, Any] | None:
    """Serialize a frozen-dataclass value-type to a canonical, deterministic dict for provenance.

    Recurses into nested dataclasses (the specs nest: ``TrialCoupling`` holds a policy +
    ``ScoredHklSelection`` holds a ``BeamSelection`` holds an ``IntegrationGeometry``), tagging each
    with ``__type__`` = its class name so a *fieldless* discriminated-union arm (e.g.
    ``TiltIndependent``, whose ``asdict`` is ``{}``) is distinguishable from any other empty spec.
    Non-dataclass leaves (int/float/str/bool/None, Literals-as-str) pass through; tuples/lists
    recurse elementwise. ``None`` returns ``None`` (a paramless step).
    """
    if spec is None:
        return None
    frozen = _freeze(spec)
    assert isinstance(frozen, dict)  # spec is a dataclass or a mapping of specs -> a dict
    return frozen


def _freeze(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        out: dict[str, Any] = {"__type__": type(obj).__name__}
        for f in fields(obj):
            out[f.name] = _freeze(getattr(obj, f.name))
        return out
    if isinstance(obj, Mapping):  # a step recording several named specs, e.g. {search, coupling}
        return {k: _freeze(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [_freeze(x) for x in obj]
    return obj


@dataclass(frozen=True)
class StepRecord:
    """A step's provenance entry: its ``name`` and canonical serialized ``params`` (or ``None``).

    Two records compare equal iff the step and its params are identical, so a recipe's provenance is
    a stable, comparable identity. ``params`` is the :func:`spec_to_params` form (JSON-able, with
    ``__type__`` tags), so the record round-trips through the lock and the ``.npz`` ``__meta__``.
    """

    name: str
    params: dict[str, Any] | None = None


# A plan whose provenance contains this can never satisfy a freshness check -- a bare/opaque step
# (nested pipeline, iterate_until, a caller's custom closure) forces a safe cache miss.
OPAQUE = StepRecord(name="<opaque>", params=None)


@dataclass(frozen=True)
class Step:
    """A self-describing ``Plan -> Plan`` step: its provenance ``record`` + the ``run`` transform.

    Callable, so a ``Step`` satisfies :data:`PlanStep` structurally and every existing caller
    (``pipeline``, ``run_inference(prepare=...)``) treats it as before; :func:`pipeline` also
    reads ``record`` to stamp provenance.
    """

    record: StepRecord
    run: PlanStep

    def __call__(self, plan: Plan) -> Plan:
        return self.run(plan)


def as_step(name: str, spec: Any, run: PlanStep) -> Step:
    """Wrap a step's ``run`` closure with its :class:`StepRecord` (``name`` + serialized spec)."""
    return Step(record=StepRecord(name=name, params=spec_to_params(spec)), run=run)


def _record_of(step: PlanStep) -> StepRecord:
    return step.record if isinstance(step, Step) else OPAQUE


def step_records(steps: Sequence[PlanStep]) -> tuple[StepRecord, ...]:
    """The recipe ``pipeline(steps)`` will stamp -- one record per step, in order.

    The checkpoint driver reads this *before* running to compare against a lock (does the intended
    recipe match / extend the snapshot's?). A bare step contributes :data:`OPAQUE`, so a recipe
    containing one can be detected and refused (never checkpointed) up front.
    """
    return tuple(_record_of(step) for step in steps)


def _identity(plan: Plan) -> Plan:
    return plan


# The no-op step (pipeline identity element); records "identity" so it is a faithful, comparable
# provenance entry rather than an opaque miss.
identity: Step = Step(record=StepRecord(name="identity"), run=_identity)


def pipeline(steps: Sequence[PlanStep], *, logger: Logger = NULL_LOGGER) -> PlanStep:
    """Compose ``steps`` left to right, stamping each step's record onto the plan's ``provenance``.

    After applying each step, appends its :class:`StepRecord` (or :data:`OPAQUE` for a bare closure)
    to the plan's ``provenance``, so the composed result records the ordered recipe. An empty list
    yields the identity (provenance unchanged).

    ``logger`` (default the null sink) receives a
    :class:`~diffBloch.observability.PlanStepCompleted` after each step -- the step's name as the
    event channel, its ordinal as the step, and :func:`~diffBloch.preprocess.plan.summarize_plan`
    of the resulting plan -- so a fresh preprocess run streams the plan's shape as it evolves.
    Reusing a checkpoint bypasses this runner, so those fire only on a fresh run (the boundary
    :class:`~diffBloch.observability.CouplingSummary` covers the reuse case). Emission is alongside
    the provenance ``tell``; the null default keeps the pure composition path unchanged.
    """

    def run(plan: Plan) -> Plan:
        for index, step in enumerate(steps):
            result = step(plan)
            result = replace(result, provenance=(*plan.provenance, _record_of(step)))
            plan = result
            if logger is not NULL_LOGGER:  # skip summarize_plan on the null path (stay pure/cheap)
                logger.report(
                    PlanStepCompleted(
                        channel=_record_of(step).name,
                        index=index,
                        measurements=summarize_plan(plan),
                    )
                )
        return plan

    return run


def iterate_until(step: PlanStep, *, until: ConvergenceCheck, max_iterations: int = 50) -> PlanStep:
    """Drive ``step`` to a fixpoint: re-apply it until ``until(previous, current)`` holds.

    Returns a ``Plan -> Plan`` step that applies ``step`` repeatedly, checking ``until`` against the
    (previous, just-produced) Plan pair after each application, and returns the first Plan that
    satisfies it. Raises ``RuntimeError`` if ``max_iterations`` is reached without convergence --
    silent non-convergence is never returned. ``max_iterations`` must be >= 1.

    Provenance: the fixpoint stamps a single :data:`OPAQUE` record -- the number of iterations is
    input-dependent, so a faithful per-iteration log would not be a stable recipe identity. A plan
    produced through ``iterate_until`` is therefore not checkpoint-reusable yet (a safe miss); a
    stable record is deferred with the convergence-driver checkpointing work.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")

    def run(plan: Plan) -> Plan:
        current = plan
        for _ in range(max_iterations):
            nxt = step(current)
            if until(current, nxt):
                return replace(nxt, provenance=(*plan.provenance, OPAQUE))
            current = nxt
        raise RuntimeError(f"iterate_until did not converge within {max_iterations} iterations")

    return run


@dataclass(frozen=True)
class Fork:
    """The *choice* combinator: run one of two step lists, chosen by a predicate on the grid.

    The recipe's fourth composition shape (see ``design/decisions/plan-composition-shapes.md``).
    The one rule that makes it checkpointable: **the predicate reads only the
    :class:`~diffBloch.engine.plan.StructureFactorGrid`, invariant across every preprocess step**
    (steps ``replace`` orientations; nothing resizes the grid). So the branch is a deterministic
    function of the experiment's fixed inputs -- knowable *before* running -- rather than of the
    mutating ``Plan``. That keeps the fork *Applicative* (its shape is static), so
    :func:`resolve_recipe` can splice the chosen branch inline into a flat, fork-free recipe before
    the checkpoint lock ever looks at it. A predicate over the ``Plan`` would be Monadic (its shape
    would depend on intermediate results) and is deliberately unrepresentable here. See
    ``design/decisions/combinators-and-recipe-identity.md``.

    Branches are step *lists*, not pre-composed ``pipeline([...])`` closures, so each branch step's
    :class:`StepRecord` survives into the resolved recipe (a composed closure would collapse to one
    :data:`OPAQUE`). :meth:`__call__` lets a ``Fork`` also run ad hoc inside a raw ``pipeline`` --
    it produces the right ``Plan`` but records :data:`OPAQUE` (a non-``Step`` in the stamping loop),
    a safe miss; checkpointable identity comes *only* from :func:`resolve_recipe`.
    """

    predicate: Callable[[StructureFactorGrid], bool]
    when_true: tuple[PlanStep, ...]
    when_false: tuple[PlanStep, ...]

    def resolve(self, grid: StructureFactorGrid) -> tuple[PlanStep, ...]:
        """The branch this fork takes for ``grid`` (the invariant discriminant)."""
        return self.when_true if self.predicate(grid) else self.when_false

    def __call__(self, plan: Plan) -> Plan:
        """Run the chosen branch (ad-hoc use inside a raw ``pipeline``; records ``OPAQUE``)."""
        return pipeline(self.resolve(plan.structure_factor_grid))(plan)


def fork(
    predicate: Callable[[StructureFactorGrid], bool],
    *,
    when_true: Sequence[PlanStep],
    when_false: Sequence[PlanStep],
) -> Fork:
    """Build a :class:`Fork` choosing between two step lists by a predicate on the grid.

    ``predicate`` receives the shared :class:`~diffBloch.engine.plan.StructureFactorGrid` (e.g. a
    cell-volume / grid-size test routing a large cell to a coarse-precision branch); ``when_true`` /
    ``when_false`` are the branch step lists (kept as lists so their records survive resolution).

    ``predicate`` **must be a pure function of the grid** -- no external or mutable state. The grid
    argument is only half the contract: the *type* stops it reading the mutating ``Plan``, but a
    predicate that closed over a global flag would desync the pre-run resolution (for the lock) from
    the runtime branch just as badly. Purity over an input that is itself pipeline-invariant is what
    makes the branch deterministic per experiment.
    """
    return Fork(predicate=predicate, when_true=tuple(when_true), when_false=tuple(when_false))


def resolve_recipe(steps: Sequence[PlanStep], grid: StructureFactorGrid) -> tuple[PlanStep, ...]:
    """Compile every :class:`Fork` away against ``grid`` -> a flat, fork-free step list.

    Splices each fork's chosen branch inline (recursively, so nested forks flatten too). Because the
    grid is invariant across the pipeline, resolving against the *base* grid here yields exactly the
    recipe that will run -- which is what lets the checkpoint lock key on a flat ``step_records``
    list with no knowledge of forks (see ``design/decisions/combinators-and-recipe-identity.md``).
    """
    out: list[PlanStep] = []
    for step in steps:
        if isinstance(step, Fork):
            out.extend(resolve_recipe(step.resolve(grid), grid))
        else:
            out.append(step)
    return tuple(out)

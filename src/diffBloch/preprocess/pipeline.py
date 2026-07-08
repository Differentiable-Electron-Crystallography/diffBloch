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
from typing import Any

from diffBloch.preprocess.plan import Plan

__all__ = [
    "OPAQUE",
    "ConvergenceCheck",
    "PlanStep",
    "Step",
    "StepRecord",
    "as_step",
    "identity",
    "iterate_until",
    "pipeline",
    "spec_to_params",
    "step_records",
]

# A step sharpens the Plan; a convergence check compares a step's (previous, just-produced) Plans.
type PlanStep = Callable[[Plan], Plan]
type ConvergenceCheck = Callable[[Plan, Plan], bool]


def spec_to_params(spec: Any) -> dict[str, Any] | None:
    """Serialize a frozen-dataclass value-type to a canonical, deterministic dict for provenance.

    Recurses into nested dataclasses (the specs nest: ``TrialCoupling`` holds a policy +
    ``ScoredSelection`` holds a ``BeamSelection`` holds an ``IntegrationGeometry``), tagging each
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


def pipeline(steps: Sequence[PlanStep]) -> PlanStep:
    """Compose ``steps`` left to right, stamping each step's record onto the plan's ``provenance``.

    After applying each step, appends its :class:`StepRecord` (or :data:`OPAQUE` for a bare closure)
    to the plan's ``provenance``, so the composed result records the ordered recipe. An empty list
    yields the identity (provenance unchanged).
    """

    def run(plan: Plan) -> Plan:
        for step in steps:
            result = step(plan)
            result = replace(result, provenance=(*plan.provenance, _record_of(step)))
            plan = result
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

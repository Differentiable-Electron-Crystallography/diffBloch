"""``select_orientation_portfolio``: choose among Ben-style orientation optimizers per rotation.

This is an opt-in typed Python composition step for orientation-recreation experiments, not part of
the default app recipe. It does not implement a second orientation optimizer. Instead it runs Ben's
canonical :func:`optimize_orientation` step multiple times from the same built seed plan, changing
only the labelled :class:`~diffBloch.specs.NelderMeadSearch` parameters supplied by
:class:`~diffBloch.specs.OrientationPortfolioSearch`.

For each rotation independently, the step chooses the candidate with the lowest terminal wR2 loss
under ``selection_method``. This is what "portfolio" means here: different orientations in the
returned ``Plan`` may have been produced by different optimizer-parameter policies. The selector is
non-oracle; reference orientation CSVs are not accepted or consulted.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite

from diffBloch.core.solver import SolverMethod
from diffBloch.engine.plan import OrientationPlanLike
from diffBloch.params import Device
from diffBloch.preprocess.experiment import RefinementSetup
from diffBloch.preprocess.pipeline import PlanStep, as_step
from diffBloch.preprocess.plan import Plan, require_built_plans
from diffBloch.preprocess.scoring import build_engine
from diffBloch.preprocess.steps.beams import build_orientation_plans
from diffBloch.preprocess.steps.optimize_orientation import optimize_orientation
from diffBloch.specs import (
    NO_ABSORPTION,
    Absorption,
    Mosaicity,
    NelderMeadSearch,
    OrientationPortfolioSearch,
    RockingCurve,
    TrialCoupling,
)

__all__ = ["select_orientation_portfolio"]


@dataclass(frozen=True)
class _VariantResult:
    """One labelled optimizer variant plus terminal per-rotation wR2 losses."""

    label: str
    plan: Plan
    wr2: tuple[float, ...]


def select_orientation_portfolio(
    refinement: RefinementSetup,
    search: OrientationPortfolioSearch | None = None,
    *,
    rocking: RockingCurve,
    mosaicity: Mosaicity | None = None,
    coupling: TrialCoupling,
    orientation_method: SolverMethod = "matrix_exp",
    selection_method: SolverMethod = "bloch_eigen",
    validate: bool = True,
    workers: int = 1,
    device: Device | None = None,
    max_batch: int | None = None,
    absorption: Absorption = NO_ABSORPTION,
) -> PlanStep:
    """Return a ``Plan -> Plan`` step selecting an orientation optimizer per rotation.

    The input ``plan`` must be in the candidate phase. The step first builds the same rocking,
    mosaic, coupled seed geometry that each optimizer variant will refine. It then applies
    :func:`optimize_orientation` once per configured variant and evaluates each resulting plan with
    terminal wR2. The returned plan splices together the best-scoring orientation at each rotation
    position.

    ``orientation_method`` controls the solver used during each Nelder-Mead search.
    ``selection_method`` controls the solver used for terminal wR2 winner selection. In the quartz
    anchor diagnostics this mirrors the e2e shape: matrix-exponential search, Bloch-eigen terminal
    scoring. ``validate``, ``workers``, ``device``, and ``max_batch`` are execution controls
    forwarded to the optimizer/evaluator; the portfolio spec and scientific geometry knobs are
    recorded in the step provenance.
    """
    portfolio = OrientationPortfolioSearch() if search is None else search
    build = build_orientation_plans(
        rocking,
        mosaicity,
        coupling=coupling.policy,
        scoring_selection=coupling.scored.klar,
    )

    def run(plan: Plan) -> Plan:
        seed_plan = build(plan)
        variant_results = tuple(
            _run_variant(
                label,
                seed_plan,
                refinement,
                search_variant,
                coupling=coupling,
                orientation_method=orientation_method,
                selection_method=selection_method,
                validate=validate,
                workers=workers,
                device=device,
                max_batch=max_batch,
                absorption=absorption,
            )
            for label, search_variant in portfolio.variants
        )
        selected = _select_by_wr2(variant_results)
        return replace(seed_plan, orientations=selected)

    return as_step(
        "select_orientation_portfolio",
        {
            "search": portfolio,
            "rocking": rocking,
            "mosaicity": mosaicity,
            "coupling": coupling,
            "orientation_method": orientation_method,
            "selection_method": selection_method,
            "absorption": absorption,
        },
        run,
    )


def _run_variant(
    label: str,
    seed_plan: Plan,
    refinement: RefinementSetup,
    search: NelderMeadSearch,
    *,
    coupling: TrialCoupling,
    orientation_method: SolverMethod,
    selection_method: SolverMethod,
    validate: bool,
    workers: int,
    device: Device | None,
    max_batch: int | None,
    absorption: Absorption,
) -> _VariantResult:
    step = optimize_orientation(
        refinement,
        search,
        method=orientation_method,
        coupling=coupling,
        validate=validate,
        workers=workers,
        device=device,
        max_batch=max_batch,
        absorption=absorption,
    )
    fitted = step(seed_plan)
    engine = build_engine(
        fitted,
        refinement,
        method=selection_method,
        max_batch=max_batch,
        absorption=absorption,
    )
    params = refinement.params if device is None else refinement.params.to(device)
    fgb = engine.fgb(params)
    return _VariantResult(
        label=label,
        plan=fitted,
        wr2=tuple(
            float(engine.score_orientation(orientation, fgb))
            for orientation in require_built_plans(fitted)
        ),
    )


def _select_by_wr2(results: tuple[_VariantResult, ...]) -> tuple[OrientationPlanLike, ...]:
    """Choose the lowest finite terminal wR2 independently at each plan position.

    Finite candidates always beat non-finite candidates; if every variant is non-finite for a
    rotation, the first variant wins deterministically rather than making the whole preprocess step
    fail.
    """
    if not results:
        raise ValueError("orientation portfolio needs at least one variant result")
    n_orientations = len(results[0].plan.orientations)
    if any(len(result.plan.orientations) != n_orientations for result in results):
        raise ValueError("orientation portfolio variant plans must have the same length")
    if any(len(result.wr2) != n_orientations for result in results):
        raise ValueError("orientation portfolio scores must match variant plan length")

    selected = []
    for index in range(n_orientations):
        best = min(results, key=lambda result: _wr2_key(result.wr2[index]))
        selected.append(require_built_plans(best.plan)[index])
    return tuple(selected)


def _wr2_key(value: float) -> tuple[bool, float]:
    """Sort key: finite scores first, lower scores better, all non-finite tied."""
    return (not isfinite(value), value if isfinite(value) else 0.0)

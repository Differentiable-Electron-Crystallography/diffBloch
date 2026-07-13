"""The default recipe's large-cell fork (``_recipe_steps``): routing + checkpoint transparency.

The fit tail forks on unit-cell volume -- a large cell takes a coarse fp32 search with the gather
integrity checks skipped; a small cell takes the exact fp64 path. Two properties matter and are
pinned here: (1) the fork routes at ``_LARGE_CELL_THRESHOLD_A3``, and (2) it is *transparent to the
recipe identity* -- both branches record the same step names/params, so resolving it (as
``_prepare`` does) never changes the lock. Property (2) keeps the committed quartz checkpoint valid.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from diffBloch.app.program import _LARGE_CELL_THRESHOLD_A3, _recipe_steps
from diffBloch.config import load_experiment
from diffBloch.io import read_observations, read_structure
from diffBloch.observability import NULL_LOGGER
from diffBloch.preprocess import from_experiment, resolve_recipe, step_records
from diffBloch.preprocess.pipeline import Fork

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _steps(material: str = "quartz_anchor") -> list:
    root = FIXTURES / material
    cfg, _ = load_experiment(root)
    structure = read_structure(root / cfg.inputs.structure)
    observations = read_observations(root / cfg.inputs.observations)
    setup = from_experiment(structure, observations, cfg)
    return _recipe_steps(cfg, setup.refinement, NULL_LOGGER)


def _the_fork() -> Fork:
    forks = [s for s in _steps() if isinstance(s, Fork)]
    assert len(forks) == 1, "the fit tail should be exactly one fork"
    return forks[0]


def test_fork_predicate_routes_at_the_threshold() -> None:
    """Just above the volume threshold -> large branch; just below -> small branch."""
    fork = _the_fork()
    assert fork.predicate(SimpleNamespace(cell_volume=_LARGE_CELL_THRESHOLD_A3 + 1.0)) is True
    assert fork.predicate(SimpleNamespace(cell_volume=_LARGE_CELL_THRESHOLD_A3 - 1.0)) is False


def test_fork_is_transparent_to_recipe_identity() -> None:
    """For one config, the small-cell and large-cell branches resolve to the *same* records.

    Resolving the *same* recipe against a below-threshold vs above-threshold grid takes the fp64 vs
    fp32 branch, yet both record identical step names *and* params: fp32 / validate live only in the
    branch closures (execution-only), never in the step identity. This is the property that keeps
    the committed quartz checkpoint reusable after the fork lands (the fork is invisible to the
    lock).
    """
    steps = _steps("quartz_anchor")
    small = step_records(resolve_recipe(steps, SimpleNamespace(cell_volume=100.0)))
    large = step_records(resolve_recipe(steps, SimpleNamespace(cell_volume=5000.0)))
    names = [
        "select_beams",
        "build_orientation_plans",
        "integrate_rocking_curve",
        "mosaicity",
        "fit_orientation",
        "fit_thickness",
    ]
    assert [r.name for r in small] == names == [r.name for r in large]
    # identical params too (fit_orientation records only {search, coupling}; no precision/validate)
    assert [r.params for r in small] == [r.params for r in large]

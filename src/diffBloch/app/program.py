"""The default experiment runner the ``run infer`` CLI exposes.

:func:`run_experiment` encodes the faithful default recipe as one ordered ``Plan -> Plan``
pipeline: plan-shaping (``select_beams`` -> ``integrate_rocking_curve`` -> ``mosaicity``) followed
by parameter fitting (``fit_orientation`` -> ``fit_thickness``), then ``run_inference`` evaluates it
-- so a caller
with an experiment directory gets the faithful result in one call. It is a *convenience*, not the
only path: every step is ordinary public API, so a Python user who wants a different composition
(convergence, a custom order, dropping mosaicity) composes their own ``pipeline([...])`` with
``from_experiment`` + ``run_inference`` directly. The CLI stays thin by delegating here and holds no
science.
"""

from __future__ import annotations

from pathlib import Path

from diffBloch.config import load_experiment
from diffBloch.io import read_observations, read_structure
from diffBloch.observability import NULL_LOGGER, Logger
from diffBloch.preprocess import (
    fit_orientation,
    fit_thickness,
    from_experiment,
    integrate_rocking_curve,
    mosaicity,
    pipeline,
    run_inference,
    select_beams,
)
from diffBloch.preprocess.inference import InferenceResult

__all__ = ["run_experiment"]


def run_experiment(experiment_dir: str | Path, *, logger: Logger = NULL_LOGGER) -> InferenceResult:
    """Load, preprocess, and score every rotation of the experiment at ``experiment_dir``.

    Loads ``experiment.yaml`` (verifying the input lock), reads the structure + observations, builds
    the geometry via ``from_experiment``, runs the standard integrated recipe, and evaluates the
    forward model over all rotations with ``run_inference`` -- emitting per-rotation observations to
    ``logger`` (the null default discards them). Returns the
    :class:`~diffBloch.preprocess.inference.InferenceResult` (per-rotation ``R_obs`` + aggregate).

    For a different preprocess, compose your own pipeline with ``from_experiment`` +
    ``run_inference``; this function is just the common case the CLI runs.
    """
    root = Path(experiment_dir)
    cfg, _lock = load_experiment(root)
    structure = read_structure(root / cfg.inputs.structure)
    observations = read_observations(root / cfg.inputs.observations)
    setup = from_experiment(structure, observations, cfg)
    refinement = setup.refinement
    prepare = pipeline(
        [
            select_beams(cfg.numerics.to_beam_selection()),
            integrate_rocking_curve(cfg.numerics.to_rocking_curve()),
            mosaicity(cfg.numerics.mosaicity),
            fit_orientation(
                refinement, cfg.preprocess.orientation.to_search(), method=cfg.solver.refine
            ),
            fit_thickness(refinement, cfg.preprocess.thickness.to_grid(), method=cfg.solver.refine),
        ]
    )
    return run_inference(
        setup.plans.combined,
        refinement,
        prepare=prepare,
        method=cfg.solver.inference,
        logger=logger,
    )

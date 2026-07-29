"""Real-data anchor for the forward-model convergence lever: how many rocking tilts quartz needs.

Complements the accuracy anchors in ``test_anchor.py`` (which fit the model to *observed* data)
with a **numerical resolution study** on the same quartz fixture: :func:`converge_sampling` grows
the rocking-curve tilt count until two *consecutive* integrated simulations stop changing (the
consecutive-simulation R-factor drops below the tolerance), asking "has the tilt integration stopped
depending on the tilt count?" -- run orthogonally to the accuracy fit.

The result is a genuine finding on real quartz: at the faithful ``diffBloch_private`` threshold
(``r_factor_threshold = 0.005``) the integrated pattern converges at **39 tilts**, just below the
reference experiment's chosen ``rocking_curve_sampling = 42`` -- i.e. 42 is mildly oversampled, and
the convergence lever recovers the resolution requirement from the physics rather than a hardcoded
guess. The consecutive-simulation R-factor falls monotonically past the noisy low-tilt regime
(~0.006 at 37 tilts, ~0.0044 at 39), so 39 is the first sub-threshold step of the sampling=1, step=2
sweep.

This is an opt-in ``e2e`` test (excluded from ``just check``). Full-99 is the default (the
representative resolution requirement -- a subset gives a slightly different count, 39-41, so it is
not pinned); the ~9s full run is cheap thanks to the precomputed structure-factor gather.
``DIFFBLOCH_ANCHOR_ROTATIONS=N`` runs the first ``N`` rotations for a quick sanity check.
"""

import os
from dataclasses import replace
from pathlib import Path

import pytest

from diffBloch.config import load_experiment
from diffBloch.io import read_observations, read_structure
from diffBloch.preprocess import (
    ConvergenceTolerance,
    build_orientation_plans,
    converge_sampling,
    from_experiment,
    require_orientation_plans,
    select_beams,
)

pytestmark = pytest.mark.e2e

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "quartz_anchor"

# The sampling=1, step=2 sweep at the private's 0.005 threshold converges the full-99 integrated
# quartz pattern at 39 tilts (repeatable: CPU / float64 / no RNG). Pinned exactly as a
# characterization; the durable scientific claim -- 39 < the reference 42, and well above a
# spurious low-tilt dip -- is asserted separately so a cross-platform eigensolver shift of +/-1 tilt
# would flag the exact pin without hiding the finding.
EXPECTED_CONVERGED_SAMPLING = 39
SWEEP_STEP = 2.0


def test_quartz_sampling_convergence() -> None:
    cfg, _lock = load_experiment(FIXTURE_ROOT)
    structure = read_structure(FIXTURE_ROOT / cfg.inputs.structure)
    observations = read_observations(FIXTURE_ROOT / cfg.inputs.exp_data)
    setup = from_experiment(structure, observations, cfg)

    plan = setup.plans.combined
    reference_sampling = cfg.blochwave.rocking_curve_sampling  # 42, the reference's tilt count
    subset_env = os.environ.get("DIFFBLOCH_ANCHOR_ROTATIONS")
    n_rotations = int(subset_env) if subset_env else len(plan.orientations)
    if not 1 <= n_rotations <= len(plan.orientations):
        raise ValueError(f"DIFFBLOCH_ANCHOR_ROTATIONS must be in 1..{len(plan.orientations)}")
    plan = replace(plan, orientations=plan.orientations[:n_rotations])

    # Select the beam set once, build it, then sweep the tilt count from a single static solve
    # (sampling=1) upward until consecutive integrated simulations agree to the tolerance.
    seed = build_orientation_plans()(
        select_beams(cfg.blochwave.to_beam_selection(setup.integration))(plan)
    )
    rocking = replace(cfg.blochwave.to_rocking_curve(setup.integration), sampling=1)
    converge = converge_sampling(
        rocking,
        setup.refinement,
        ConvergenceTolerance(),
        step=SWEEP_STEP,
        method=cfg.blochwave.solver.refine,
    )
    converged = converge(seed)
    # One beam plan is built per tilt, so the beam-plan count is the converged tilt count.
    converged_sampling = len(require_orientation_plans(converged)[0].beam_plans)

    # Durable scientific claim (platform-independent): the pattern converges below the reference's
    # 42 tilts (so 42 is not under-sampled) but well above the noisy low-tilt regime (a genuine
    # resolution requirement, not a spurious early dip).
    assert converged_sampling < reference_sampling
    assert converged_sampling >= 30

    if n_rotations == observations.n_rotations:  # the representative full-experiment requirement
        assert converged_sampling == EXPECTED_CONVERGED_SAMPLING

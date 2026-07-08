"""North-star characterization anchor: the quartz reference experiment, run end-to-end.

The fixture is self-contained: the real quartz input files and the private reference metrics are
present in this package and hash-verified. This test loads the experiment from the filesystem
through the **public API** (``from_experiment`` + the ``preprocess`` pipeline + ``run_inference``),
runs the whole 99-rotation experiment, and pins the aggregate observed R-factor -- it reads like a
real experiment, reaching into no engine internals.

What is pinned (deterministic; CPU / float64; no RNG), each an aggregate ``mean_r_obs`` over the
99 rotations (each yielding a finite ``R_obs`` -- guarding the ``rbragg`` NaN-safety regression):

  * **coupled (0.0506)** -- the faithful default (:func:`test_quartz_coupled_anchor`), scored
    CI-fast against a committed frozen checkpoint; the full from-scratch fit is opt-in
    (:func:`test_quartz_coupled_anchor_full`, ``DIFFBLOCH_ANCHOR_FULL=1``, ~6-16 min);
  * **tilt-independent (0.0686)** -- the non-coupled integrated recipe, kept as a power-user
    composition (:func:`test_quartz_tilt_independent_anchor`);
  * **static baseline (0.174)** -- ``select_beams -> fit_orientation -> fit_thickness`` **without**
    rocking-curve integration (:func:`test_quartz_reference_anchor`).

The coupled 0.0506 is the closest from-scratch approach to the private reference ``R_obs``
(0.043766); the remaining gap is basin chaos in the greedy hexagonal descent, not a physics
divergence (the forward-solver coupling parity is proven exactly in ``test_coupling_parity``). The
tilt-independent 0.0686 is the cost of holding the beam set at the seed vs the private's 12-segment
tilt-union. An earlier, larger static baseline (~0.298) was an artefact of a beam-selection geometry
bug (the ``sg_max`` lever arm used the beam-transverse plane instead of the goniometer-rock-axis
distance); fixing it is what makes the integrated recipe reproduce the reference reflection counts.
The reference metadata is checked first as an independent provenance guard.

Still pending: the finer-grained per-rotation intermediate-tensor goldens (``Fgb``, the structure
matrix ``A``, the exit wave ``psi``, ``I_sim``) -- a separate, heavier deliverable.

This is an opt-in ``e2e`` test (excluded from ``just check``). Default e2e cost: the coupled
headline is seconds (checkpoint reuse), the tilt-independent + static pins are full fits (~3-4 min
each); the coupled full fit is opt-in only.
"""

import json
import logging
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from diffBloch.app.program import run_experiment
from diffBloch.config import load_experiment, sha256_file
from diffBloch.io import read_observations, read_structure
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

pytestmark = pytest.mark.e2e

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "quartz_anchor"

# From-scratch static baseline: the fit pipeline (select_beams -> fit_orientation -> fit_thickness)
# evaluated over all 99 rotations without rocking-curve integration, under the corrected
# goniometer-rock-axis sg_max lever arm. The reference R_obs (0.043766) is reached only with the
# reference's optim orientations + integrated recipe (which our model matches to 0.0594); C3 (fit/
# eval integration coupling) closes the rest. The tolerance is loose enough to absorb cross-platform
# eigensolver differences (degenerate-eigenvector ordering) while still catching a physics
# regression; tighten once CI confirms the value is stable.
EXPECTED_MEAN_R_OBS = 0.174
MEAN_R_OBS_TOL = 1e-2

# The faithful coupled recipe -- the DEFAULT of run_experiment / `run infer`: the integrated
# preprocess (select_beams -> integrate_rocking_curve -> mosaicity) then the refine phase
# (fit_orientation -> fit_thickness) with fit_orientation under the private's per-trial beam
# coupling (SOLVE union + SCORED set re-derived at every trial orientation). It reaches mean
# R_obs = 0.0506, the closest from-scratch approach to the private reference 0.043766 (the remaining
# gap is basin chaos in the greedy hexagonal descent, not a physics divergence). The tolerance is
# loose enough for cross-platform eigensolver degeneracy while still catching a physics regression.
EXPECTED_COUPLED_MEAN_R_OBS = 0.0506
COUPLED_MEAN_R_OBS_TOL = 1e-2

# The tilt-independent recipe -- the same integrated pipeline but fit WITHOUT per-trial coupling
# (one fixed beam set across the search). It is no longer the default (coupling is), but stays
# characterized as a power-user composition: it reaches 0.0686, the cost of holding the beam set at
# the seed vs the private's 12-segment tilt-union. A subset run (DIFFBLOCH_ANCHOR_ROTATIONS below)
# is unrepresentative of this mean, so it only asserts the pipeline runs and beats the static base.
EXPECTED_TILT_INDEPENDENT_MEAN_R_OBS = 0.0686
TILT_INDEPENDENT_MEAN_R_OBS_TOL = 1e-2
STATIC_BASELINE_R_OBS = 0.174  # subset sanity: the integrated fit must sit comfortably below this


@pytest.mark.parametrize("material", ["quartz"])
def test_quartz_reference_anchor(material: str) -> None:
    assert material == "quartz"

    cfg, lock = load_experiment(FIXTURE_ROOT)
    assert cfg.name == "quartz-anchor"
    assert cfg.solver.inference == "bloch_eigen"
    assert cfg.numerics.g_max == 4.5
    assert cfg.sample.thicknesses == (820.0,)
    assert cfg.inputs.structure == lock.structure.ref
    assert cfg.inputs.observations == lock.observations.ref

    structure = read_structure(FIXTURE_ROOT / cfg.inputs.structure)
    observations = read_observations(FIXTURE_ROOT / cfg.inputs.observations)
    assert structure.n_atoms == 2
    assert structure.n_symops == 6
    assert observations.n_rotations == 99
    assert observations.n_reflections == 6666

    # Independent provenance guard: the private reference metadata matches the anchor manifest.
    manifest = json.loads((FIXTURE_ROOT / "anchor_manifest.json").read_text())
    reference = json.loads((FIXTURE_ROOT / manifest["reference_results"]["path"]).read_text())
    assert (
        sha256_file(FIXTURE_ROOT / "reference_results.json")
        == manifest["reference_results"]["sha256"]
    )
    assert reference["seed"] == manifest["execution"]["seed"]
    assert reference["n_rotations"] == manifest["reference_results"]["n_rotations"]
    assert reference["N_int_all"] == manifest["reference_results"]["N_int_all"]
    assert reference["N_int_obs"] == manifest["reference_results"]["N_int_obs"]
    assert reference["summary"]["R_obs"] == pytest.approx(
        manifest["reference_results"]["summary"]["R_obs"]
    )
    assert cfg.sample.thicknesses == tuple(manifest["execution"]["thicknesses"])

    # Run the whole experiment through the public API over all 99 rotations.
    setup = from_experiment(structure, observations, cfg)
    refinement = setup.refinement
    prepare = pipeline(
        [
            select_beams(cfg.numerics.to_beam_selection()),
            fit_orientation(
                refinement, cfg.preprocess.orientation.to_search(), method=cfg.solver.refine
            ),
            fit_thickness(refinement, cfg.preprocess.thickness.to_grid(), method=cfg.solver.refine),
        ]
    )
    result = run_inference(
        setup.plans.combined, refinement, prepare=prepare, method=cfg.solver.inference
    )

    assert result.n_evaluated == observations.n_rotations
    assert result.mean_r_obs == pytest.approx(EXPECTED_MEAN_R_OBS, abs=MEAN_R_OBS_TOL)

    # Finer-grained per-rotation tensor goldens remain a separate deliverable.
    for tensor in ("Fgb", "A", "psi", "I_sim"):
        assert manifest["intermediate_tensors"][tensor]["status"] == "pending"


def test_quartz_coupled_anchor(caplog: pytest.LogCaptureFixture) -> None:
    """Headline north-star (CI-fast): score the committed frozen coupled checkpoint, no fit.

    The fixture ships ``plan.npz`` + ``plan.lock`` -- the settled coupled ``Plan`` (fitted
    orientations, 12-segment tilt-union couplings, pinned scored sets), the output of the faithful
    default recipe. ``run_experiment`` **reuses** it (release-gated, so it stays valid across
    commits within a release) and only runs ``run_inference``, reaching ``mean R_obs = 0.0506`` in
    seconds -- the CI cost of the coupled north-star without paying the ~6-16 min fit. The full fit
    is exercised on demand by :func:`test_quartz_coupled_anchor_full`; asserting the ``full reuse``
    diagnostic proves this path did *not* re-fit.
    """
    assert (FIXTURE_ROOT / "plan.npz").exists() and (FIXTURE_ROOT / "plan.lock").exists()
    with caplog.at_level(logging.INFO, logger="diffBloch.app.program"):
        result = run_experiment(FIXTURE_ROOT)
    assert "full reuse" in caplog.text  # reused the checkpoint; no fit ran (the CI-fast guarantee)
    assert result.n_evaluated == 99  # every rotation yields a finite R_obs
    assert result.mean_r_obs == pytest.approx(
        EXPECTED_COUPLED_MEAN_R_OBS, abs=COUPLED_MEAN_R_OBS_TOL
    )


def test_quartz_coupled_anchor_full(tmp_path: Path) -> None:
    """Full coupled preprocess + refine from scratch (~6-16 min); opt-in, local-only.

    Set ``DIFFBLOCH_ANCHOR_FULL=1`` to run. Copies the CSV-less inputs into a scratch dir (no
    checkpoint present), so ``run_experiment`` recomputes the whole faithful coupled recipe and must
    reach the same ``0.0506`` the committed checkpoint scores -- proving the shipped checkpoint is
    reproducible from the inputs, not a stale artifact.
    """
    if os.environ.get("DIFFBLOCH_ANCHOR_FULL") != "1":
        pytest.skip("set DIFFBLOCH_ANCHOR_FULL=1 to run the full coupled fit (~6-16 min)")
    exp = tmp_path / "quartz"
    exp.mkdir()
    for name in ("experiment.yaml", "experiment.lock", "enantiomer_1.cif", "exp_data.cif_pets"):
        shutil.copy(FIXTURE_ROOT / name, exp / name)
    result = run_experiment(exp)  # no checkpoint in the scratch dir -> full coupled fit
    assert result.n_evaluated == 99
    assert result.mean_r_obs == pytest.approx(
        EXPECTED_COUPLED_MEAN_R_OBS, abs=COUPLED_MEAN_R_OBS_TOL
    )


def test_quartz_tilt_independent_anchor() -> None:
    """The tilt-independent recipe (no per-trial coupling) -> 0.0686. Power-user composition.

    No longer the default (coupling is), but kept characterized: the same integrated pipeline with
    ``fit_orientation`` holding one fixed beam set across the search (``coupling=None``), composed
    directly via the public steps. Runs all 99 rotations by default (~3-4 min); set
    ``DIFFBLOCH_ANCHOR_ROTATIONS=N`` for a subset sanity (unrepresentative of the full mean, so it
    only asserts the pipeline runs and beats the static baseline).
    """
    cfg, _lock = load_experiment(FIXTURE_ROOT)
    structure = read_structure(FIXTURE_ROOT / cfg.inputs.structure)
    observations = read_observations(FIXTURE_ROOT / cfg.inputs.observations)
    setup = from_experiment(structure, observations, cfg)
    refinement = setup.refinement

    plan = setup.plans.combined
    subset_env = os.environ.get("DIFFBLOCH_ANCHOR_ROTATIONS")
    n_rotations = int(subset_env) if subset_env is not None else len(plan.orientations)
    if not 1 <= n_rotations <= len(plan.orientations):
        raise ValueError(f"DIFFBLOCH_ANCHOR_ROTATIONS must be in 1..{len(plan.orientations)}")
    plan = replace(plan, orientations=plan.orientations[:n_rotations])

    prepare = pipeline(
        [
            select_beams(cfg.numerics.to_beam_selection()),
            integrate_rocking_curve(cfg.numerics.to_rocking_curve()),
            mosaicity(cfg.numerics.mosaicity),
            fit_orientation(  # coupling=None (default): the tilt-independent fit
                refinement, cfg.preprocess.orientation.to_search(), method=cfg.solver.refine
            ),
            fit_thickness(refinement, cfg.preprocess.thickness.to_grid(), method=cfg.solver.refine),
        ]
    )
    result = run_inference(plan, refinement, prepare=prepare, method=cfg.solver.inference)

    assert result.n_evaluated == n_rotations
    if n_rotations == observations.n_rotations:
        assert result.mean_r_obs == pytest.approx(
            EXPECTED_TILT_INDEPENDENT_MEAN_R_OBS, abs=TILT_INDEPENDENT_MEAN_R_OBS_TOL
        )
    else:  # subset sanity: proves the pipeline runs and beats the static baseline
        assert result.mean_r_obs < STATIC_BASELINE_R_OBS

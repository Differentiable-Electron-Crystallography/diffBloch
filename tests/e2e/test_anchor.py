"""North-star characterization anchor: the quartz reference experiment, run end-to-end.

The fixture is self-contained: the real quartz input files and the private reference metrics are
present in this package and hash-verified. This test loads the experiment from the filesystem
through the **public API** (``from_experiment`` + the ``preprocess`` pipeline + ``run_inference``),
runs the whole 99-rotation experiment, and pins the aggregate observed R-factor -- it reads like a
real experiment, reaching into no engine internals.

What is pinned today (deterministic; CPU / float64; no RNG):

  * ``n_evaluated == 99`` -- every rotation yields a finite ``R_obs`` (guards the ``rbragg``
    NaN-safety regression directly), and
  * ``mean_r_obs`` -- the from-scratch ``select_beams -> fit_orientation -> fit_thickness``
    pipeline **without** rocking-curve integration.

The from-scratch static baseline (~0.174) sits above the private reference ``R_obs`` (0.043766)
because the reference evaluates *pre-optimised* orientations with 42-tilt rocking-curve integration,
whereas this pins a from-scratch static fit. With the reference's own optim orientations + the
integrated recipe our forward model reaches 0.0594; the
remaining path to the reference is the fit/eval integration coupling (C3) plus a small residual. An
earlier, larger baseline (~0.298) was an artefact of a beam-selection geometry bug (the ``sg_max``
lever arm used the beam-transverse plane instead of the goniometer-rock-axis distance); fixing it
lowered this baseline and is what makes the integrated recipe reproduce the reference reflection
counts. The reference metadata is checked first as an independent provenance guard.

Still pending: the finer-grained per-rotation intermediate-tensor goldens (``Fgb``, the structure
matrix ``A``, the exit wave ``psi``, ``I_sim``) -- a separate, heavier deliverable.

This is an opt-in ``e2e`` test (excluded from ``just check``); the full run takes ~80s.
"""

import json
import os
import re
from dataclasses import replace
from pathlib import Path

import pytest

from diffBloch.app.cli import main as cli_main
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

# The integrated recipe: the plan-shaping preprocess phase (select_beams -> integrate_rocking_curve
# -> mosaicity) then the refine phase (fit_orientation -> fit_thickness), so each candidate is fit
# AND evaluated under the 42-tilt integration WITH the private's window-5 mosaicity smoothing (C3
# fit/eval coupling). Declaring mosaicity is the faithful choice (the private runs mosaicity: true);
# it moves the mean from 0.0655 (mosaicity omitted -- a half-faithful cancellation with our coupling
# divergence) to 0.0686. The residual is now the single named cause: our tilt-independent coupling
# vs the private's 12-segment tilt-union (closing that reaches the reference 0.0438). The tolerance
# mirrors the static pin (loose enough for cross-platform eigensolver degeneracy, tight enough to
# catch a physics regression); tighten once CI confirms stability. A subset run (see
# DIFFBLOCH_ANCHOR_ROTATIONS below) is unrepresentative of this mean, so it only asserts the coupled
# pipeline runs and beats the static baseline.
EXPECTED_INTEGRATED_MEAN_R_OBS = 0.0686
INTEGRATED_MEAN_R_OBS_TOL = 1e-2
STATIC_BASELINE_R_OBS = 0.174  # subset sanity: coupled must sit comfortably below this


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


def test_quartz_integrated_anchor(capsys: pytest.CaptureFixture[str]) -> None:
    """Pin the integrated fit/eval-coupled recipe (C3) end-to-end over all 99 rotations.

    This is the coupled counterpart to :func:`test_quartz_reference_anchor`: the same public-API
    pipeline with ``integrate_rocking_curve`` + ``mosaicity`` in the preprocess phase, so every
    candidate is fit *and* scored under the 42-tilt rocking-curve integration with the private's
    window-5 mosaicity smoothing. It reaches ``mean R_obs = 0.0686``,
    well below the static ``0.174`` -- the measured fit/eval-consistency payoff.

    Runs all 99 rotations by default (~3-4 min), driven through the CLI (``diffbloch run infer``),
    which executes this same integrated recipe via ``app.program.run_experiment`` -- so the
    headline pin also covers the user-facing entry point. For a quick sanity check or a cloud run
    where compute is expensive, set ``DIFFBLOCH_ANCHOR_ROTATIONS=N`` to run only the first ``N``
    rotations via the Python API (the CLI has no subset lever); the subset mean is not
    representative of the full-99 headline, so that mode only asserts the coupled pipeline
    completes and sits comfortably below the static baseline.
    """
    subset_env = os.environ.get("DIFFBLOCH_ANCHOR_ROTATIONS")
    if subset_env is None:
        rc = cli_main(["run", "infer", str(FIXTURE_ROOT)])
        out = capsys.readouterr().out
        assert rc == 0, out
        match = re.fullmatch(r"evaluated (\d+) rotations; mean R_obs = (\d+\.\d+)\n", out)
        assert match is not None, out
        assert int(match.group(1)) == 99  # every rotation yields a finite R_obs
        assert float(match.group(2)) == pytest.approx(
            EXPECTED_INTEGRATED_MEAN_R_OBS, abs=INTEGRATED_MEAN_R_OBS_TOL
        )
        return

    cfg, _lock = load_experiment(FIXTURE_ROOT)
    structure = read_structure(FIXTURE_ROOT / cfg.inputs.structure)
    observations = read_observations(FIXTURE_ROOT / cfg.inputs.observations)
    setup = from_experiment(structure, observations, cfg)
    refinement = setup.refinement

    plan = setup.plans.combined
    n_rotations = int(subset_env)
    if not 1 <= n_rotations <= len(plan.orientations):
        raise ValueError(f"DIFFBLOCH_ANCHOR_ROTATIONS must be in 1..{len(plan.orientations)}")
    plan = replace(plan, orientations=plan.orientations[:n_rotations])

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
    result = run_inference(plan, refinement, prepare=prepare, method=cfg.solver.inference)

    assert result.n_evaluated == n_rotations
    if n_rotations == observations.n_rotations:
        assert result.mean_r_obs == pytest.approx(
            EXPECTED_INTEGRATED_MEAN_R_OBS, abs=INTEGRATED_MEAN_R_OBS_TOL
        )
    else:  # subset sanity: proves the coupled pipeline runs and beats the static baseline
        assert result.mean_r_obs < STATIC_BASELINE_R_OBS

"""Real-arithmetic ``matrix_exp`` against the complex path on a real centrosymmetric crystal.

CsPbBr3 (Pbnm, inversion at the origin, no absorption) is the shipped example whose structure matrix
is real symmetric, so ``build_engine`` routes it to the real path. The quartz anchor (P3221) never
reaches that path, so this is its full-pipeline check: the same settled geometry is simulated both
ways and compared per reflection under one global scale, not only by an aggregate R.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from diffBloch.config import load_experiment
from diffBloch.io import read_experimental_data, read_structure
from diffBloch.preprocess import build_orientation_plans, setup_datasets
from diffBloch.preprocess.scoring import build_engine

pytestmark = pytest.mark.e2e

EXPERIMENT = (
    Path(__file__).parents[2] / "examples" / "Colmey_et_al_2026" / "data" / "cspbbr3-no-abs"
)
N_ROTATIONS = 3
# Measured on 3 rotations (3602 reflection-thickness values): after one global scale, the largest
# gap is 3.5e-5 of the strongest reflection and 2.9e-4 relative on any reflection above 1e-3 of it.
# The scale itself is 1.0002 -- a uniform offset, which every objective's fitted scale absorbs.
# Bounds sit ~3x above those measurements.
MAX_GAP = 1e-4
MAX_RELATIVE = 1e-3
MAX_SCALE_OFFSET = 1e-3


def test_real_path_matches_complex_per_reflection_on_cspbbr3() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, _ = load_experiment(EXPERIMENT)
    structure = read_structure(
        EXPERIMENT / cfg.inputs.structure, load_hydrogens=cfg.inputs.load_hydrogens
    )
    record = read_experimental_data(EXPERIMENT / str(cfg.inputs.exp_data))
    setup, (dataset,) = setup_datasets(structure, (record,), cfg)
    refinement = replace(setup, params=setup.params.to(device))
    plan = replace(dataset.plan, orientations=dataset.plan.orientations[:N_ROTATIONS])
    built = build_orientation_plans(
        cfg.blochwave.to_rocking_curve(dataset.integration),
        dataset.mosaicity,
        coupling=cfg.blochwave.to_policy(),
        scoring_selection=cfg.blochwave.to_beam_selection(dataset.integration),
    )(plan)

    real_engine = build_engine(
        built, refinement, method="matrix_exp", absorption=cfg.blochwave.to_absorption()
    )
    assert real_engine.real_operator, "CsPbBr3 (Pbnm, no absorption) must take the real path"
    complex_engine = replace(real_engine, real_operator=False)

    with torch.no_grad():
        real = torch.cat([s.intensities.flatten() for s in real_engine.simulate(refinement.params)])
        direct = torch.cat(
            [s.intensities.flatten() for s in complex_engine.simulate(refinement.params)]
        )
    scale = (real * direct).sum() / (direct * direct).sum()
    residual = (real - scale * direct).abs()
    peak = direct.abs().max()
    assert float(residual.max() / peak) < MAX_GAP
    strong = direct > 1e-3 * peak
    assert float((residual[strong] / direct[strong]).max()) < MAX_RELATIVE
    assert abs(float(scale) - 1.0) < MAX_SCALE_OFFSET

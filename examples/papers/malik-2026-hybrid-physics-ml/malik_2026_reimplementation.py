# %% [markdown]
# # Malik et al. 2026 — CsPbBr3 reimplementation scaffold
#
# Paper: **Hybrid physics-machine learning models for quantitative electron diffraction
# refinements**, Nature Communications (2026), DOI:
# https://doi.org/10.1038/s41467-026-71673-9.
#
# Data/code source cited by the paper: https://doi.org/10.5281/zenodo.18281349.
#
# This notebook is a Jupytext notebook (`py:percent` format). It starts with the CsPbBr3,
# paracetamol, and quartz synthetic/experimental datasets because the Zenodo bundle includes the
# exact CIF/CIF-PETS inputs for those runs. The current public implementation below is a **baseline
# port** into diffBloch's public `ExperimentConfig`; the paper's hybrid physics-ML pieces are marked
# as future sections.

# %%
from __future__ import annotations

import json
import os
from pathlib import Path

from diffBloch.config import load_experiment
from diffBloch.io import read_observations, read_structure

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = tuple(sorted((ROOT / "experiments").glob("*/experiment.yaml")))

print(ROOT)

# %% [markdown]
# ## 1. Verify the copied data and public experiment ports
#
# The experiment directories contain `experiment.yaml` + `experiment.lock`. Their local `data/`
# entries are symlinks to the shared `../../data` directory, so the copied Zenodo data remains
# colocated once.

# %%
for experiment_yaml in EXPERIMENTS:
    experiment_dir = experiment_yaml.parent
    cfg, lock = load_experiment(experiment_dir)
    structure = read_structure(
        experiment_dir / cfg.inputs.structure, load_hydrogens=cfg.inputs.load_hydrogens
    )
    observations = read_observations(experiment_dir / cfg.inputs.observations)
    print(
        json.dumps(
            {
                "experiment": cfg.name,
                "structure": cfg.inputs.structure,
                "observations": cfg.inputs.observations,
                "n_atoms": structure.n_atoms,
                "n_rotations": observations.n_rotations,
                "g_max": cfg.numerics.g_max,
                "g_max_refine": cfg.numerics.g_max_refine,
                "rocking_curve_sampling": cfg.numerics.rocking_curve_sampling,
                "semiangle": cfg.numerics.integration.semiangle,
                "coupling": cfg.preprocess.coupling.model_dump(mode="json")
                if cfg.preprocess.coupling
                else None,
                "structure_lock_bytes": lock.structure.bytes,
                "observations_lock_bytes": lock.observations.bytes,
            },
            indent=2,
            sort_keys=True,
        )
    )

# %% [markdown]
# ## 2. Mapping from the Malik/Zenodo configuration to public diffBloch config
#
# The public ports preserve the key data/preprocess values from the Zenodo submission:
#
# | Malik/Zenodo field | public field |
# | --- | --- |
# | `atoms.data.cif_file_path` | `inputs.structure` |
# | `refinement.data.pets_path` | `inputs.observations` |
# | `bloch.g_max_sf` | `numerics.g_max` |
# | `bloch.g_max_refine` | `numerics.g_max_refine` |
# | `refinement.data.integration_semiangle` | `numerics.integration.semiangle` |
# | `refinement.data.rocking_curve_sampling` | `numerics.rocking_curve_sampling` |
# | `refinement.data.dsg/rsg` | `numerics.dsg/rsg` |
# | implicit private/default tilt-segment union | explicit `preprocess.coupling` |
#
# The original paper workflow also includes a learned thickness-profile model (`thicknessNN`) and a
# hybrid physics-ML refinement loop. Those are not yet public diffBloch components.

# %% [markdown]
# ## 3. Optional: run a public Bloch-physics baseline
#
# A full preprocess can be expensive because it fits orientations/thickness over all rotations.
# Therefore this notebook does not run it by default. To opt in from the command line:
#
# ```bash
# RUN_MALIK_BASELINE=1 MALIK_EXPERIMENT=cspbbr3-synthetic \
#   uv run python examples/papers/malik-2026-hybrid-physics-ml/malik_2026_reimplementation.py
# ```
#
# For interactive use, set `RUN_BASELINE = True` and `BASELINE_EXPERIMENT` below.

# %%
RUN_BASELINE = os.environ.get("RUN_MALIK_BASELINE") == "1"
BASELINE_EXPERIMENT = os.environ.get("MALIK_EXPERIMENT", "cspbbr3-synthetic")

if RUN_BASELINE:
    from diffBloch.app.program import run_experiment

    baseline_dir = ROOT / "experiments" / BASELINE_EXPERIMENT
    result = run_experiment(baseline_dir, checkpoint=True, refresh=False)
    summary = {
        "experiment": BASELINE_EXPERIMENT,
        "n_evaluated": result.n_evaluated,
        "mean_r_obs": result.mean_r_obs,
    }
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    (output_dir / f"{BASELINE_EXPERIMENT}_public_baseline.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
else:
    print("Skipping expensive baseline run. Set RUN_MALIK_BASELINE=1 to run it.")

# %% [markdown]
# ## 4. Future implementation sections
#
# Planned notebook sections for the actual paper reimplementation:
#
# 1. reproduce the paper's synthetic CsPbBr3 baseline refinement setup;
# 2. port or approximate the learned thickness-profile component;
# 3. compare fixed-thickness Bloch physics vs hybrid thickness/profile prediction;
# 4. add the hybrid learned correction objective if/when the public API has the required seam;
# 5. report metrics in the paper's style (`wR`, RMSD where a ground-truth synthetic model is
#    available, and thickness-profile diagnostics).

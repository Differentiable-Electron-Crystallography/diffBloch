# diffBloch

![Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fdifferentiable-electron-crystallography.github.io%2FdiffBloch%2Fcoverage-endpoint.json)
![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?logo=pydantic&logoColor=white)
![uv](https://img.shields.io/badge/uv-DE5FE9?logo=uv&logoColor=white)
![Ruff](https://img.shields.io/badge/Ruff-D7FF64?logo=ruff&logoColor=black)
![mypy](https://img.shields.io/badge/mypy-strict-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-97ca00)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-0969da?logo=sphinx&logoColor=white)](https://differentiable-electron-crystallography.github.io/diffBloch/)

Differentiable Bloch-wave structure refinement for 3D electron diffraction.


📖 **Documentation:** <https://differentiable-electron-crystallography.github.io/diffBloch/> — API reference (Sphinx + furo), rendered from the source on every green `main`.

## Quickstart

Prerequisite: install [uv](https://docs.astral.sh/uv/getting-started/installation/) and
[Git LFS](https://git-lfs.com/) for the bundled `.cif_pets` experimental data and `plan.npz`
checkpoints, then sync the project environment from the repository root:

```bash
git lfs install
git lfs pull
uv sync --dev
```

Run CLI commands through `uv run` unless you have separately installed the `diffbloch` console script
on your shell `PATH`.

```bash
# Run the optimizer and update trainable structural parameters:
uv run diffbloch run refine examples/Colmey_et_al_2026_Acta_Cryst_A/data/quartz-no-abs
```

## For Developers

We welcome collaborations and interested parties may contribute to the codebase. 

Every merge to `main` re-runs the end to end (e2e) physics anchors. These correspond to rapid tests of the codebase's key functionalities and integration. Spefically, the measured mean R_obs for 
the example quartz refinement is expected to remain the same upon changes made to the directory. This Fig. tracks the result of this test upon every merge; the shaded band is the pinned tolerance. A flat line inside the band is the desired
outcome, evidence the physics is reproducible commit over commit. Gaps are commits with no valid
measurement (the committed checkpoint was stale for that commit's recipe, so the fast anchor
could not score it).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://differentiable-electron-crystallography.github.io/diffBloch/anchor-trend-dark.svg">
  <img alt="Mean R_obs of the quartz e2e test for every merge to main" src="https://differentiable-electron-crystallography.github.io/diffBloch/anchor-trend.svg">
</picture>

## Citation

If you use diffBloch in your research, please cite it as:

@misc{diffBloch,
  author  = {Doherty, Tiarnan and Malik, Shreshth and Colmey, Benjamin and Maitland, Iain, and Midgley, Paul},
  title   = {diffBloch},
  version = {0.2.0},
  year = {2026},
  url     = {https://github.com/Differentiable-Electron-Crystallography/diffBloch}
}
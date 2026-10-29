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
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-0969da?logo=sphinx&logoColor=white)](https://diffbloch.com/)

Differentiable Bloch-wave structure refinement for 3D electron diffraction.


📖 **Documentation:** <https://diffbloch.com/> — Usage guide and API reference.

## Quickstart

diffBloch is currently tagged as a release candidate and is under active development; `v0.2.0` will
be the first stable public release. Install [uv](https://docs.astral.sh/uv/getting-started/installation/),
then:

```bash
uv tool install diffBloch
```

Installing using `uv tool` makes diffBloch available as a CLI command on the system, to get started try:

```bash
diffBloch --help
```

Running diffBloch requires an experiment directory consisting of `.cif`, `.cif_pets` and `experiment.yml`, take a look at the [`/examples`](/examples) directory with some easy to run sample experiments, and refer to the documentation on [required inputs](https://diffbloch.com/inputs.html)

If you plan on exploring diffBloch more thoroughly, using features that are under development and have not been published to pypi, or developing/contributing, we recommend working from a git checkout instead.

To run diffBloch locally using the git repo, copy and past this into your terminal:

```bash
git clone https://github.com/Differentiable-Electron-Crystallography/diffBloch.git
cd ./diffBloch
git lfs install
git lfs pull
uv sync --dev
uv run diffBloch --help
```

[Installation](https://differentiable-electron-crystallography.github.io/diffBloch/installation.html)
covers installation beyond quickstart.

## For Developers

We welcome collaborations and interested parties may contribute to the codebase.

Every merge to `main` re-runs the end to end (e2e) physics anchor: forward inference on ten quartz
rotations, each rotation's `R_obs` checked against a committed reference value. This figure plots,
for every merge, the largest deviation of any rotation's `R_obs` from its reference; the dashed
line is the tolerance the test allows. Points on or under the line were within tolerance. Values
come from CI's runner and vary by about 10⁻⁷ between runs and hardware, so a flat line at zero is
not expected. Gaps are merges that recorded no measurement.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://differentiable-electron-crystallography.github.io/diffBloch/anchor-trend-dark.svg">
  <img alt="Largest per-rotation R_obs deviation from reference in the quartz e2e test, for every merge to main" src="https://differentiable-electron-crystallography.github.io/diffBloch/anchor-trend.svg">
</picture>

## Citation

If you use diffBloch in your research, please cite it as:

> Doherty, T., Malik, S., Colmey, B., Maitland, I. & Midgley, P. (2026). diffBloch, version 0.2.0. https://github.com/Differentiable-Electron-Crystallography/diffBloch

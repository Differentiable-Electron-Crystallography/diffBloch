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

Differentiable Bloch-wave structure refinement for rotating-stage 3D electron diffraction.

diffBloch refines crystal structures against continuous-rotation 3DED observations: diffraction
frames collected as the crystal is tilted/rocked through reciprocal space and reduced upstream by
PETS2 into `.cif_pets`. Because the diffraction is dynamical — a multiple-scattering problem —
diffBloch uses a Bloch-wave simulation rather than a kinematical approximation.

The central model is a raw crystal structure plus a settled `Plan`: the structure supplies the
differentiable crystallographic parameters, while the `Plan` supplies the reusable geometry/data
scaffold around them. The refinement engine iteratively minimizes a scalar R-loss so simulated and
observed diffraction intensities agree.

📖 **Documentation:** <https://differentiable-electron-crystallography.github.io/diffBloch/> — API reference (Sphinx + furo), rendered from the source on every green `main`.

## Quickstart

Prerequisite: install [uv](https://docs.astral.sh/uv/getting-started/installation/) and
[Git LFS](https://git-lfs.com/) for the bundled `.cif_pets` observations and `plan.npz`
checkpoints, then sync the project environment from the repository root:

```bash
git lfs install
git lfs pull
uv sync --dev
```

Run CLI commands through `uv run` unless you have separately installed the `diffbloch` console script
on your shell `PATH`.

```bash
# Validate a config before launching a longer job:
uv run diffbloch validate examples/experiments/quartz-checkpoint/experiment.yaml

# Simulate and score without changing parameters:
uv run diffbloch run infer examples/experiments/quartz-checkpoint

# Run the optimizer and update trainable structural parameters:
uv run diffbloch run refine examples/experiments/quartz

# Start refinement faster from the bundled quartz preprocessing checkpoint:
uv run diffbloch run refine examples/experiments/quartz-checkpoint
```

Use `infer` to score a settled `Plan` without changing parameters; use `refine` to optimize
trainable structural parameters. Checkpointed examples include `plan.npz` + `plan.lock`, so they can
reuse expensive preprocessing instead of rebuilding the `Plan` from scratch.

See `examples/experiments/quartz/README.md` for the worked example and its expected residual.

## Examples

- `examples/experiments` contains runnable example experiments for small and large compounds,
  including checkpointed runs that start from a committed `Plan`.

## Anchor trend

Every merge to `main` re-runs the e2e physics anchors. The plot tracks the measured mean R_obs of
the coupled quartz anchor (the checkpointed 99-rotation scoring in `tests/e2e/test_anchor.py`) for
every merge; the shaded band is the pinned tolerance. A flat line inside the band is the desired
outcome — evidence the physics is reproducible commit over commit. Gaps are commits with no valid
measurement (the committed checkpoint was stale for that commit's recipe, so the fast anchor
could not score it).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://differentiable-electron-crystallography.github.io/diffBloch/anchor-trend-dark.svg">
  <img alt="Mean R_obs of the coupled quartz anchor for every merge to main" src="https://differentiable-electron-crystallography.github.io/diffBloch/anchor-trend.svg">
</picture>

## Layout

```
src/diffBloch/    the library (config/, io/, core/, engine, preprocess/, app/)
examples/         runnable experiments and paper-style composition examples
tests/unit/       fast per-kernel tests
tests/e2e/        characterization anchors
docs/             docs site (Sphinx + furo, MyST)
```

## Funding

This work is funded by [Schmidt Sciences](https://www.schmidtsciences.org/) and developed with
the [Scientific Software Engineering Center (SSEC)](https://ai.jhu.edu/ssec/) at Johns Hopkins
University.

## Layers

| Layer | Role | Guide |
|---|---|---|
| IO | Parse CIF/PETS files into validated typed records. | [Inputs](https://differentiable-electron-crystallography.github.io/diffBloch/inputs.html) |
| Config | Validate experiment settings and lock input/checkpoint identity. | [Inputs](https://differentiable-electron-crystallography.github.io/diffBloch/inputs.html), [Reproducibility](https://differentiable-electron-crystallography.github.io/diffBloch/reproducibility.html) |
| Preprocess | Build and improve the immutable `Plan` the simulator consumes. | [Preprocessing](https://differentiable-electron-crystallography.github.io/diffBloch/preprocessing.html) |
| Core | Repeatable crystallographic and Bloch-wave numerical kernels. | [Architecture](https://differentiable-electron-crystallography.github.io/diffBloch/architecture.html) |
| Params | Differentiable structural parameters and physical constraints. | [Refinement](https://differentiable-electron-crystallography.github.io/diffBloch/refinement.html) |
| Engine | Combine `Plan` + parameters, simulate, score, and refine. | [Refinement](https://differentiable-electron-crystallography.github.io/diffBloch/refinement.html) |
| Observability | Track progress and debug refinements with typed events/loggers. | [Observability](https://differentiable-electron-crystallography.github.io/diffBloch/observability-guide.html) |
| App | CLI and default orchestration around the reusable API. | [Examples](https://differentiable-electron-crystallography.github.io/diffBloch/examples.html) |

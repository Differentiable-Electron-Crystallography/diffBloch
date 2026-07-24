# diffBloch

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

Prerequisite: install [uv](https://docs.astral.sh/uv/getting-started/installation/), then sync the
project environment from the repository root:

```bash
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
- `examples/papers` contains paper implementations that demonstrate doing science with diffBloch via
  richer Python API composition.

## Layout

```
src/diffBloch/    the library (config/, io/, core/, engine, preprocess/, app/)
examples/         runnable experiments and paper-style composition examples
tests/unit/       fast per-kernel tests
tests/e2e/        characterization anchors
docs/             docs site (Sphinx + furo, MyST)
```

## Layers

| Layer | Role | Guide |
|---|---|---|
| IO | Parse CIF/PETS files into validated typed records. | [Inputs](https://differentiable-electron-crystallography.github.io/diffBloch/inputs.html) |
| Config | Validate experiment settings and lock input/checkpoint identity. | [Inputs](https://differentiable-electron-crystallography.github.io/diffBloch/inputs.html), [Reproducibility](https://differentiable-electron-crystallography.github.io/diffBloch/reproducibility.html) |
| Preprocess | Build and improve the immutable `Plan` the simulator consumes. | [Preprocessing](https://differentiable-electron-crystallography.github.io/diffBloch/preprocessing.html) |
| Core | Deterministic crystallographic and Bloch-wave numerical kernels. | [Architecture](https://differentiable-electron-crystallography.github.io/diffBloch/architecture.html) |
| Params | Differentiable structural parameters and physical constraints. | [Refinement](https://differentiable-electron-crystallography.github.io/diffBloch/refinement.html) |
| Engine | Combine `Plan` + parameters, simulate, score, and refine. | [Refinement](https://differentiable-electron-crystallography.github.io/diffBloch/refinement.html) |
| Observability | Track progress and debug refinements with typed events/loggers. | [Observability](https://differentiable-electron-crystallography.github.io/diffBloch/observability-guide.html) |
| App | CLI and default orchestration around the reusable API. | [Examples](https://differentiable-electron-crystallography.github.io/diffBloch/examples.html) |

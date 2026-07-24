# diffBloch

Differentiable Bloch-wave electron-diffraction structure refinement: a small differentiable map from
structural parameters to a scalar R-loss, minimised by gradient descent so that simulated and observed
diffraction intensities agree.

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

See `examples/experiments/quartz/README.md` for the worked example and its expected residual.

## Layout

```
src/diffBloch/    the library (config/, io/, core/, engine, app/ — added stage by stage)
examples/         runnable experiment directories (`examples/experiments/quartz{,-checkpoint}`)
tests/unit/       fast per-kernel tests
tests/e2e/        characterization anchors (opt-in: `just test-e2e`)
docs/             Docs (Sphinx + furo, MyST); `just docs`
```

## Layers

| Layer | Role |
|---|---|
| IO | Parse CIF/PETS files into validated typed records. |
| Config | Validate experiment settings and lock input/checkpoint identity. |
| Preprocess | Build and improve the immutable `Plan` the simulator consumes. |
| Core | Deterministic crystallographic and Bloch-wave numerical kernels. |
| Params | Differentiable structural parameters and physical constraints. |
| Engine | Combine `Plan` + parameters, simulate, score, and refine. |
| Observability | Emit typed events without coupling the core to logger backends. |
| App | CLI and default orchestration around the reusable API. |

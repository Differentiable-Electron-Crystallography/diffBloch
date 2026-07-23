# diffBloch

Differentiable Bloch-wave electron-diffraction structure refinement: a small differentiable map from
structural parameters to a scalar R-loss, minimised by gradient descent so that simulated and observed
diffraction intensities agree.

> **2.0 rewrite in progress.** This package is being built from scratch, porting from the research
> codebase stepwise — one discrete, tested commit per stage.

📖 **Documentation:** <https://differentiable-electron-crystallography.github.io/diffBloch/> — API reference (mkdocstrings), rendered from the source on every green `main`.

## Quickstart

```bash
just install      # uv sync --dev + pre-commit install
just check        # lint + typecheck + unit tests
just docs-serve   # live API docs
just anchor       # the quartz physics anchors (opt-in e2e)

# Score the worked quartz example (faithful coupled recipe, mean R_obs = 0.0506):
diffbloch run infer examples/experiments/quartz-checkpoint   # instant — ships a frozen checkpoint
diffbloch run infer examples/experiments/quartz              # ~6–16 min — fits from scratch
```

See `examples/experiments/quartz/README.md` for the worked example and its expected residual.

## Layout

```
src/diffBloch/    the library (config/, io/, core/, engine, app/ — added stage by stage)
examples/         runnable experiment directories (`examples/experiments/quartz{,-checkpoint}`)
tests/unit/       fast per-kernel tests
tests/e2e/        characterization anchors (opt-in: `just test-e2e`)
docs/             API docs (mkdocstrings); `just docs`
```

## Principles

- **Functional core, imperative shell** — pure tensor-in/tensor-out kernels; `nn.Module` only holds
  parameters.
- **Pydantic config, validated at the boundary** — no Hydra.
- **gemmi** (blessed CIF/PETS reader) **+ diffpy** (symmetry constraints); nothing in `core/` imports
  a parser.
- **Swappable solver** (`matrix_exp` + `bloch_eigen`), **typed products**, **plans not caches**,
  **effects at the edges** (pluggable `Logger`; no vendor SDK in core).
- **Characterization tests first** — the single-rotation quartz anchor pins the physics; it stays
  green at every commit.

# diffBloch

Differentiable Bloch-wave electron-diffraction structure refinement: a small differentiable map from
structural parameters to a scalar R-loss, minimised by gradient descent so that simulated and observed
diffraction intensities agree.

> **2.0 rewrite in progress.** This package is being built from scratch, porting from the research
> codebase stepwise — one discrete, tested commit per stage.

## Quickstart

```bash
just install      # uv sync --dev + pre-commit install
just check        # lint + typecheck + unit tests
just docs-serve   # live API docs
just anchor       # the single-rotation quartz physics anchor (once ported)
```

## Layout

```
src/diffBloch/    the library (config/, io/, core/, engine, app/ — added stage by stage)
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

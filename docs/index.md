# diffBloch

Differentiable Bloch-wave electron-diffraction structure refinement.

The valuable core is a small **differentiable map from a handful of structural parameters to a
scalar R-loss**, minimised by gradient descent so that simulated and observed diffraction
intensities agree. Everything else — parsing, config, logging, orchestration — exists to make that
map inspectable, reproducible, and safe to evolve, and lives *around* the core, never inside it.

## Status

The 2.0 package is a from-scratch, stepwise port of the research codebase, now feature-complete
through the full inference recipe: config + IO at the boundary, the differentiable Bloch-wave core,
the refinement engine, the composable `Plan → Plan` preprocess pipeline (beam selection,
orientation/thickness fits, rocking-curve integration, mosaicity, convergence sweeps), typed
observability with pluggable logger backends, and the thin `diffbloch` CLI. The physics is pinned
end-to-end by the opt-in quartz anchor e2e tests.

## Quickstart

```bash
diffbloch validate experiment.yaml     # validate an experiment config
diffbloch run infer <experiment_dir>   # score every rotation (add --console / --csv PATH)
diffbloch run pack <run_dir>           # export a run directory (zip/tar/BagIt/RO-Crate)
```

Python users compose their own preprocess with the public API (`from_experiment` + the `preprocess`
steps + `run_inference`); the CLI is the friendly default runner, not the only path.

## API

API docs are generated from docstrings and type signatures via `mkdocstrings`. The navigation
covers every public layer: **Config**, **IO**, **Core**, **Params**, **Specs**, **Engine**,
**Preprocess**, **Observability**, and **App**.

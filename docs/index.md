# diffBloch

Differentiable Bloch-wave electron-diffraction structure refinement.

The valuable core is a small **differentiable map from a handful of structural parameters to a
scalar R-loss**, minimised by gradient descent so that simulated and observed diffraction
intensities agree. Everything else — parsing, config, logging, orchestration — exists to make that
map inspectable, reproducible, and safe to evolve, and lives *around* the core, never inside it.

## Status

The 2.0 package is being built from scratch, ported from the research codebase **stepwise** (one
discrete, tested commit per stage). The architecture is documented in the synthesis notebook
(`notebooks/iain/principled_refactor_synthesis.ipynb`); the staged plan is in
[`ROADMAP.md`](https://github.com/Differentiable-Electron-Crystallography/diffBloch/blob/main/ROADMAP.md).

## API

API docs are generated from docstrings and type signatures via `mkdocstrings`. See **Config** in the
navigation; more modules appear here as each stage lands.

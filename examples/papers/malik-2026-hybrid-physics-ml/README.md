# Malik et al. 2026 — hybrid physics-machine learning electron-diffraction refinements

Paper:

- Shreshth A. Malik, Tiarnan A. S. Doherty, Benjamin Colmey, Stephen J. Roberts, Yarin Gal,
  Paul A. Midgley, **Hybrid physics-machine learning models for quantitative electron diffraction
  refinements**, *Nature Communications* (2026), DOI: <https://doi.org/10.1038/s41467-026-71673-9>.

Data/code source:

- Zenodo record cited by the paper: <https://doi.org/10.5281/zenodo.18281349>.
- Local paper PDF source, outside the repo because of the large-file hook:
  `../papers/Malik_2026_HybridPhysicsML_ElectronDiffraction_NatCommun.pdf`.
- The files in `data/` were copied from `../papers/diffBloch-zenodo-submitted/data`.

## Contents

```text
malik_2026_reimplementation.py  # Jupytext notebook
data/                           # copied Zenodo compound data
experiments/                    # public diffBloch experiment ports
```

The copied data includes the synthetic and experimental inputs referenced by the paper for:

```text
data/cspbbr3/
data/paracetamol_cr/
data/quartz/
```

Each compound has public diffBloch experiment ports under `experiments/`:

```text
experiments/cspbbr3-synthetic/
experiments/cspbbr3-experimental/
experiments/paracetamol-synthetic/
experiments/paracetamol-experimental/
experiments/quartz-synthetic/
experiments/quartz-experimental/
```

## Running the notebook

The notebook is stored as a Jupytext paired `.py` file:

```bash
uv run jupytext --to ipynb examples/papers/malik-2026-hybrid-physics-ml/malik_2026_reimplementation.py
uv run jupyter lab examples/papers/malik-2026-hybrid-physics-ml/malik_2026_reimplementation.ipynb
```

If `jupytext` or `jupyter` is not installed in the current environment, run the `.py` notebook as a
plain script for the lightweight metadata/config checks:

```bash
uv run python examples/papers/malik-2026-hybrid-physics-ml/malik_2026_reimplementation.py
```

## Running the public diffBloch baseline configs

The experiment directories contain `experiment.yaml` + `experiment.lock` and symlink their local
`data/` entry to the shared `../../data` directory. From the repository root:

```bash
for exp in examples/papers/malik-2026-hybrid-physics-ml/experiments/*/experiment.yaml; do
  uv run diffbloch validate "$exp"
done
```

A full preprocess/refinement-style run may be expensive. Start with config/data inspection in the
notebook before launching:

```bash
uv run diffbloch run preprocess examples/papers/malik-2026-hybrid-physics-ml/experiments/cspbbr3-synthetic --device cuda
uv run diffbloch run infer examples/papers/malik-2026-hybrid-physics-ml/experiments/cspbbr3-synthetic --device cuda
```

## Current implementation status

This is the first public scaffold for the Malik reimplementation:

1. colocate the paper and Zenodo data for CsPbBr3, paracetamol, and quartz;
2. port the synthetic and experimental configs into the current public `ExperimentConfig`;
3. provide a Jupytext notebook for data inspection and a public Bloch-physics baseline.

The actual hybrid physics-ML components from the paper are not implemented yet. The notebook marks
those gaps explicitly: thickness-profile prediction, learned correction components, and the paper's
training loop are future sections.

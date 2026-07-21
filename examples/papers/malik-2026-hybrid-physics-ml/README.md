# Malik et al. 2026 — quartz ThicknessNN public API example

Paper:

- Shreshth A. Malik, Tiarnan A. S. Doherty, Benjamin Colmey, Stephen J. Roberts, Yarin Gal,
  Paul A. Midgley, **Hybrid physics-machine learning models for quantitative electron diffraction
  refinements**, *Nature Communications* (2026), DOI: <https://doi.org/10.1038/s41467-026-71673-9>.

This v1 example intentionally keeps the story small: it starts with **quartz synthetic** only and
runs end to end as a simple declarative public `diffBloch` script. It follows the same shape as
`diffBloch.app.program`: load config/data, run the configured preprocess pipeline to settle a Plan,
build a refinement engine, compose the structure component plus `ApparentThicknessNN`, and run a
CPU-safe one-step joint refinement evaluation.

## Contents

```text
malik_2026_reimplementation.py  # Jupytext py:percent notebook/script
experiments/quartz-synthetic/   # public diffBloch experiment port used by v1
data/quartz/                    # copied quartz data
```

Other copied Malik/Zenodo datasets may be present in this directory, but they are not part of the
v1 example path.

## Running the example

As a plain script from the repository root:

```bash
uv run python examples/papers/malik-2026-hybrid-physics-ml/malik_2026_reimplementation.py
```

As a Jupyter notebook:

```bash
uv add --dev jupyterlab jupytext ipykernel
uv run python -m ipykernel install --user --name diffbloch --display-name "diffBloch"
uv run jupytext --to ipynb examples/papers/malik-2026-hybrid-physics-ml/malik_2026_reimplementation.py
uv run jupyter lab examples/papers/malik-2026-hybrid-physics-ml/malik_2026_reimplementation.ipynb
```

In JupyterLab, select:

```text
Kernel -> Change Kernel -> diffBloch
```

## Public API shape

The notebook presents the refinement as ordinary Python composition:

```python
cfg, lock = load_experiment(EXPERIMENT_DIR)
structure = read_structure(EXPERIMENT_DIR / cfg.inputs.structure)
observations = read_observations(EXPERIMENT_DIR / cfg.inputs.observations)
setup = from_experiment(structure, observations, cfg)
plan = preprocess_experiment(EXPERIMENT_DIR, checkpoint=True, refresh=False, device=device)
plan = keep_finite_loss_quartz_frames(plan, setup, cfg)
engine = build_engine(plan, setup.refinement, loss=cfg.refinement.objective.to_loss())

trainable = TrainableSpec(positions=AtomSelection.all())
trainable, constraints = with_hydrogen_riding(structure, trainable)  # optional hard constraint
penalties = (perceive_bond_length_penalty(...),)                    # optional soft term

thickness_nn = ApparentThicknessNN(bounds=ThicknessBounds(100.0, 2000.0))
component_params = {
    thickness_nn.key: thickness_nn.initial_params(
        dtype=setup.refinement.params.asu_positions.dtype,
        device=device,
        initial_thickness=mean_plan_thickness(engine.orientations),
    )
}

model = build_refinement_model(
    initial=setup.refinement.params,
    constraints=constraints,
    components=(thickness_nn,),
    component_params=component_params,
)
problem = build_refinement_problem(penalties=penalties)
result = run_refinement_model(engine, model, problem, trainable=trainable, steps=1, lr=1e-5)
```

Taxonomy:

- `StructureComponent` is built into `RefinementModel` from `initial=...` and `constraints=...`.
- `components` provide values to the forward model. Here, `ApparentThicknessNN` provides apparent
  thickness per orientation.
- `constraints` are hard structure transforms. H-riding belongs on the structure component.
- `penalties` are soft additive objective terms. Bond/angle/plane priors belong in the objective
  problem.

Omit `initial_thickness` for de novo bounded-thickness initialization at the bounds midpoint:

```python
component_params = {
    thickness_nn.key: thickness_nn.initial_params(dtype=torch.float64, device=device)
}
```

## Outputs

The script transparently drops quartz rotations whose initial post-preprocess diffraction loss is
not finite; the current v1 retains 79 of 99 observed rotations. It writes lightweight diagnostics to:

```text
examples/papers/malik-2026-hybrid-physics-ml/outputs/quartz-synthetic/
```

including:

```text
thickness_nn_summary.json
thickness_nn_profile.json
```

## Next step after v1

Keep this same declarative API shape while adding Malik-style reporting and broadening the dataset
matrix.

Deferred for full paper parity:

- optimizer parameter groups / separate learning rates for structure, ADPs, and ThicknessNN;
- sigma output and stochastic thickness sampling;
- full train/validation reporting in the Malik style;
- full six-dataset evaluation and synthetic RMSD comparison.

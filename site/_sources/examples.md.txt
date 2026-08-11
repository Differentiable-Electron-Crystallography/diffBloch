# Examples

The repository ships runnable experiment directories under `examples/`. Each holds an
`experiment.yaml` plus its structure `.cif` and `.cif_pets` data -- see
[Inputs](inputs.md) for the directory layout.

```bash
uv run diffbloch run preprocess <experiment_dir>
uv run diffbloch run refine <experiment_dir>
```

## Colmey et al. (2026)

[`examples/Colmey_et_al_2026`](https://github.com/Differentiable-Electron-Crystallography/diffBloch/tree/main/examples/Colmey_et_al_2026)
holds six runnable experiment configs reproducing the three-material (CsPbBr₃, alpha-quartz,
borane), elastic-vs-absorptive refinement comparison from Colmey *et al.* (2026), *The role of
absorption in three-dimensional electron diffraction dynamical structure refinement*, submitted to
Acta Crystallographica A ([arXiv:2602.08935](https://arxiv.org/abs/2602.08935)):

```bash
uv run diffbloch run preprocess examples/Colmey_et_al_2026/data/quartz-absorption
uv run diffbloch run refine examples/Colmey_et_al_2026/data/quartz-absorption
```

Swap in any of `quartz-no-abs`, `cspbbr3-no-abs`, `cspbbr3-absorption`, `borane-no-abs`,
`borane-absorption` for the other materials/absorption settings. See that directory's own
`README.md` for the published-vs-reproduced R-factor table and data provenance; diffBloch has
changed since the runs behind the paper's own numbers, so this reproduces the same experiments
rather than bit-for-bit values.

## Python API example

```python
from diffBloch.app import run_experiment

result = run_experiment("<experiment_dir>")
print(result.n_evaluated, result.mean_r_obs)
```

# Examples

Runnable examples are provided in `examples/`. Each experiment directory contains an
`experiment.yaml`, a starting structure `.cif`, and experimental `.cif_pets` data.

## Colmey et al. (2026)

The examples in
[`examples/Colmey_et_al_2026`](https://github.com/Differentiable-Electron-Crystallography/diffBloch/tree/main/examples/Colmey_et_al_2026)
reproduce the refinements from Colmey *et al.* (2026), *The role of absorption in
three-dimensional electron diffraction dynamical structure refinement*
([arXiv:2602.08935](https://arxiv.org/abs/2602.08935)).

Six experiments compare refinements with and without absorption:

| Material | Without absorption | With absorption |
|---|---|---|
| Alpha-quartz | `quartz-no-abs` | `quartz-absorption` |
| Borane | `borane-no-abs` | `borane-absorption` |
| CsPbBr₃ | `cspbbr3-no-abs` | `cspbbr3-absorption` |

The experimental data were originally published by Suresh *et al.* (2024), *Ionisation of atoms
determined by kappa refinement against 3D electron diffraction data*
([Nature Communications 15, 9066](https://doi.org/10.1038/s41467-024-53448-2)).

## Running an example

For example, run the absorptive quartz refinement with:

```bash
uv run diffbloch preprocess examples/Colmey_et_al_2026/data/quartz-absorption
uv run diffbloch refine examples/Colmey_et_al_2026/data/quartz-absorption
```

Replace `quartz-absorption` with any directory listed above. Add `--refresh` to the `refine`
command to rebuild preprocessing before refinement:

```bash
uv run diffbloch refine examples/Colmey_et_al_2026/data/quartz-absorption --refresh
```

## Results

Each successful refinement writes a refined CIF in its experiment directory and promotes a canonical
JSONL report under `reproducibility/`.

The current code has changed since the calculations reported in the paper, so the examples repeat
the same experiments but are not expected to reproduce the published values exactly. The original
published values and full data provenance are given in the example directory's `README.md`.

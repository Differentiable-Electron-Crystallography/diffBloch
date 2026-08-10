# Examples

The repository ships six runnable experiment directories under
`examples/Colmey_et_al_2026_Acta_Cryst_A/data/`. They reproduce the three materials refined in
Colmey et al. (2026), each in an elastic and an absorptive configuration — so the pair is also the
worked demonstration of what `blochwave.absorption` changes:

```text
data/quartz-absorption/    data/quartz-no-abs/
data/cspbbr3-absorption/   data/cspbbr3-no-abs/
data/borane-absorption/    data/borane-no-abs/
```

Each directory holds its own `experiment.yaml`, the starting `.cif`, the PETS `.cif_pets`
experimental data, and the committed outputs of a reference run (`refined_structure.cif`,
`refinement_report.txt`, and the `reproducibility/` locks and parameter snapshots) — so you can read
the expected result before spending the compute. See that directory's `README.md` for the paper's
published residuals alongside this repository's.

## Quick examples

### Quartz, elastic

The smallest cell (~113 Å³) and the cheapest run, so start here.

```bash
uv run diffbloch run refine examples/Colmey_et_al_2026_Acta_Cryst_A/data/quartz-no-abs
```

Each epoch reports `wR2` and `R_obs` with the rotation count each mean was taken over, the
diffraction loss, and any composed penalty. On completion the run writes `refined_structure.cif`,
the raw parameter snapshot, and `refinement_report.txt` beside the experiment.

### Quartz, absorptive

The same structure and data with `blochwave.absorption: true`, which is the comparison the paper
makes. Diffing the two `refinement_report.txt` files is the quickest way to see the effect.

```bash
uv run diffbloch run refine examples/Colmey_et_al_2026_Acta_Cryst_A/data/quartz-absorption
```

### CsPbBr3 on an accelerator

The high-Z case, where absorption matters most and the eigensolve is dear enough to want a GPU.

```bash
uv run diffbloch run refine examples/Colmey_et_al_2026_Acta_Cryst_A/data/cspbbr3-absorption --device cuda
```

### Watching a run

Add `--tui` (the `diffBloch[tui]` extra) to replace the scrolling log with a live dashboard: the
per-stage beam and reflection survival counts, the declared objective, a progress bar per phase, and
the epoch table. `--csv PATH` writes the same event stream to a long-format log, and composes with
either console mode.

## Cost, and why these runs are not instant

**No example ships a preprocess checkpoint.** Every run above settles the `Plan` from raw inputs
first — the orientation and thickness fits — before refinement starts, which is the dominant cost.
A run writes `plan.npz` + `plan.lock` into the experiment's `reproducibility/` when it finishes, so
a *second* run of the same directory reuses it and starts at refinement, provided the inputs,
config, code release, and recipe all still match.

These runs are also long by default: `refinement.steps` is 40 unless the experiment overrides it.
Copy a directory and lower `steps` if you want a quick look rather than a reproduction — there is
deliberately no CLI override, because it would make the recorded artifact disagree with the run.

## Catalog

The six directories span the axes that matter for cost and behaviour: unit-cell size (which sets the
beam count, and with it the O(N³) eigensolve cost and whether the large-cell fork routes past it —
see [Preprocessing](preprocessing.md#routing-on-cell-size)), atomic number (how much absorption
changes the answer), and structural complexity (whether hydrogens and their riding-model treatment
are in play — see [Refinement](refinement.md)).

| Example | What it demonstrates |
|---|---|
| `data/quartz-no-abs`, `data/quartz-absorption` | Small cell (~113 Å³), no hydrogens. The cheapest end-to-end run, and the structure the CI physics anchor is built from. |
| `data/cspbbr3-no-abs`, `data/cspbbr3-absorption` | High-Z, where absorption gives the paper's clearest improvement in R_obs. Benefits most from a GPU. |
| `data/borane-no-abs`, `data/borane-absorption` | An organic molecule with hydrogens, so the riding-model constraint is in play. |

## Python API example

```python
from diffBloch.app import run_experiment

result = run_experiment("examples/Colmey_et_al_2026_Acta_Cryst_A/data/quartz-no-abs")
print(result.n_evaluated, result.mean_r_obs)
```

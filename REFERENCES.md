# References & credits

diffBloch is a from-scratch port of a research codebase and stands on published science and several
open-source projects. This file records the scientific references, vendored reference data, the
software we depend on / extract from / verify against, **and the codebases we examined for their
approaches** (even where no code is used). It is a **living document** — each stage that introduces a
new method, dataset, dependency, or studied design adds its credit here.

## Origin

- **diffBloch (research codebase)** — the private predecessor this package is ported from, stage by
  stage. Physics conventions and reference behaviour derive from it.
  <https://github.com/Differentiable-Electron-Crystallography/diffBloch>

## Scientific references

- **Electron scattering factors (Lobato parametrization).**
  Lobato, I. & Van Dyck, D. (2014). *An accurate parameterization for scattering factors, electron
  densities and electrostatic potentials for neutral atoms that obey all physical constraints.*
  **Acta Crystallographica A70(6), 636–649.** DOI: [10.1107/S205327331401643X](https://doi.org/10.1107/S205327331401643X)
  — open access ([PDF](https://nano.uantwerpen.be/nanorefs/pdfs/OA_10.1107_S205327331401643X.pdf)).
  This is the **canonical source** of the form-factor coefficients used in `core/scattering`. Five
  hydrogenic basis functions fit to first-principles scattering factors for 103 neutral atoms.

_(Dynamical-diffraction / Bloch-wave theory and PETS references will be added when stages 7–8 land
the solver and observation model.)_

## Vendored reference data

- **`src/diffBloch/core/data/lobato.json`** — Lobato–Van Dyck (2014) parametrization coefficients
  `{element: [[a₁…a₅], [b₁…b₅]]}` for 103 elements. These are **published reference numbers from the
  paper above** (scientific data, not third-party IP). They were extracted via **abTEM**'s
  `lobato.json` and cross-checked against **diffsims**' independent tabulation; our native functional
  form reproduces abTEM's values to ~1e-5. Credit for the numbers belongs to
  [Lobato & Van Dyck 2014](https://doi.org/10.1107/S205327331401643X); abTEM
  (<https://github.com/abTEM/abTEM>) and diffsims (<https://github.com/pyxem/diffsims>) are the
  extraction/verification path, not the provenance authority.

## Software we depend on, extract from, or verify against

### Runtime dependencies
- **gemmi** — CIF/mmCIF and PETS parsing, unit-cell handling, and space-group symmetry operations
  (the blessed parser behind `io/`). Wojdyr, M. (2022), *GEMMI: A library for structural biology*,
  Journal of Open Source Software 7(73), 4200. DOI: [10.21105/joss.04200](https://doi.org/10.21105/joss.04200)
  · <https://github.com/project-gemmi/gemmi>
- **PyTorch** — differentiable tensor backend (`core/` constraints, ADP, symmetry expansion onward).
  <https://github.com/pytorch/pytorch>
- **NumPy** — array backend for the pure planning/geometry helpers. <https://github.com/numpy/numpy>
- **pydantic** — boundary validation for config and IO records. <https://github.com/pydantic/pydantic>
- **PyYAML** — experiment/config and lock parsing. <https://github.com/yaml/pyyaml>

### Planned dependencies (seams already in place)
- **diffpy.structure** — special-position and ADP symmetry-constraint expansion, behind the
  `io.symmetry_setup` seam (lands with the constraints/symmetry stage). <https://github.com/diffpy/diffpy.structure>

### Development & verification oracles (not runtime dependencies)
- **abTEM** — Madsen, J. & Susi, T. (2021), *The abTEM code: transmission electron microscopy from
  first principles*, Open Research Europe 1:24. <https://open-research-europe.ec.europa.eu/articles/1-24>
  — used **only** as a test oracle for form-factor values and as the extraction path for the Lobato
  table. It is deliberately **not** a runtime dependency (its numba/dask/llvmlite tree is too heavy
  for the lean core).
- **diffsims** (pyxem) — independent tabulation of the Lobato coefficients
  (`diffsims.utils.lobato_scattering_params.ATOMIC_SCATTERING_PARAMS_LOBATO`), used to cross-check our
  vendored data against a second source. <https://github.com/pyxem/diffsims>
- **MULTEM** — Ivan Lobato's own GPU multislice code; the author's reference implementation of the
  parametrization. Cited for provenance; not used directly. <https://github.com/Ivanlh20/multem>

## Codebases examined for approaches (not used directly)

Studied while designing the 2.0 architecture (see `notebooks/iain/principled_refactor_synthesis.ipynb`)
and while choosing methods. We adopted ideas, not code.

- **mythos** — sibling differentiable-science codebase (JAX/JAX-MD). Source of the
  `Simulator → Observable → Objective → Optimizer` lifecycle, explicit `OptimizerState` threading,
  readiness/`needs_update` invalidation, typed product objects, the `Logger`/`NullLogger` protocol,
  and engine-agnostic `SchedulerHints`. <https://github.com/mythos-bio/mythos>
- **DIALS** — `Target / Parameterisation / Engine` refinement decomposition, reflection-table flags,
  and "normalize all input at the boundary". <https://github.com/dials/dials>
- **cctbx** — `miller.array` (symmetry-bound reflection data), special-position/ADP symmetry, and the
  u\*-canonical ADP conventions we follow. <https://github.com/cctbx/cctbx_project>
- **NumPyro** — the `biject_to` constraint→transform registry (our `constrain` seam) and the
  effect-handler-as-edges principle. <https://github.com/pyro-ppl/numpyro>
- **Diffrax** — solver-as-strategy (`Term / Solver / Adjoint`) and implicit differentiation for the
  eigen path, informing the `BlochSystem`/`Propagator` design. <https://github.com/patrick-kidger/diffrax>
- **JAX-MD** — precompute-once-with-an-overflow-signal (the `BeamPlan`/plan pattern) and
  `(init, apply)` immutable-state style. <https://github.com/jax-md/jax-md>
- **Equinox** — params-as-frozen-pytree and static/dynamic leaf filtering, informing
  `RefinableParams`. <https://github.com/patrick-kidger/equinox>
- **Alternative scattering parametrizations examined** (we chose Lobato):
  **Kirkland** (Dirac–Fock Gaussian/Lorentzian fit, *Advanced Computing in Electron Microscopy*; code
  at <https://sourceforge.net/projects/computem/>), **Peng et al.** (Peng, L.-M. et al. (1996),
  *Acta Cryst.* A52, 257, DOI: [10.1107/S0108767395014371](https://doi.org/10.1107/S0108767395014371))
  — both also tabulated by abTEM — and the **ReciPro** crystallographic suite
  (<https://github.com/seto77/ReciPro>).

## Conventions credited
- Reciprocal-cell convention (`pinv(cell).T`) follows the ASE-compatible helper from the research
  codebase; ADP frame conventions follow standard crystallographic (cctbx-compatible) definitions.

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

- **Dynamical electron diffraction (Bloch-wave method).**
  Spence, J. C. H. & Zuo, J. M. (1992). *Electron Microdiffraction.* Plenum Press, New York.
  The canonical reference for the Bloch-wave formulation diffBloch ports: the structure matrix `A`,
  the diagonalisation/`matrix_exp` propagation of the wavefunction through thickness, the diagonal
  excitation errors `Sg` (the "Spence and Zuo method" named in the private `excitation_errors`
  docstring), and the `Mii` Lorentz/obliquity factors. The relativistic electron-optics relations
  (`energy → wavelength`, interaction parameter `σ`) are standard and follow the abTEM
  implementation noted below.

- **Refinement loss / agreement metrics.**
  The Bragg R(obs) factor `R = Σ|√I_obs − √I_calc| / Σ√I_obs` (over reflections with `I_obs > 3σ`)
  is the standard crystallographic agreement index (e.g. Spence & Zuo above; Giacovazzo, *Fundamentals
  of Crystallography*). The **weighted R2** `w_rbragg` and its weighting scheme (instability factor
  `μ`, weak-reflection floor) follow the SI of Klar, P. B. et al. (2023). *Accurate structure models
  and absolute configuration determination using dynamical effects in continuous-rotation 3D electron
  diffraction data.* **Nature Chemistry 15, 848–855.** DOI:
  [10.1038/s41557-023-01186-1](https://doi.org/10.1038/s41557-023-01186-1). Loss bodies are ported
  from the private `diffBloch/metrics.py`.

_(PETS / observation-model references will be added when stage 9+ lands the observation model.)_

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
  for the lean core). The dynamical stage additionally reimplements abTEM's relativistic
  electron-optics helpers natively — `energy2wavelength` and `energy2sigma` (`abtem.core.energy`)
  and the interaction constant `kappa` (`abtem.core.constants`) — and follows the structure-matrix
  conventions of `abtem.bloch`. The private predecessor imported these directly from abTEM; the
  port re-derives them from the underlying physical constants and verifies against published values
  (and abTEM as oracle), so abTEM remains a credited source but not a runtime dependency.
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

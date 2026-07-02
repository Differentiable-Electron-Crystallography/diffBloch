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

- **Atomic displacement parameter (ADP) frame conventions.**
  Trueblood, K. N. et al. (1996). *Atomic displacement parameter nomenclature: report of a
  subcommittee on atomic displacement parameter nomenclature.* **Acta Crystallographica A52,
  770–781.** DOI: [10.1107/S0108767396005697](https://doi.org/10.1107/S0108767396005697). Defines
  the `U_cif` / `U_cart` / `U*` frames and the orthogonalization matrix `A` (their eq. 50) the
  private `diffBloch` uses. `core/adp` maps raw ADPs into the reciprocal `U*` frame that
  `core/scattering` consumes: Uani via `U*_ij = d*_i d*_j U_cif_ij` (the private CIF→Cartesian→star
  chain with `A` cancelled algebraically — exactly faithful), and Uiso via the textbook
  `U* = Uiso G*` (`G* = B B^T`, `B = reciprocal_cell`).
  **Note (private bug, flag upstream):** the private `Atoms.A_matrix()` builds `A[2,2] = 1/c*` from
  `c_star = |cross(c, b)|/V = |a*|` instead of `|cross(a, b)|/V = |c*|`, mislabelling `|a*|` as
  `c*`. This only enters the Uiso path (Uani cancels `A`) and only matters for anisotropic cells;
  diffBloch uses the convention-correct reciprocal metric instead, so it intentionally does **not**
  reproduce that quantity.

- **Refinement loss / agreement metrics.**
  The Bragg R(obs) factor `R = Σ|√I_obs − √I_calc| / Σ√I_obs` (over reflections with `I_obs > 3σ`)
  is the standard crystallographic agreement index (e.g. Spence & Zuo above; Giacovazzo, *Fundamentals
  of Crystallography*). The **weighted R2** `w_rbragg` and its weighting scheme (instability factor
  `μ`, weak-reflection floor) follow the SI of Klar, P. B. et al. (2023). *Accurate structure models
  and absolute configuration determination using dynamical effects in continuous-rotation 3D electron
  diffraction data.* **Nature Chemistry 15, 848–855.** DOI:
  [10.1038/s41557-023-01186-1](https://doi.org/10.1038/s41557-023-01186-1). Loss bodies are ported
  from the private `diffBloch/metrics.py`. The same paper's SI also defines the **rsg/dsg active-beam
  selection** (`preprocess.select_beams`): keep a reflection when `|Sg|/sg_max < rsg` *and*
  `sg_max - |Sg| > dsg`, with `sg_max = |g_perp|·deg2rad(semiangle)` the excitation-error spread over
  the integration cone. We take `g_perp = (g_x, g_y)` (perpendicular to the `-z` beam); see
  `DIVERGENCE.md` for the private `filter_hkls` transverse-axis divergence we correct.

- **Crystal orientation from the UB matrix (Busing-Levy formalism).**
  Busing, W. R. & Levy, H. A. (1967). *Angle calculations for 3- and 4-circle X-ray and neutron
  diffractometers.* **Acta Crystallographica 22, 457-464.** DOI:
  [10.1107/S0365110X67000970](https://doi.org/10.1107/S0365110X67000970). Defines the reciprocal
  `B` matrix (`a*`-along-x setting) and the `UB = U B` orientation formalism. `preprocess/orientation`
  derives per-rotation crystal orientations natively as `R_z(ω) R_x(α) R_y(β) @ (UB B^-1)`, the
  rotation ordering and `B` convention taken from the private `diffBloch/rotation_dataset.py`. The
  resulting orientation matrices are deliberately **non-orthonormal** (`U = UB B^-1` folds a ~1%
  measured-vs-ideal cell correction); geometry uses `reciprocal_cell(cell @ orientation.T)`.

_(PETS / observation-model references will be added when stage 9+ lands the observation model.)_

- **Convergence-testing method (concepts borrowed, no dependency).** The `converge_*` sweep design
  draws on established numerical-convergence and optimization practice, not a hyperparameter-search
  framework: **block coordinate descent** (cyclic per-parameter minimization to a joint fixpoint --
  Wright, S. J. (2015). *Coordinate descent algorithms.* **Math. Programming 151, 3-34.** DOI:
  [10.1007/s10107-015-0892-3](https://doi.org/10.1007/s10107-015-0892-3)); **early-stopping
  patience** (require several consecutive non-improving steps before stopping -- Prechelt, L.
  (1998). *Early Stopping - But When?*, in *Neural Networks: Tricks of the Trade*, LNCS 1524. DOI:
  [10.1007/3-540-49430-8_3](https://doi.org/10.1007/3-540-49430-8_3)); and the **Grid Convergence
  Index / Richardson extrapolation** asymptotic-range criterion for discretization studies (Roache,
  P. J. (1994). *A Method for Uniform Reporting of Grid Refinement Studies.* **J. Fluids Eng.
  116(3), 405-413.** DOI: [10.1115/1.2910291](https://doi.org/10.1115/1.2910291)). We adopt the
  *ideas* (skip-null + patience + cap, coordinate descent) and do **not** take a runtime dependency
  on Optuna / Ray Tune / Ax / scikit-optimize / Weights & Biases. See
  `design/decisions/stage11-convergence.md`.

- **Cross-validation for crystallographic refinement (concept, no dependency).** The decision that a
  whole-*rotation* train/validation split is a weak cross-validation guard for over-determined
  physics refinement -- and that the principled analog holds out *reflections*, not orientations --
  rests on the **free R-factor** (R_free): Brünger, A. T. (1992). *Free R value: a novel statistical
  quantity for assessing the accuracy of crystal structures.* **Nature 355, 472-475.** DOI:
  [10.1038/355472a0](https://doi.org/10.1038/355472a0). We adopt the *idea* (unbiased held-out
  reflection cross-validation) as the reference point; nothing is taken as a dependency. See
  `design/decisions/train-validation-split.md`.

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
- **toolz / funcy** — Ramda-style functional utility belts for Python (`pipe`, `compose`, `curry`).
  Reference point for our own `pipeline` / `iterate_until` combinators; not imported (we keep our
  combinators typed to `Plan -> Plan`, per `composable-methods.md`).
  <https://github.com/pytoolz/toolz>
- **returns** (dry-python) — typed FP primitives for Python, including a real `State` / `StateT`,
  `Result`, and `Maybe`. The named construct our preprocess driver hand-rolls (a `StateT`-shaped
  coordinate-descent loop; see `design/decisions/plan-composition-shapes.md`). Not imported for now;
  a possible future task (see ROADMAP cross-cutting). <https://github.com/dry-python/returns>
- **Alternative scattering parametrizations examined** (we chose Lobato):
  **Kirkland** (Dirac–Fock Gaussian/Lorentzian fit, *Advanced Computing in Electron Microscopy*; code
  at <https://sourceforge.net/projects/computem/>), **Peng et al.** (Peng, L.-M. et al. (1996),
  *Acta Cryst.* A52, 257, DOI: [10.1107/S0108767395014371](https://doi.org/10.1107/S0108767395014371))
  — both also tabulated by abTEM — and the **ReciPro** crystallographic suite
  (<https://github.com/seto77/ReciPro>).

## Conventions credited
- Reciprocal-cell convention (`pinv(cell).T`) follows the ASE-compatible helper from the research
  codebase; ADP frame conventions follow standard crystallographic (cctbx-compatible) definitions.
- **State monad / stateful-loop decomposition** — the preprocess driver's shape (pure `Plan -> Plan`
  levers + a driver holding the loop state off the value) is the standard `State` decomposition:
  Wadler, *Monads for functional programming* (Advanced Functional Programming, 1995); Haskell
  `Control.Monad.State` (mtl) and the `iterateUntilM` fixpoint loop of `Control.Monad.Loops`
  (`monad-loops` on Hackage); the Elm Architecture `Model`/`update` guide
  (<https://guide.elm-lang.org/architecture/>). See `design/decisions/plan-composition-shapes.md`.
- **Coordinate descent** — the alternate-one-lever-at-a-time-to-a-joint-fixpoint scheme the driver
  runs. Wright, S. J. (2015), *Coordinate descent algorithms*, Mathematical Programming 151, 3,
  DOI: [10.1007/s10107-015-0892-3](https://doi.org/10.1007/s10107-015-0892-3).

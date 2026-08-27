# Publications and references

## Publications using diffBloch

- Colmey, B., Doherty, T. A. S., Malik, S. A. & Midgley, P. A. (2026). *The role of absorption in
  three-dimensional electron diffraction dynamical structure refinement.* Submitted to Acta
  Crystallographica A. <https://arxiv.org/abs/2602.08935>

- Malik, S. A., Doherty, T. A. S., Colmey, B., Roberts, S. J., Gal, Y. & Midgley, P. A. (2026).
  *Hybrid physics-machine learning models for quantitative electron diffraction refinements.*
  Nature Communications. <https://www.nature.com/articles/s41467-026-71673-9>

## General references

diffBloch stands on published science and several open-source projects. This section records the
sources used throughout the codebase.

- **Electron scattering factors (Lobato parametrization, 2026, default).**
  Lobato, I., Zhang, Z., Van Aert, S. & Kirkland, A. I. (2026). *Updated all-electron Dirac–Fock
  densities and an element-adaptive parameterisation of scattering factors and potentials for
  neutral atoms.* Submitted to Acta Crystallographica A.
  <https://arxiv.org/abs/2608.14934>

- **Electron scattering factors (Lobato parametrization, 2014, `blochwave.scattering_factors:
  "lobato2014"`).**
  Lobato, I. & Van Dyck, D. (2014). *An accurate parameterization for scattering factors, electron
  densities and electrostatic potentials for neutral atoms that obey all physical constraints.*
  **Acta Crystallographica A70(6), 636–649.** DOI: [10.1107/S205327331401643X](https://doi.org/10.1107/S205327331401643X)
  — open access ([PDF](https://nano.uantwerpen.be/nanorefs/pdfs/OA_10.1107_S205327331401643X.pdf)).

- **Dynamical electron diffraction (Bloch-wave method).**
  Spence, J. C. H. & Zuo, J. M. (1992). *Electron Microdiffraction.* Plenum Press, New York.

- **Bloch-wave matrix-exponential propagation (GPU-accelerated).**
  Pennington, R. S., Wang, F. & Koch, C. T. (2014). *Stacked-Bloch-wave electron diffraction
  simulations using GPU acceleration.* **Ultramicroscopy 141, 32–37.**
  <https://www.sciencedirect.com/science/article/pii/S0304399114000485>

- **Dynamical refinement of 3D electron diffraction data (rocking-curve integration framework).**
  Palatinus, L., Petříček, V. & Corrêa, C. A. (2015). *Structure refinement using precession electron
  diffraction tomography and dynamical diffraction: theory and implementation.* **Acta
  Crystallographica A71, 235–244.** DOI:
  [10.1107/S2053273315001266](https://doi.org/10.1107/S2053273315001266); with the companion tests
  paper Palatinus, L. et al. (2015). *…: tests on experimental data.* **Acta Crystallographica B71,
  740–751.** DOI: [10.1107/S2052520615017023](https://doi.org/10.1107/S2052520615017023).

- **Orientation refinement (hexagonal modified-simplex search).**
  Palatinus, L., Jacob, D., Cuvillier, P., Klementová, M., Sinkler, W. & Marks, L. D. (2013).
  *Structure refinement from precession electron diffraction data.* **Acta Crystallographica A69,
  171–188.** DOI: [10.1107/S010876731204946X](https://doi.org/10.1107/S010876731204946X).

- **Incoherent isotropic mosaicity model.**
  Palatinus, L. (2024). *Including mosaicity effects in the dynamical refinement against 3D ED
  data.* **Acta Crystallographica A80, e225.**

### Software we depend on, extract from, or verify against

**Runtime dependencies**
- **gemmi** — CIF/mmCIF and PETS parsing, unit-cell handling, and space-group symmetry operations
  (the blessed parser behind `io/`). Wojdyr, M. (2022), *GEMMI: A library for structural biology*,
  Journal of Open Source Software 7(73), 4200. DOI: [10.21105/joss.04200](https://doi.org/10.21105/joss.04200)
  · <https://github.com/project-gemmi/gemmi>
- **PyTorch** — differentiable tensor backend (`core/` constraints, ADP, symmetry expansion onward).
  <https://github.com/pytorch/pytorch>
- **NumPy** — array backend for the pure planning/geometry helpers. <https://github.com/numpy/numpy>
- **pydantic** — boundary validation for config and IO records. <https://github.com/pydantic/pydantic>
- **PyYAML** — experiment/config and lock parsing. <https://github.com/yaml/pyyaml>

**Planned dependencies (seams already in place)**
- **diffpy.structure** — special-position and ADP symmetry-constraint expansion, behind the
  `io.symmetry_setup` seam (lands with the constraints/symmetry stage). <https://github.com/diffpy/diffpy.structure>

**Development & verification oracles (not runtime dependencies)**
- **abTEM** — Madsen, J. & Susi, T. (2021), *The abTEM code: transmission electron microscopy from
  first principles*, Open Research Europe 1:24. <https://open-research-europe.ec.europa.eu/articles/1-24>

## Citing diffBloch

```bibtex
@misc{diffBloch,
  author  = {Doherty, Tiarnan and Malik, Shreshth and Colmey, Benjamin and Maitland, Iain, and Midgley, Paul},
  title   = {diffBloch},
  version = {0.2.0},
  year = {2026},
  url     = {https://github.com/Differentiable-Electron-Crystallography/diffBloch}
}
```

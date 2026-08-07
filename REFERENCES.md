# References 

diffBloch stands on published science and several open-source projects. This file attempts to record all relevant sources used throughout the codebase. 



- **Electron scattering factors (Lobato parametrization).**
  Lobato, I. & Van Dyck, D. (2014). *An accurate parameterization for scattering factors, electron
  densities and electrostatic potentials for neutral atoms that obey all physical constraints.*
  **Acta Crystallographica A70(6), 636–649.** DOI: [10.1107/S205327331401643X](https://doi.org/10.1107/S205327331401643X)
  — open access ([PDF](https://nano.uantwerpen.be/nanorefs/pdfs/OA_10.1107_S205327331401643X.pdf)).
  

- **Dynamical electron diffraction (Bloch-wave method).**
  Spence, J. C. H. & Zuo, J. M. (1992). *Electron Microdiffraction.* Plenum Press, New York.
  

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

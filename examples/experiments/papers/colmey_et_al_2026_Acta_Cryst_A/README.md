# Colmey et al. (2026), pending review, submitted to Acta Crystallographica A, Feb 6th, 2026

---

### The role of absorption in three-dimensional electron diffraction dynamical structure refinement

**Benjamin Colmey**¹, **Tiarnan A. S. Doherty**¹ ² , **Shreshth A. Malik**² , **Paul A. Midgley**¹

¹ Department of Materials Science and Metallurgy, University of Cambridge, 27 Charles Babbage Rd,
Cambridge, CB3 0FS, United Kingdom
² OATML, Department of Computer Science, University of Oxford, Wolfson Building, Parks Rd, Oxford,
OX1 3QG, United Kingdom

**arXiv:** https://arxiv.org/abs/2602.08935

> The role of absorption in 3D electron diffraction is established through analytical theory,
> simulation, and dynamical refinement. A two-beam expression for the absorbed integrated intensity
> in centrosymmetric crystals is derived, showing that for t/ξg ≪ 1 reflections follow a uniform
> exponential decay set by the mean absorptive potential U'0. Many-beam simulations of both
> centrosymmetric and non-centrosymmetric crystals reveal additional reflection-specific anomalous
> absorption beyond the uniform attenuation set by U'0. Neglecting these effects in dynamical
> refinement of integrated intensities incurs an error that increases approximately linearly with
> thickness, with this error becoming more severe near zone axes. Dynamical refinements were
> performed on CsPbBr3, quartz, and borane, with the inclusion of absorption yielding an improvement
> in R_obs from 6.4 to 5.3% for CsPbBr3 and negligible improvements for quartz and borane. Anomalous
> absorption may therefore be ignored for routine refinement of integrated intensities except in
> high-Z materials at thicknesses approaching ξg.

---

This directory holds six runnable diffBloch experiment configs reproducing the three materials refinement in
the paper, elastic and absorptive:

```text
data/quartz-absorption/    data/quartz-no-abs/
data/cspbbr3-absorption/   data/cspbbr3-no-abs/
data/borane-absorption/    data/borane-no-abs/
```

## Data source

CsPbBr3, alpha-quartz, and borane data:

> Suresh, A., Yörük, E., Cabaj, M. K., Brázda, P., Výborný, K., Sedláček, O., Müller, C.,
> Chintakindi, H., Eigner, V. & Palatinus, L. (2024). *Ionisation of atoms determined by kappa
> refinement against 3D electron diffraction data.* Nature Communications 15, 9066.
> https://doi.org/10.1038/s41467-024-53448-2

```bibtex
@article{Suresh2024,
  author  = {Ashwin Suresh and Emre Yörük and Małgorzata K. Cabaj and Petr Brázda and
             Karel Výborný and Ondřej Sedláček and Christian Müller and
             Hrushikesh Chintakindi and Václav Eigner and Lukáš Palatinus},
  title   = {Ionisation of atoms determined by kappa refinement against 3D electron diffraction data},
  journal = {Nature Communications},
  volume  = {15},
  pages   = {9066},
  year    = {2024},
  doi     = {10.1038/s41467-024-53448-2},
  url     = {https://doi.org/10.1038/s41467-024-53448-2}
}
```

## Results

#### Published (Colmey et al. 2026)

|                          | CsPbBr3  | alpha-quartz | Borane   |
| ------------------------ | :------: | :----------: | :------: |
| **R_obs (%), elastic**   |   6.40   |     4.12     |   9.54   |
| **wR2 (%), elastic**     |   6.73   |     3.84     |   8.56   |
| **R_obs (%), absorptive**|   5.26   |     4.00     |   9.48   |
| **wR2 (%), absorptive**  |   5.31   |     3.66     |   8.51   |

#### This repository (diffBloch 0.2.0)

|                          | CsPbBr3  | alpha-quartz | Borane   |
| ------------------------ | :------: | :----------: | :------: |
| **R_obs (%), elastic**   | pending  |    pending   | pending  |
| **wR2 (%), elastic**     | pending  |    pending   | pending  |
| **R_obs (%), absorptive**| pending  |     4.92     | pending  |
| **wR2 (%), absorptive**  | pending  |     3.72     | pending  |

Only `quartz-absorption` has a completed run recorded here so far (epoch 40/40). Fill in the rest
from `refinement_report.txt`/`refinement_summary.json` as the other five runs complete.

## Reproducing these numbers

diffBloch has changed substantially since the runs that produced the numbers above -- checkpoint
format, orientation search, absorption handling, and more have all moved since then. Running these
configs on the current codebase is not guaranteed to reproduce the published numbers exactly.

To reproduce the paper's results bit-for-bit, use the original prototype linked in the paper itself,
not this repository. That prototype predates GPU support and several of the changes in this
codebase, so it is considerably slower -- a single one of these refinements could take on the order
of days, start to finish.

## Running these examples

```bash
uv run diffbloch run preprocess examples/experiments/papers/colmey_et_al_2026_Acta_Cryst_A/data/quartz-absorption
uv run diffbloch run refine examples/experiments/papers/colmey_et_al_2026_Acta_Cryst_A/data/quartz-absorption
```

Swap in any of the six directory names above for the other materials/absorption settings.

# Bloch wave simulation

diffBloch solves the electron wave equation inside the crystal exactly (for a finite beam set)
using the Bloch wave formalism, rather than the multislice method. Multislice scales better to very
large simulations ({math}`N\log_2 N` in the number of Fourier components vs Bloch wave's
{math}`N^3`), but the Bloch wave method gives closed-form intensities that are analytically
differentiable with respect to structural parameters and handle arbitrary crystal orientation
naturally — both essential for gradient-based refinement against a continuous-rotation tilt series.

This page derives the structure matrix diffBloch actually assembles and solves; for how orientation
and thickness are fitted around it, see [Preprocessing](preprocessing.md), and for optimal choice of simulation parameters see [Convergence testing](convergence-testing.md) and [Hyperparameter selection](hyperparameter-selection.md).

This section of the codebase draws heavily on the abTEM code (Madsen, J. & Susi, T. (2021), *The
abTEM code: transmission electron microscopy from first principles*, Open Research Europe 1:24,
<https://open-research-europe.ec.europa.eu/articles/1-24>).

## Diffraction geometry and the excitation error

A reciprocal lattice vector {math}`\mathbf{g}_{hkl}` satisfies the Bragg condition when it lies on
the Ewald sphere: {math}`\mathbf{k}_g - \mathbf{k}_0 = \mathbf{g}`, for incident and diffracted
wavevectors {math}`\mathbf{k}_0`, {math}`\mathbf{k}_g` of magnitude {math}`1/\lambda`. A finite
crystal thickness elongates each reciprocal lattice point into a `relrod`, so a beam can be excited
even when the **excitation error** ({math}`S_{\mathbf{g}}`), the distance of the {math}`\mathbf{g}` to the Ewald sphere surface is non-zero. {math}`S_{\mathbf{g}}` is given by:

```{math}
S_{\mathbf{g}} = \frac{|\mathbf{K}|^2 - |\mathbf{K}+\mathbf{g}|^2}{2|\mathbf{K}|}.
```

diffBloch calculates this value (`core.dynamical.excitation_errors`),
with the beam wavevector {math}`\mathbf{K}` corrected for the mean-inner-potential offset {math}`U_0`:
{math}`K_n = \sqrt{1/\lambda^2 + U_0}`. `blochwave.sg_max` (see
[Hyperparameter selection](hyperparameter-selection.md)) is the cutoff on {math}`|S_{\mathbf{g}}|`
admitting a beam into the calculation at a given tilt.

## Elastic scattering and the structure factor

Electrons interact with the crystal's total electrostatic potential {math}`V(\mathbf{r}). Its Fourier coefficients {math} `V_{\mathbf{g}}` sum
over the atoms in the unit cell, with each atom each contributing its electron scattering factor
{math}`f^e(s)` damped by thermal motion
(the Debye–Waller factor) and phased by its fractional position:

```{math}
F_{\mathbf{g}} = \frac{1}{\Omega}\sum_j f^e_j(s)\,T_j(\mathbf{h})\,O_j\,
\exp(2\pi i\,\mathbf{h}\cdot\mathbf{r}_j),
```

diffBloch first computes the Born-approximation structure factor given above (`core.scattering.structure_factors`), using the Lobato–Van Dyck (2014) parametrization for {math}`f^e(s)`. 

These values may then be converted to {math}`U_{\mathbf{g}}` using:

```{math}
U_{\mathbf{g}} = \gamma\,\frac{F_{\mathbf{g}}}{\pi\Omega} = \frac{2m|e|V_{\mathbf{g}}}{h^2},
```

with {math}`\gamma` the relativistic mass factor, {math}`m` the (relativistic) electron mass,
{math}`e` the elementary charge, and {math}`h` Planck's constant. 


## The Bloch wave formalism

Expanding the electron wavefunction inside the crystal as a sum of Bloch states,

```{math}
\psi(\mathbf{r}) = \sum_i c_i \exp(2\pi i\mathbf{k}^{(i)}\cdot\mathbf{r})
\sum_{\mathbf{g}} C_{\mathbf{g}}^{(i)}\exp(2\pi i\mathbf{g}\cdot\mathbf{r}),
```

and substituting into the time-independent Schrödinger equation yields, a coupled
linear system for the Fourier coefficients — an eigenvalue problem
{math}`AC^{(i)} = 2K_n\gamma^{(i)}C^{(i)}` in the **structure matrix** {math}`A`:

```{math}
A_{ii} = 2K_nS_{g_i}M_{ii}, \qquad
A_{ij} = \sigma M_{ii}M_{jj}F(\mathbf{g}_j-\mathbf{g}_i) \quad (i\ne j).
```

The diagonal holds each beam's own excitation error; the off-diagonal couples every pair of beams
through the structure factor of their difference vector — an electron diffracted from
{math}`(000)` into {math}`(200)` can be rescattered into {math}`(220)`, and so on. Retaining every
beam within `blochwave.g_max`/`sg_max` at a given tilt is the **many-beam** solution; the classical
**two-beam approximation** keeps only {math}`(000)` and one diffracted beam, reducing {math}`A` to
a {math}`2\times2` matrix — useful for intuition (see the rocking-curve width formula in
[Convergence testing](convergence-testing.md#simulation-convergence)) but too coarse for
quantitative refinement once several reflections are simultaneously strongly excited.

### Example structure matrix

Instantiating {math}`A_{ii}`/{math}`A_{ij}` above for a concrete systematic row of five beams
{math}`(\overline{2}00), (\overline{1}00), (000), (100), (200)` (all in the zero-order Laue zone, so
{math}`M_{ii}=1` throughout):

```{math}
A =
\begin{pmatrix}
2K_nS_{\overline{2}00} & \sigma F(100) & \sigma F(200) & \sigma F(300) & \sigma F(400) \\
\sigma F(100)^{*} & 2K_nS_{\overline{1}00} & \sigma F(100) & \sigma F(200) & \sigma F(300) \\
\sigma F(200)^{*} & \sigma F(100)^{*} & 2K_nS_{000} & \sigma F(100) & \sigma F(200) \\
\sigma F(300)^{*} & \sigma F(200)^{*} & \sigma F(100)^{*} & 2K_nS_{100} & \sigma F(100) \\
\sigma F(400)^{*} & \sigma F(300)^{*} & \sigma F(200)^{*} & \sigma F(100)^{*} & 2K_nS_{200}
\end{pmatrix}
```

The diagonal holds each beam's excitation error, growing outward along the row. Every off-diagonal
entry depends only on the difference vector {math}`\mathbf{g}_j-\mathbf{g}_i`, so the matrix is
Toeplitz along each band — the {math}`(000,100)` and {math}`(100,200)` couplings are both
{math}`\sigma F(100)`. The lower triangle is the conjugate of the upper triangle, since {math}`A`
must be Hermitian for `bloch_eigen`'s eigendecomposition to apply.

## Solving: two equivalent routes, one gradient-safe

With boundary conditions {math}`\psi(0)` fixed at the entrance surface, the wavefield at thickness
{math}`t` is

```{math}
\psi(t) = \exp\!\left(\frac{i\pi t}{K_n}A\right)\psi(0).
```

There are two mathematically equivalent ways to evaluate this, and diffBloch implements both as
`blochwave.solver` choices (see [Hyperparameter selection](hyperparameter-selection.md)):

- **`bloch_eigen`** diagonalises {math}`A = C\,\mathrm{diag}(\gamma)\,C^{-1}` once and reads off
  {math}`\psi_{\mathbf{g}}(t) = \sum_i C_0^{(i)-1}C_{\mathbf{g}}^{(i)}\exp(2\pi i\gamma^{(i)}t)` —
  the classical closed-form Bloch wave solution, cheap once diagonalised.
- **`matrix_exp`** evaluates the matrix exponential directly, without an intermediate
  eigendecomposition. It is the default: eigendecomposition of a **non-Hermitian** matrix (the case
  whenever `absorption: true` adds an imaginary component to {math}`A`) is numerically unstable to
  differentiate through, so `bloch_eigen` is rejected outright when absorption is enabled — see
  `BlochwaveConfig`'s validation.

{math}`\psi(t)` has one entry per beam in {math}`A`'s own index set — the **solve beams** (see
[Beam sets and scoring](preprocessing.md#beam-sets-and-scoring-inside-a-plan)) — so an intensity is
only ever calculated for a reflection that beam selection actually admitted into {math}`A`, never
for an arbitrary {math}`\mathbf{g}` outside it. For each solve beam, the calculated intensity is
{math}`I_{\mathbf{g}}(t) = |\psi_{\mathbf{g}}(t)|^2`. A continuous-rotation frame sums this over sampled sub-orientations
across the rocking curve (and, when `blochwave.mosaicity` is enabled, smoothed over a moving-average
window derived from the apparent mosaicity recorded in `.cif_pets`) rather than evaluating a single static orientation
— see [Preprocessing](preprocessing.md) for how those tilts and beam couplings are assembled around
this solve.

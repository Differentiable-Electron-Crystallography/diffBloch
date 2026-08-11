# Workflow

diffBloch converts a starting crystal structure and 3D electron-diffraction data into a
refined structure through the following calculation:

```text
.cif structure + .cif_pets experiment
  -> crystal potential
  -> Bloch-wave simulation
  -> calculated/observed intensity comparison
  -> preprocessing
  -> structural refinement
```

This page walks through that calculation end to end. For how the codebase itself is organised into
packages and modules, see [Architecture](architecture.md).

## Structural and experimental data

The structure `.cif` supplies asymmetric-unit atoms, fractional coordinates, elemental identities,
occupancies, atomic displacement parameters (ADPs), and space-group symmetry operations. It also
carries its own unit cell, but that cell is used only as a consistency check, never for simulation
geometry — see [Unit-cell authority](#unit-cell-authority) below.

The `.cif_pets` file supplies the measured experimental data: observed reflection intensities and
uncertainties for each rotation, the UB orientation matrix, electron wavelength, the goniometer
angles, and its own unit cell.

Both files are parsed into validated numerical records before simulation begins. Malformed values
and unsupported representations fail at this boundary. See [Inputs](inputs.md) for the file
requirements and experiment-directory layout.

### Unit-cell authority

**PETS's cell, not the structure CIF's, is authoritative for every piece of simulation
geometry**: the structure-factor grid, the reciprocal basis, the cell volume, the ADP `U*`-frame
conversion, and the beam geometry derived from that grid. The CIF's own cell is checked against
PETS's on load — a >1% relative difference on any of `a, b, c, alpha, beta, gamma` logs a warning
(stating that PETS overrides the CIF), a >5% difference raises and stops, listing every offending
parameter, both values, and the percentage difference. Fractional atomic coordinates are read from the CIF
unchanged; they are simply interpreted against PETS's cell rather than the CIF's own. For a combined
(`inputs.multi_dataset`) experiment, the first combined `.cif_pets` file is the shared anchor every
other input — the CIF's and every further combined file's — is checked against, under the same two
thresholds; see [Inputs](inputs.md#unit-cell-authority-pets-overrides-the-structure-cif) for the
full rule and [Preprocessing](preprocessing.md#unit-cell-authority-pets-overrides-the-structure-cif)
for how each combined file's own orientation matrix still comes from its own UB and cell before
being composed with that shared anchor.

## Constructing the crystal potential

The asymmetric unit is expanded using the symmetry operations from the `.cif`: symmetry operation
{math}`(R_m, \mathbf{t}_m)` maps atomic position {math}`\mathbf{r}_i` to
{math}`\mathbf{r}'_{mi} = R_m\mathbf{r}_i + \mathbf{t}_m`. Each expanded atom then contributes to
the elastic electron structure factor {math}`F_{\mathbf{g}}` (optionally with an imaginary
absorptive component) -- see
[Elastic scattering and the structure factor](bloch-wave-simulation.md#elastic-scattering-and-the-structure-factor)
for the full expression. {math}`F_{\mathbf{g}}` is the reciprocal-space crystal potential the
Bloch-wave calculation consumes.

## Bloch-wave simulation

For each experimental orientation, diffBloch selects the reciprocal-lattice vectors included in the
calculation, calculates their excitation errors, and constructs and solves the Bloch structure
matrix — the full derivation, from the elastic structure factor through the relativistic
interaction parameter to the structure matrix and its two equivalent solvers, is in
[Bloch-wave simulation](bloch-wave-simulation.md). The incident wave begins entirely in the
transmitted beam; propagating it through specimen thickness {math}`t` and reading off
{math}`|\psi_{\mathbf{g}}(t)|^2` gives the calculated intensity for every included reflection.

A continuous-rotation frame covers an angular interval rather than one static orientation.
diffBloch samples that rocking curve, performs a Bloch-wave solve at each sub-orientation, and sums
the intensities incoherently:

```{math}
I_{\mathrm{frame}}(t) = \sum_k |\psi^{(k)}(t)|^2.
```

With `mosaicity: true`, diffBloch reads the apparent mosaicity from `.cif_pets` and converts it to a
moving-average width using the angular spacing between sampled orientations. The calculated
rocking curve is smoothed over that width before it is summed. This does not add Bloch-wave solves.
`mosaicity: false` (the default) applies no smoothing. The legacy `{window: N}` form sets the width
directly. The output is one calculated diffraction pattern for each experimental rotation.

## Comparing simulation with experiment

The calculated intensities are matched to the observed reflections from the corresponding `.cif_pets`
rotation. diffBloch distinguishes three reflection sets:

- **solve beams** participate in the dynamical calculation;
- **matched reflections** occur in both the calculated and experimental patterns;
- **scored reflections** are the matched reflections admitted to the objective.

Refinement minimises a scaling-optimised residual. The default is {math}`wR_2`.
{math}`R_{\mathrm{obs}}`, restricted to observations satisfying {math}`I>3\sigma`, is also
reported.

## Preprocessing the experiment

The PETS orientations provide the starting geometry. diffBloch searches a small angular region
around each orientation and retains the orientation with the best agreement to the observed
intensities.

Specimen thickness strongly affects dynamical intensities. Preprocessing determines thickness before
orientation optimization. This improves the orientation comparison and gives the thickness model a
better starting value during refinement.

Convergence tests determine the reciprocal-space cutoffs and rocking-curve sampling used during
refinement.

The output is a `Plan` containing the orientations, thicknesses, beam geometry, experimental
observations, and calculated/observed reflection alignment. The `Plan` remains fixed during
structural refinement. See [Preprocessing](preprocessing.md) and
[Convergence testing](convergence-testing.md).

## Refinement

Refinement holds the settled `Plan` fixed and repeatedly evaluates the scientific calculation:

```text
raw structural parameters
  -> physical crystallographic structure
  -> crystal potential
  -> Bloch-wave intensities
  -> comparison with experiment
  -> scalar objective
  -> gradients and updated parameters
```

Automatic differentiation calculates the gradient of the objective with respect to the selected
atomic positions, ADPs, and occupancies. Crystallographic constraints are applied before the
potential is calculated. Molecular constraints, such as hydrogen riding, and soft structural
restraints can also be included. See [Refinement](refinement.md).

## Learned model components

Differentiable models can be refined alongside the atomic structure. The current thickness neural
network takes the experimental rotation coordinate and supplies a thickness to each Bloch-wave
calculation. Its parameters are updated using the same diffraction objective as the structure.

The same design can support other quantities that vary through an experiment. Planned beam
damage models will describe changes across an ordered dataset. Future components could describe
other systematic experimental changes or additional scattering contributions, including inelastic
scattering.

## Logs, locks, and reproducibility

Logs report preprocessing, comparison metrics, and refinement progress. Locks record the input
files, configuration, code version, and generated results. See
[Reproducibility](reproducibility.md).

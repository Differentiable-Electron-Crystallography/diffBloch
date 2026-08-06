# Architecture

## Building the potential

Everything below starts after the `io` boundary: `positions`, `numbers`, `occupancies`, `uij_star`
are already validated tensors on the ASU.

**Symmetry expansion.** The refined parameters live only on the asymmetric unit. Before any
scattering physics, every symmetry operation {math}`(R_m, \mathbf{t}_m)` of the space group is
applied to each ASU atom {math}`i`:

```{math}
\mathbf{r}'_m = R_m \mathbf{r}_i + \mathbf{t}_m ,
```

producing the full-cell atom list (positions, atomic numbers, occupancies, ADPs all carried through
the same expansion). This is `core.symmetry.expand_asu` — purely geometric, no scattering physics
yet, differentiable in `positions` because {math}`R_m`/{math}`\mathbf{t}_m` are fixed constants and
only {math}`\mathbf{r}_i` carries gradient.

**Elastic structure factor.** For each reflection {math}`\mathbf{h}`, sum a scattering contribution
over every expanded atom:

```{math}
F_{\mathbf{g}}(\mathbf{h}) = \frac{1}{V}\sum_{j} f_j(s)\, T_j(\mathbf{h})\, O_j\,
\exp(2\pi i\, \mathbf{h}\cdot\mathbf{r}_j) ,
```

with {math}`s = |\mathbf{g}|/2` and {math}`V` the cell volume. {math}`f_j(s)` is the electron
scattering factor: diffBloch hardcodes the Lobato & Van Dyck (2014) parametrization — a
five-Gaussian-rational fit, {math}`f_e(s) = \sum_i a_i(2+b_i s^2)/(1+b_i s^2)^2`, coefficients
vendored per element from the published table (`core/data/lobato.json`). There is no runtime
switch: Kirkland and Peng *et al.* parametrizations were evaluated during design (see
`REFERENCES.md`) but neither is wired in as a swappable option.

{math}`T_j(\mathbf{h})` is the Debye–Waller factor. Every atom's ADP is carried in the reciprocal
`U*` frame regardless of whether the CIF gave `Uiso` or `Uani` (Trueblood *et al.*, 1996
convention: `Uani` maps directly, {math}`U^*_{ij} = d^*_i d^*_j\, U^{\mathrm{cif}}_{ij}`; `Uiso`
maps via {math}`U^* = U_{\mathrm{iso}} G^*`, {math}`G^* = BB^\top`). One formula consumes both:

```{math}
T_j(\mathbf{h}) = \exp(-2\pi^2\, \mathbf{h}^\top U^*_j\, \mathbf{h}) .
```

**Absorption.** With `absorption.enabled`, {math}`f_j(s)` picks up an imaginary part:
{math}`f_j \to f_j + i f'_j(s, B_j)`, where {math}`f'_j` is a fitted absorptive form factor
(Gaussian-sum interpolation over the atom's isotropic-equivalent {math}`B_j`, converted from its
`U*`), following Bird & King (1990). {math}`F_{\mathbf{g}}` is then complex for every reflection —
which is what later makes the structure matrix {math}`A` non-Hermitian.

{math}`F_{\mathbf{g}}` is a setup constant with respect to orientation — computed once per
parameter state (`core.scattering.structure_factors`) and reused across every rotation, since it
depends only on the structure, not on the beam geometry.

## The Bloch-wave forward model

The forward model maps a beam set, an orientation, an energy, and the structure factors built above
to an exit-wave intensity vector.

**Structure matrix.** For a beam set {math}`\{\mathbf{g}_i\}_{i=1}^{N}` at a given orientation,
`core.dynamical.assembly` assembles the Hermitian (or, with absorption, non-Hermitian) {math}`N
\times N` structure matrix {math}`A`:

```{math}
A_{ii} = 2 K_n \, S_{g_i} \, M_{ii}, \qquad
A_{ij} = \sigma \, M_{ii} M_{jj} \, F(\mathbf{g}_j - \mathbf{g}_i) \;\; (i \neq j),
```

with Lorentz/obliquity factor {math}`M_{ii} = (1 - g_{i,z}/K_n)^{-1/2}` (equal to 1 at
{math}`\mathbf{g}=0`), interaction constant {math}`\sigma`, and in-crystal wavevector magnitude
{math}`K_n = \sqrt{1/\lambda^2 + U_0}`, where {math}`U_0` is the mean inner potential correction,
{math}`U_0 = |F_{000}| \cdot \sigma / (\kappa \lambda \pi)`, computed once from the starting
structure and held fixed through refinement. The off-diagonal is a differentiable gather of
{math}`F_{\mathbf{g}}` onto every beam-pair difference; only {math}`F` carries gradient — the
gather indices are precomputed geometry.

The excitation error {math}`S_{g_i}` is the signed distance of {math}`\mathbf{g}_i` from the Ewald
sphere: with beam {math}`\mathbf{K}` fixed along {math}`-z` at magnitude {math}`K_n`,

```{math}
S_{g} = \frac{|\mathbf{K}|^2 - |\mathbf{K}+\mathbf{g}|^2}{2|\mathbf{K}|}
```

(Spence & Zuo, 1992 convention); {math}`S_{g}=0` exactly at {math}`\mathbf{g}=0`, so the
transmitted beam always sits on the diagonal at zero.

{math}`N` is not fixed by the structure matrix code — it is whatever beam set preprocessing handed
it, and that is entirely config-controlled: `blochwave.g_max` bounds the candidate pool radius
(and sets the {math}`F_{\mathbf{g}}` support at {math}`2\times g_{\max}`); `select_beams` prunes
candidates to the active set with the Klar cutoffs `rsg`/`dsg` against
{math}`sg_{\max}=|g_\perp|\cdot\text{semiangle}`. On the default coupled path, {math}`N` is the
*union* of active beams across a whole tilt segment, not one orientation's active set alone —
governed by `coupling_mode`, `fixed_n_segments`, `rocking_curve_sampling` (sub-tilts per segment),
and `union_adaptive`/`union_max_new_beams_pct` (adaptive segment splitting). Since matrix assembly
is {math}`O(N^2)` and the dense solve worse, these are the knobs that set simulation cost.

**Propagation.** `psi0` is the boundary condition: amplitude 1 on the {math}`\mathbf{g}=000` beam,
0 on every other beam in the set — all incident amplitude in the transmitted beam at {math}`t=0`.
`core.solver.propagate` integrates the exit wavefunction through specimen thickness {math}`t`:

```{math}
\psi(t) = \exp\!\left(\frac{i \pi t}{K_n} A\right) \psi_0 .
```

`thicknesses` is a scalar or `(T,)` array of trial values (Å); `propagate` returns `psi` of shape
`(T, N)` — one full {math}`N`-beam amplitude vector per requested thickness, in a single batched
call. Two propagators evaluate this, both over the same `BlochSystem` and swappable without
touching geometry code:

- `matrix_exp` — a dense matrix exponential per thickness (batched over {math}`T`), stable
  autograd. The refinement default.
- `bloch_eigen` — diagonalize {math}`A` once (`eigh`), then each of the {math}`T` thicknesses is a
  cheap phase multiply on the eigenbasis-projected `psi0`: exact and fast for many thicknesses at
  once, but `eigh`'s backward is ill-conditioned near degenerate eigenvalues — routine in symmetric
  crystals — so it is eval-only, never on the differentiated path. The two coincide only where
  {math}`M_{ii}=1` for every beam; otherwise one returns symmetrised amplitudes, the other physical
  ones — a property of the two formulations, not a bug in either.

`intensities = |psi|**2` gives shape `(T, N)`: the full simulated pattern for every candidate
thickness in one shot. That single batched call is why the thickness grid search
(`optimize_thickness`) is cheap — assembling and exponentiating/diagonalizing {math}`A` is paid
once, not once per candidate {math}`t`.

**Alignment.** `core.products.align` places simulated and observed intensities on a common
reflection index for loss evaluation. Before that comparison is meaningful, each virtual frame's
single-orientation exit wave above has to become the actual integrated pattern PETS measured — see
[Rocking-curve integration and mosaicity](#rocking-curve-integration-and-mosaicity).

**SOLVE and SCORED reflection sets are independent.** The SOLVE set is which beams couple
dynamically inside the structure matrix; the SCORED set is which reflections enter the residual.
Every scored reflection must be solved (SCORED {math}`\subseteq` SOLVE), but widening the solve
basis for numerical convergence must not silently change what is scored, and a scoring-set change
must not perturb the dynamical coupling. `preprocess.select_beams` selects the SOLVE set from
excitation error and resolution thresholds (`sg_max`, `g_max`); alignment against the PETS
experimental data determines the SCORED set independently.

## Rocking-curve integration and mosaicity

Everything in [The Bloch-wave forward model](#the-bloch-wave-forward-model) above solves one static
orientation. A continuous-rotation 3DED frame is not static — it integrates over the crystal
sweeping through the Ewald sphere — so a single Bloch solve at the frame's nominal orientation is
not what was measured. `integrate_rocking_curve` closes that gap.

**Building the tilt set.** `preprocess.orientation.rocking_curve_tilts(semiangle, sampling)`
generates `sampling` rotation matrices about the goniometer x-axis, at angles
{math}`\mathrm{linspace}(-\text{semiangle}, +\text{semiangle}, \text{sampling})`. These are pure,
orientation-independent geometry, built once and reused for every rotation. For a virtual frame
with nominal (already PETS-derived) orientation {math}`M_i`, the sub-orientations are

```{math}
M_i^{(k)} = R_{\mathrm{tilt}}(k)\, M_i, \qquad k = 1,\dots,\text{sampling} .
```

`sampling = 1` is special-cased to the identity tilt at angle {math}`0` exactly (not the
`linspace`-start value), so composing this step with `sampling = 1` leaves the `Plan` unchanged —
the integration is opt-in, not baked into `from_experiment`. This step is pure geometry: it rebuilds
each orientation with `N = sampling` sub-orientations sharing the *same* beam set and gather, no
engine or structure factors involved. It runs last in the default recipe, after
`optimize_orientation`/`optimize_thickness`, because those fit against the fast single-solve at the
nominal orientation; only once both are settled is the plan expanded to the full tilt set they will
actually be evaluated at.

**Summing the Bloch waves.** Each of the `N` sub-orientations is solved independently — same beam
set, same {math}`F_{\mathbf{g}}`, but a different structure matrix {math}`A` per tilt, since
{math}`S_g` and {math}`M_{ii}` depend on orientation. That gives `N` separate exit waves
{math}`\psi^{(k)}(t)`, reduced over the tilt axis as an **incoherent** sum of intensities
(`core.products.BlochSolution.integrate` / `reduce_tilts`):

```{math}
I_{\mathrm{frame}}(t) = \sum_{k=1}^{N} \left|\psi^{(k)}(t)\right|^2 ,
```

not {math}`\bigl|\sum_k \psi^{(k)}(t)\bigr|^2`. This is the physical rotation-frame integration —
summing counts across the sweep, not interfering amplitudes — so phase is deliberately discarded;
the integrated solution stores only {math}`\sqrt{I_{\mathrm{frame}}}` as a nominal amplitude with no
downstream use, and only `intensities` feeds alignment and the loss.

**Mosaicity.** At the low-level step-composition API, omitting mosaicity (`mosaicity=None`)
keeps `PlainSum`, the bare sum above. The default app/CLI recipe is different: it passes the
configured `blochwave.mosaicity` into `build_orientation_plans`, and that config currently defaults
to `Mosaicity(window=5)`. The window is a sampled-tilt moving-average width recorded in the plan
recipe/provenance; it is not derived from the PETS free-text mosaicity value. The `mosaicity` step
or configured build reduction swaps the reduction to `MosaicSmoothed(window)`: a `window`-wide moving
average over consecutive tilts on the tilt axis, applied *before* the sum —

```{math}
I_{\mathrm{frame}}(t) = \sum_{k} \left(\frac{1}{\text{window}}\sum_{l=0}^{\text{window}-1}
\left|\psi^{(k+l)}(t)\right|^2\right) ,
```

modelling crystal mosaic spread as a blur of the sampled rocking curve before it is integrated,
rather than integrating the sharp unbroadened curve. `window = 1` is the identity. It can only
compose after `integrate_rocking_curve` (raises below two tilts — a moving average over one point is
meaningless) and must precede the fits: whatever reduction is used at evaluation has to be the same
one the orientation/thickness fits saw, or the fit and the eval would be scoring against two
different forward models — the fit/eval consistency invariant.

## Scoring

**The reflection-comparison set.** Not a union of anything — an intersection.
`build_alignment_plan(solution_hkl, pattern_hkl, restrict_to=scored_hkl)` computes
{math}`\text{pattern} \cap \text{solution} \cap \text{restrict\_to}`: `pattern` is what PETS
observed for that rotation, `solution` is what beams the Bloch solve actually computed (the SOLVE
set), `restrict_to` is the SCORED set. A reflection not solved cannot be scored even if PETS
measured it (SCORED {math}`\subseteq` SOLVE); a reflection PETS did not record cannot be scored even
if it was solved.

**What decides the SCORED set — the fully-integrated filter.** `klar_beam_mask`, applied via
`scoring_selection` on the candidate pool before that intersection. Per reflection {math}`\mathbf{g}`:

```{math}
sg_{\max} = |\mathbf{g}_{\mathrm{lever}}| \cdot \mathrm{deg2rad}(\text{semiangle})
```

is how far {math}`S_g` *sweeps* for that reflection as the crystal rocks through the measured
semiangle, with {math}`\mathbf{g}_{\mathrm{lever}}` the component of {math}`\mathbf{g}`
perpendicular to the rock axis (`(g_y, g_z)` for continuous rotation about the goniometer x-axis) —
reflections far from the rock axis sweep a bigger {math}`S_g` range for the same tilt than
reflections near it. A reflection is kept — judged fully integrated — when both

```{math}
\frac{|S_g|}{sg_{\max}} < rsg, \qquad sg_{\max} - |S_g| > dsg .
```

The first is relative: {math}`S_g` small against the sweep range means the rocking curve's peak
({math}`S_g \approx 0`, exact Bragg) falls inside the measured window, not at or beyond its edge.
The second is an absolute safety margin on top, so a reflection sitting right at the edge — which
might narrowly pass the relative test alone — is still excluded. Fail either and the reflection is
dropped from the SCORED candidate pool before the intersection runs: only part of its rocking curve
was actually measured, so comparing it would compare a partial integration to an implicitly-full
one.

**Getting R.** Once the intersection is fixed, `align(solution, pattern, plan)` gathers
`calculated`/`observed`/`sigmas` onto that shared axis (index-gather, no further filtering). The
conventional Bragg residual:

```{math}
R_{\mathrm{obs}} = \frac{\sum |\sqrt{I_{\mathrm{obs}}} - \sqrt{I_{\mathrm{calc}}}|}
{\sum \sqrt{I_{\mathrm{obs}}}}, \quad \text{restricted to } I_{\mathrm{obs}} > 3\sigma
```

— an extra runtime cut on top of the SCORED-set selection above: even a reflection that passed the
`rsg`/`dsg` fully-integrated test only enters this particular sum if its measured intensity clears
the conventional {math}`3\sigma` significance threshold. `R_obs` is reported at every step but is
not itself minimized; see [Refinement](#refinement) for `w_rbragg`, the quantity actually optimized.

## Refinement

Refinement holds a settled `Plan` fixed and differentiates a scalar objective back to the
structural parameters through the complete forward chain:

```text
RefinableParams
  -> constrain (crystallographic hard constraints)                  -> PhysicalState
  -> expand_asu (space-group expansion)                             -> Fgb (structure_factors)
  -> per rotation: build_bloch_system -> propagate -> intensities   -> AlignedIntensities
  -> optional molecular constraints, components, penalties
  -> scalar objective (w_rbragg, Klar et al. 2023) + soft penalty terms
```

`w_rbragg` is the directly minimized, scaling-optimized weighted residual; the conventional Bragg
{math}`R_{\mathrm{obs}} = \sum |{\sqrt{I_{\mathrm{obs}}}} - \sqrt{I_{\mathrm{calc}}}| / \sum
\sqrt{I_{\mathrm{obs}}}`, restricted to {math}`I_{\mathrm{obs}} > 3\sigma`, is reported at every step
as the standard crystallographic residual but is not itself optimized; each is independently
scale-optimized against the same simulated intensities. Changes to the discrete problem — beam
selection, orientation, thickness, tilt coupling — are `Plan -> Plan` preprocessing steps and are
held fixed during refinement; refinement only varies the structural parameters and any composed
model components. Composition follows a fixed taxonomy: a value supplied to the forward model is a
`ModelComponent`; an invariant enforced exactly is a `ConstraintTransform` (or lives inside
`constrain`); an additive soft cost is a `PenaltyTerm`; a choice of which parameters vary is a
`TrainableSpec`. The optimizer loop (`engine/refine.py`) is the only stateful, mutating corner of
the codebase: it clones selected trainable leaves into fresh optimizer variables and never mutates
caller-owned parameters in place. See [Refinement](refinement.md) for the config surface and CLI
behaviour, and [Preprocessing](preprocessing.md) for how the `Plan` refinement holds fixed is
itself constructed.

## Reproducibility and provenance

Every stage that consumes a checkpoint verifies it before use rather than trusting it. `experiment.lock`
pins the raw structure and experimental-data files by content hash. `plan.lock` binds a preprocessing
checkpoint to that input identity, the preprocessing-determining projection of the resolved
configuration, the exact ordered recipe that produced it, and the producing code version.
`refinement.lock` extends the same discipline to the refinement stage: it records the `plan.lock`
this run refined from, the refinement-determining configuration, the code version, and hashes of the
refined outputs, so a refined structure is independently verifiable against exactly what produced
it. None of this is a performance optimization incidentally used for provenance; the reverse is
true — the mechanism exists to make a result verifiable, and checkpoint reuse is a consequence of
that verification succeeding. See [Reproducibility](reproducibility.md) for the full guarantee and
its limits.

## Observability

The pure core communicates progress by returning or emitting typed `Event` values
(`observability.py`); it runs correctly with no logger attached (`NULL_LOGGER`). Logger backends —
console, CSV, Weights & Biases, Comet — are attached at the application boundary and import their
vendor SDKs lazily. Scientific results are never routed through the diagnostic `logging` channel,
and no step reads back another step's logged output as pipeline state; a log is a write-only report.
See [Observability](observability-guide.md).

## Design invariants

The following properties are treated as defects when violated, independent of whether a forward
value looks numerically correct:

- **Differentiability.** The forward path is autograd-differentiable end to end, from
  `RefinableParams` to the scalar objective. An in-place operation on a leaf tensor, a stray
  `.item()`/`.detach()`, or non-differentiable indexing on the differentiated path is a defect even
  when the forward value is unchanged.
- **Determinism.** Identical inputs produce identical outputs; the simulation carries no hidden
  state across calls.
- **Precision as a boundary condition.** The Bloch solve runs at float32/complex64
  throughout (`core.solver.propagate`); precision is not threaded ad hoc through individual
  kernels.
- **Device/dtype preservation.** Kernels preserve the device and dtype of their inputs; the
  authoritative device for a forward pass is `RefinableParams`'s own device, and invariants are
  co-located onto it at the point of use.
- **Strategy as value, not as class hierarchy.** `SolverMethod` is a literal selecting between
  interchangeable propagators over one shared `BlochSystem`; geometry logic is not duplicated per
  solver.
- **Caching discipline.** Geometry-only lookup structures (`StructureFactorGather`, `BeamPlan`
  constants) may be cached; differentiable values that can go stale, most importantly
  {math}`F_{\mathbf{g}}`, are never cached.

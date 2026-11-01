# Review: `feat/isotropic-mosaicity`

## Scope, and what I did not check

The branch replaces an axis-only moving-average mosaicity model with one that solves the
calculation at several perturbed orientations and sums the intensities. Five samples by
default, placed by `blochwave.mosaicity_placement`, which chooses between `golden_spiral`
and `symmetric`.

Everything below is about the code's behaviour against the code's own stated intentions.
I verified each claim by running it. Where I report a number, the command that produced
it is in the notebook alongside this review.

One thing I deliberately do not claim. The code says `sigma_degrees` is the standard
deviation of a two-dimensional isotropic Gaussian orientation distribution, and calls
that "the usual crystallographic mosaicity convention". I have not verified that the
PETS2 apparent mosaicity the code feeds into it is that quantity. It is scraped from
free text with no definition recorded:

```python
# io/pets.py:165-171
def _mosaicity(block: gemmi.cif.Block) -> float | None:
    """PETS2 apparent mosaicity in degrees from measurement details."""
```

`record.py:207` is a bare `mosaicity_degrees: float | None = None`. So I make no claim
about whether the absolute amount of spread is right. The first finding does not need
one.

---

## 1. The two placements disagree with each other by a factor of √2

This is a contradiction inside the repository, provable without reference to any
convention.

A 2D isotropic Gaussian of standard deviation `s` has mean squared tilt `2s^2` and mean
fourth power `8s^4`. Those are the two numbers the repository's own test asserts, in
`test_symmetric_mosaic_placement_matches_the_gaussian_moments`.

Measured, for the same `sigma` and sample count:

```
placement        <rho^2>/s^2   <rho^4>/s^4
symmetric             2.0000        8.0000
golden_spiral         1.0260        2.0869     (5 samples)
golden_spiral         1.0000        2.0000     (1001 samples)
```

The two placements are presented in the docstrings as alternative discretisations of one
distribution. They model spreads differing by √2. Both are reachable from config, and
only one of them has a moment test. Changing `mosaicity_placement` changes the result,
not the precision of the result.

### Why `golden_spiral` comes out that way

It applies its own density twice. The placement is inverse-CDF sampling on evenly spaced
quantiles:

```python
# orientation.py:239
polar = sigma_degrees * np.sqrt(-2.0 * np.log(1.0 - (k + 0.5) / samples))
```

That is stratified sampling. Each sample stands for an equal slice of probability,
`1/N`, so the samples must be combined with equal weight. The code weights them by the
density as well:

```python
# orientation.py:195-199
if placement == "symmetric":
    weights = np.full(samples, 0.5 / (samples - 1))
    weights[0] = 0.5
else:
    weights = np.exp(-(polar**2) / (2.0 * sigma_degrees**2))
```

You can see the double application without any statistics. Substitute the first
expression into the second and the `sigma` cancels, leaving `1 - (k+0.5)/N`. For the
default of five samples the weights are `0.9, 0.7, 0.5, 0.3, 0.1`, and they are the same
numbers at `sigma = 0.3` as at `sigma = 3.0`. A quantity documented as "each sample's
relative Gaussian density" for a given sigma does not depend on sigma.

### Why the docstring's defence does not hold

The docstring says the weighting corrects a bias that quantile placement leaves at small
sample counts. Growing the sample count settles that. A small-sample correction would
approach the target as samples rise. This moves away from it:

```
samples   <rho^2>/s^2   <rho^4>/s^4   weighting
      5        1.0260        2.0869    as shipped
     21        1.0020        2.0022    as shipped
   1001        1.0000        2.0000    as shipped
      5        1.8647        5.8962    equal weights
   1001        1.9993        7.9996    equal weights
 target        2.0000        8.0000
```

Equal weights converge on the target. The shipped weighting converges on something else.

### Which one is wrong

I cannot tell you from the code. If the PETS convention matches `symmetric`, then
`golden_spiral` under-represents the spread and the fix is to use equal weights. If it
matches `golden_spiral`, then `symmetric` and its test are wrong instead. What is not
tenable is shipping both as interchangeable.

This survived because the two tests `golden_spiral` does have check that the weights
fall off with distance and that samples reach past sigma. The bug satisfies both.

---

## 2. Interleaving the mosaic samples silently drops strong beams

Selecting the relevant beams for every orientation would be slow, so the code partitions
the tilts into chunks and uses, for each chunk, the union of the sets at its two
endpoints. That is valid only when consecutive tilts are adjacent in angle, so the
endpoints bracket what lies between them. The existing orientation sweep satisfies this
because it advances monotonically.

The branch flattens the sweep and the mosaic samples with the mosaic sample as the fast
index:

```python
# beams.py:155-158
plain_tilts = compose_mosaic_tilts(plain_tilts, mosaic_tilts)
# index = r * M + m (compose_mosaic_tilts): the same per-mosaic-sample weights repeat for
# every rocking tilt.
reduction = WeightedSum(weights=torch.tensor(np.tile(mosaic_weights, n_rocking)))
```

With `index = r*M + m`, two consecutive entries are the same sweep position at different
mosaic directions. They are not neighbours in angle, so the endpoints no longer bracket
the interior and the union misses beams that matter in the middle of a chunk. A beam
missing from a chunk's set is not approximated; it contributes zero.

```
sweep only, adaptive policy:      0 / 1326 missing  (0.00%)
sweep only, fixed 12 segments:    0 / 1326 missing  (0.00%)
composed,   adaptive policy:    189 / 6662 missing  (2.84%)
composed,   fixed 12 segments:  187 / 6662 missing  (2.81%)
```

Worst case is 8 beams missing at a single tilt. The selection keeps beams near the
diffraction condition, so what is dropped is from the strong end, not the negligible
tail.

Two things make this worse than 2.84% sounds.

The shipped configuration always partitions (`program.py:1239` passes the policy
unconditionally), while every new test for the composed plans uses the unpartitioned
path. The combination that ships is the one not tested.

And the loss grows with sigma, which is the direction refinement pushes:

```
sigma as built:  201 / 6765 missing  (2.97%), worst  9
sigma x 1.414:   264 / 6801 missing  (3.88%), worst 10
sigma x 2.000:   450 / 6889 missing  (6.53%), worst 12
```

The beams that should switch on as sigma grows are the ones carrying the gradient with
respect to sigma, so the gradient is biased rather than merely noisy.

Making the mosaic sample the slow index keeps each chunk within one sample and restores
monotonicity. It is a coordinated change: `components.py:426` and the `np.tile` above
both encode the present ordering.

These figures come from a scratch harness rather than a test in the suite, so the first
thing to do is reproduce the loss in-tree, before changing the ordering.

---

## 3. The mosaic weights are dropped when a plan is written to disk

```python
# serialize.py:188-191
def _dump_reduction(reduction: TiltReduction) -> dict[str, Any]:
    if isinstance(reduction, MosaicSmoothed):
        return {"kind": "mosaic", "samples": reduction.samples}
    return {"kind": "plain"}
```

`WeightedSum` has no arm, so it falls through to `"plain"`, and loading maps `"plain"`
back to `PlainSum()`. The tilts are written out, so the geometry survives and nothing
raises. Only the weights go, and the samples are then summed equally.

The failure is the second run. The first writes the plan and its lock. An unchanged
re-run finds the lock valid, reuses the plan, and computes something different. The
`resume` path has the same hole. The lock is not at fault: the digest covers the whole
`blochwave` block, so it correctly reports that the inputs match. The file is lossy.

On magnitude, the raw intensity change is dominated by a single uniform factor, and a
uniform factor is absorbed by the fitted intensity scale, which `WeightedSum`'s own
docstring states. The number that matters is what survives a best-fit rescale, measured
end to end through the real write and read path: about 16% maximum and 6% mean. That is
far above refinement noise.

There is a round-trip test for `MosaicSmoothed` and none for `WeightedSum`.

---

## 4. At the default settings the refined sigma cannot move more than about 4%

Sigma is kept positive through `softplus`, so `d_sigma/d_raw` is `sigmoid(raw)`, which is
small when sigma is small. At `sigma = 0.05` the raw value is about -2.97 and the slope is
0.049. Adam caps each step at roughly the learning rate in raw space, so total raw travel
is bounded by `steps * lr` whatever the gradient says.

With the defaults, `steps = 40` (`schema.py:329`) and `lr = 1e-3` (`schema.py:262`):

```
sigma0    d_sigma/d_raw   best-case sigma   change
 0.050           0.0488          0.051988    3.98%
 0.018           0.0178          0.018728    4.04%
```

That assumes a maximally favourable gradient of the same sign on every step, so it is a
ceiling, not an estimate. The refined sigma lands within about 4% of its starting value
for almost any data, and the reported result reads as refinement agreeing with PETS
rather than as refinement having been unable to move.

If finding 1 is resolved in favour of `symmetric`, the correct answer is about 41% away,
which this budget cannot reach even in principle.

A per-component learning rate or a different parameterisation fixes it. At minimum,
report how far the parameter actually travelled.

---

## 5. The documentation describes the previous model under the new name

The docs change is a rename of `mosaicity` to `incoherent_mosaicity`. The surrounding
prose was not updated and is now false. `docs/preprocessing.md:226-242` still gives the
old moving-average span formula, still says the curve is "averaged over that span", and
still says:

> This uses the existing N Bloch wave solves rather than adding orientations, so enabling
> mosaicity adds no preprocessing or refinement cost.

The shipped model multiplies the solves by `mosaicity_samples`, five by default.
`docs/workflow.md:47-50` repeats it: "This does not add Bloch-wave solves."

The repository contradicts itself. `config/schema.py:98-100` has it right:

```python
# Each additional sample is a whole extra Bloch wave calculation solved through the full
# thickness, so this scales preprocessing/refinement runtime linearly and only matters when
# incoherent_mosaicity is true.
mosaicity_samples: int = 5
```

Someone reading the docs will enable the flag expecting it to be free, get a fivefold
bill, and believe the model smooths along one axis. `mosaicity_samples`,
`mosaicity_placement` and `refinement.trainable.mosaicity_sigma` appear nowhere in
`docs/`.

---

## 6. Asking to refine sigma without enabling mosaicity does nothing, quietly

```python
# program.py:789-798
resolved = resolve_dataset_mosaicity(
    cfg.blochwave.incoherent_mosaicity,
    ...
)
if not isinstance(resolved, IsotropicMosaicity):
    continue
```

`resolve_dataset_mosaicity` returns `None` as soon as the switch is false, so every
dataset hits `continue`. Setting `refinement.trainable.mosaicity_sigma: true` while
`blochwave.incoherent_mosaicity` stays false yields no component, no event and no
warning. The run completes and the user concludes mosaicity was refined and found
unimportant.

The same `continue` covers a dataset whose PETS mosaicity is exactly zero, which some of
the fixtures are. That case is documented; the switch case is not. Other config pairs are
cross-validated at load, so rejecting this one there would be consistent.

---

## 7. `mosaicity_samples: 1` creates a parameter that cannot move

```python
# components.py:424
delta = _in_plane_rotation(azimuth, (scale - 1.0) * polar)  # polar is [0.0]
```

With one sample the only polar angle is zero, so this is the identity for every sigma and
the gradient is structurally zero. The validator accepts `samples = 1`, the summary still
reports mosaicity as enabled, and the refinement event still reports a sigma as though it
had been fitted. The constructor checks the polar list is non-empty and the initial sigma
positive, but not that any angle is non-zero.

---

## 8. The sigma validity checks do not match what the arithmetic needs

Two cases with one cause. Neither check asks whether sigma is a value the arithmetic can
divide by.

```python
# specs.py:328
if not (self.sigma_degrees >= 0.0):  # catches NaN too
    raise ValueError("sigma_degrees must be finite and non-negative")
```

`inf >= 0.0` is true, so infinity is accepted while the message promises otherwise. The
tilts come out non-finite and the weights are all NaN.

The second case is quieter and worse. `orientation.py:192` returns early only on
`sigma_degrees == 0.0` exactly, and `orientation.py:199` divides by `sigma_degrees**2`.
That square underflows to zero for any sigma below about 2.2e-162, so the weights become
`0/0`:

```
     sigma    sigma**2   tilts finite  weights finite
    1e-100   1.00e-200           True            True
    1e-160   1.00e-320           True            True
    1e-170    0.00e+00           True           False
    5e-324    0.00e+00           True           False
```

The tilts stay finite there, because the polar angles underflow to zero and `cos(0)` is 1.
So the geometry looks healthy while the weights are NaN, which is the failure that is
hardest to notice: nothing raises, and the NaN only appears once it has propagated into
the intensities.

Neither case is reachable from a real instrument. `resolve_dataset_mosaicity` guards with
`np.isfinite`, and no PETS file reports 1e-170 degrees. Both are value types failing to
enforce their own stated invariants, and the fix for both is the same: check that sigma is
finite and large enough to square, rather than checking `>= 0.0` and `== 0.0`.

---

## An open question for someone who knows the instrument

I could not settle this from the code, and it may matter more than finding 1.

The default geometry is continuous rotation (`record.py:202`). The sweep tilts are about
x, with the goniometer axis offset azimuthally from x (`record.py:209-212`), and the
sweep half-width is taken from the PETS integration semi-angle rather than from config.
So the frame integration already sweeps one direction, and isotropic mosaicity then adds
spread in every direction, including that one.

Is the PETS apparent mosaicity already net of the rotation sweep? If it is, applying it
isotropically on top double-counts along the rotation direction, and that is a larger
problem than a √2 in the weighting.

---

## Smaller notes

`refined_components.npz` stores the `softplus` pre-image of sigma rather than degrees, and
nothing reads it back. The inference path does not load that file, so an `infer` run after
a refined run evaluates at the original PETS sigma. This matches the existing thickness
network, but it means the JSONL event is the only place the branch's headline output can
be consumed.

`_tilt_geometry` (`forward.py:725-740`) recomputes a 3x3 pseudo-inverse, the g-vectors and
a host-to-device transfer on every forward step, per rotation and per chunk. Nothing in it
depends on sigma. Correct, and a synchronisation point in the hot loop that could be
computed once per plan.

`_positive_inverse` raises "thickness values must be positive" when reached from
mosaicity. `resolve_dataset_mosaicity` still defaults to `axis_smoothed` though every
caller passes `isotropic` explicitly, its `degrees_override` has no caller, and the branch
removed the only test of that path, leaving it unreachable from config and untested.

---

## What is sound

Checked by running, and correct.

The new tilted geometry reproduces the existing path exactly, including the shortcut that
pulls the tilt out of the pseudo-inverse, verified against an exact per-tilt reference for
a non-orthonormal orientation. The trainable sigma move is exactly a rebuild at the new
sigma, agreeing to 2.2e-16, because the two rotations share an axis. The weight tiling
matches the index ordering it documents. Gradients flow end to end and are non-zero.
Replacing the two tilt-dependent fields leaves no stale state elsewhere in the batch.
Chunk slicing feeds the geometry in the order it was built. `symmetric`'s moment claims
are exact for every sample count from 4 up. Config identity is right: the digest covers
the new fields, so changing them invalidates the cached plan, and the rename raises a hard
validation error rather than being ignored. The new component tests are well aimed at the
plumbing, which is why the gaps sit where they do: the default placement's moments, the
weight round-trip, and beam coverage when partitioning meets mosaicity.

---

## Tests

898 pass, 3 skip. One failure,
`test_propagate_zero_thickness_returns_initial_wavefunction[bloch_eigen]`, a complex64
against complex128 assertion. It fails identically on `main` in a separate worktree, so it
predates this branch. `mypy` and `ruff` are clean. The single end-to-end test does not
exercise mosaicity, because the feature defaults to off.

---

## Order I would fix them in

Findings 1 and 2 change computed results on the path that ships, and should block.

Finding 1 needs a decision about the convention first, then a moment test for whichever
placement survives. Finding 2 needs the index ordering swapped in three coordinated
places, with an in-tree test written first so the fix has something to prove.

Finding 3 belongs in the same pass as finding 1: giving `golden_spiral` equal weights
makes the default case coincide with a plain sum, which hides the serialization bug rather
than fixing it, and `symmetric` would still be corrupted.

Findings 4 and 5 are both cases of the code and its description disagreeing, and both lead
a user to a wrong conclusion about a result. Findings 6, 7 and 8 are validation gaps and
are quick.

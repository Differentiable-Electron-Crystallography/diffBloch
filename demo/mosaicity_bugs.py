#!/usr/bin/env python
"""Reproduce the isotropic-mosaicity bugs with real numbers, using the branch's own code.

    .venv/bin/python demo/mosaicity_bugs.py

Read-only: imports the shipped functions and writes nothing. Each section prints the
measured value beside the analytic target so the gap is checkable by eye.

Covered here: the golden_spiral density double-count, the WeightedSum checkpoint
round-trip, the softplus step-budget ceiling, the samples=1 no-op, and the
non-finite sigma guard.

Not covered here, needing a full crystal setup or no numbers at all: the coupling
boundary-union beam loss (reproduce with a UnionCoupling policy over a composed tilt
set), and docs/preprocessing.md:241 still describing the old moving-average model.
"""

import math

import numpy as np
import torch

from diffBloch.core.products import WeightedSum, reduce_tilts
from diffBloch.preprocess.orientation import (
    isotropic_mosaic_directions,
    isotropic_mosaic_tilts,
)
from diffBloch.preprocess.serialize import _dump_reduction, _load_reduction
from diffBloch.specs import IsotropicMosaicity


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------------------
# BUG 1  golden_spiral applies the Gaussian density twice.
#
# orientation.py:239 places the polar angles by the inverse Rayleigh CDF at
# quantiles (k + 0.5) / samples:
#
#     polar = sigma * sqrt(-2 * log(1 - (k + 0.5) / samples))
#
# That is a stratified-quantile estimator, which is only correct with EQUAL
# weights: each sample already carries probability mass 1/N. orientation.py:199
# then weights each sample by the Gaussian density on top:
#
#     weights = np.exp(-(polar**2) / (2.0 * sigma_degrees**2))
#
# Substitute the first into the second and the sigma cancels, leaving
# 1 - (k+0.5)/N. The radial density is applied a second time, so the sampled
# distribution is Rayleigh(sigma/sqrt(2)), not Rayleigh(sigma).
# ---------------------------------------------------------------------------
rule("BUG 1  golden_spiral double-counts the Gaussian density")

SIGMA = 0.3  # degrees
N = 5  # blochwave.mosaicity_samples default

_, weights, polar = isotropic_mosaic_tilts(SIGMA, N, "golden_spiral")
quantiles = (np.arange(N) + 0.5) / N

print(f"sigma = {SIGMA} deg, samples = {N}, placement = golden_spiral (the default)\n")
print(f"{'k':>2} {'quantile':>9} {'polar (deg)':>12} {'weight':>9} {'1-quantile':>11}")
for k in range(N):
    print(
        f"{k:>2} {quantiles[k]:>9.4f} {polar[k]:>12.5f} "
        f"{weights[k]:>9.5f} {1 - quantiles[k]:>11.5f}"
    )

# The weights are a function of k and N alone, so they carry no information about
# the distribution they claim to be the density of.
print(f"\nweights == 1 - (k+0.5)/samples exactly?   {np.allclose(weights, 1 - quantiles)}")
_, w_x10, _ = isotropic_mosaic_tilts(SIGMA * 10, N, "golden_spiral")
print(f"weights unchanged at sigma = {SIGMA * 10} deg?     {np.allclose(weights, w_x10)}")
print("(documented as 'each sample's relative Gaussian density' for this sigma)")

# A 2-D isotropic Gaussian of std sigma has <rho^2> = 2 sigma^2 and <rho^4> = 8 sigma^4.
# Growing N is the decisive test: a small-sample correction would converge toward the
# target. This converges away from it, onto Rayleigh(sigma/sqrt(2))'s 1 and 2.
rule("BUG 1  moments against their analytic targets, as samples grow")
print(f"{'samples':>8} {'<rho2>/s2':>11} {'<rho4>/s4':>11}   weighting")
for n in (5, 21, 101, 1001):
    _, w, p = isotropic_mosaic_tilts(SIGMA, n, "golden_spiral")
    wn = w / w.sum()
    print(
        f"{n:>8} {np.sum(wn * p**2) / SIGMA**2:>11.4f} "
        f"{np.sum(wn * p**4) / SIGMA**4:>11.4f}   as shipped"
    )
for n in (5, 1001):
    _, _, p = isotropic_mosaic_tilts(SIGMA, n, "golden_spiral")
    print(
        f"{n:>8} {np.mean(p**2) / SIGMA**2:>11.4f} "
        f"{np.mean(p**4) / SIGMA**4:>11.4f}   uniform (the fix)"
    )
print(f"{'target':>8} {2.0:>11.1f} {8.0:>11.1f}")

_, w_sym, p_sym = isotropic_mosaic_tilts(SIGMA, 5, "symmetric")
wn_sym = w_sym / w_sym.sum()
print(
    f"{5:>8} {np.sum(wn_sym * p_sym**2) / SIGMA**2:>11.4f} "
    f"{np.sum(wn_sym * p_sym**4) / SIGMA**4:>11.4f}   symmetric placement"
)
print("\nsymmetric is exact and is the only placement with a moment test in the suite.")

rule("BUG 1  the consequence, including for the shipped fixtures")
_, w, p = isotropic_mosaic_tilts(SIGMA, 1001, "golden_spiral")
ratio = math.sqrt(np.sum((w / w.sum()) * p**2) / 2.0) / SIGMA
print(f"effective / requested sigma = {ratio:.4f}   (1/sqrt(2) = {1 / math.sqrt(2):.4f})\n")
print(f"{'requested (deg)':>16} {'modelled (deg)':>16}")
for s in (SIGMA, 0.050, 0.018):  # 0.050 and 0.018 are the repo's PETS fixtures
    print(f"{s:>16.3f} {s * ratio:>16.3f}")
print(
    "\nThe two placement options therefore model spreads differing by sqrt(2),"
    "\nso mosaicity_placement changes the physics, not just the quadrature."
)


# ---------------------------------------------------------------------------
# BUG 2  WeightedSum does not survive a checkpoint round-trip.
#
# serialize.py:188 has no WeightedSum arm and falls through to {"kind": "plain"};
# serialize.py:199 maps that back to PlainSum(). The composed tilts ARE persisted,
# so the geometry survives a reload and nothing raises. Only the weights vanish.
# ---------------------------------------------------------------------------
rule("BUG 2  WeightedSum is written to the checkpoint as a plain sum")

n_rocking, n_mosaic = 3, 5
_, mosaic_weights, _ = isotropic_mosaic_tilts(SIGMA, n_mosaic, "golden_spiral")
# beams.py:158 builds exactly this: per-sample weights tiled across the rockings.
original = WeightedSum(weights=torch.tensor(np.tile(mosaic_weights, n_rocking)))

payload = _dump_reduction(original)
restored = _load_reduction(payload)
print(f"written   : {payload}")
print(f"read back : {restored}")
print(f"preserved : {isinstance(restored, WeightedSum)}")

rule("BUG 2  how much the reloaded plan's intensities move")

rng = np.random.default_rng(0)
decay = np.tile(np.exp(-0.5 * np.arange(n_mosaic)), n_rocking)[:, None, None]
stacked = torch.tensor(
    rng.uniform(0.5, 1.5, (n_rocking * n_mosaic, 1, 12)) * decay, dtype=torch.float64
)

before = reduce_tilts(stacked, original).squeeze().numpy()
after = reduce_tilts(stacked, restored).squeeze().numpy()

# Most of the raw change is a single uniform factor, and WeightedSum's own docstring
# says the overall weight scale is absorbed by the fitted intensity scale. So the raw
# number overstates the damage; the honest measure is what survives a best-fit rescale.
raw = np.abs(after / before - 1)
scale = (after @ before) / (after @ after)
rescaled = np.abs(scale * after / before - 1)

print(f"raw change             : max {raw.max() * 100:5.1f}%   mean {raw.mean() * 100:5.1f}%")
print(f"absorbed scale factor  : {scale:.4f}  (the intensity fit soaks this up)")
print(
    f"change after rescale   : max {rescaled.max() * 100:5.1f}%   "
    f"mean {rescaled.mean() * 100:5.1f}%   <-- the real effect"
)
print(
    "\nSynthetic intensities understate it. Measured end to end through real"
    "\nwrite_plan/read_plan/build_engine the surviving change is ~16% max, ~6% mean,"
    "\nwhich is well above refinement noise for wR2."
)
print(
    "\nSame config, same validated plan.lock, different answer, no warning."
    "\nBoth the reuse and resume paths in app/program.py take the lossy branch."
)


# ---------------------------------------------------------------------------
# BUG 3  the refined sigma cannot move far at the default step budget.
#
# sigma is parameterised through softplus, so d sigma / d raw = sigmoid(raw) is tiny
# for a sigma of ~0.05 deg. Adam caps each step at about lr in raw space, so
# steps * lr bounds the total raw movement no matter what the gradient says.
# ---------------------------------------------------------------------------
rule("BUG 3  the softplus step-budget ceiling (steps = 40, lr = 1e-3)")

STEPS, LR = 40, 1e-3  # config/schema.py:329 and :262
print(f"{'sigma0':>8} {'d_sig/d_raw':>12} {'best-case sigma':>16} {'change':>9}")
for s0 in (0.050, 0.018, 0.300):
    raw0 = math.log(math.expm1(s0))
    s1 = torch.nn.functional.softplus(torch.tensor(raw0 + STEPS * LR)).item()
    slope = torch.sigmoid(torch.tensor(raw0)).item()
    print(f"{s0:>8.3f} {slope:>12.4f} {s1:>16.6f} {(s1 / s0 - 1) * 100:>8.2f}%")
print(
    "\nThat is a maximally favourable, same-sign gradient on every step, so it is a"
    "\nceiling and not an estimate. The refined sigma therefore lands within ~4% of"
    "\nthe PETS value almost regardless of the data, and reads as agreement with PETS."
)
print(
    f"Bug 1 puts the true optimum near +{(1 / ratio - 1) * 100:.0f}%, which this"
    "\nbudget cannot reach."
)


# ---------------------------------------------------------------------------
# BUG 4  samples = 1 leaves a parameter that cannot move.
# ---------------------------------------------------------------------------
rule("BUG 4  mosaicity_samples = 1 is a no-op with a live parameter")

az1, po1 = isotropic_mosaic_directions(SIGMA, 1, "golden_spiral")
print(f"samples=1 -> azimuth {az1}, polar {po1}")
tilts_a, _, _ = isotropic_mosaic_tilts(0.1, 1, "golden_spiral")
tilts_b, _, _ = isotropic_mosaic_tilts(9.9, 1, "golden_spiral")
print(f"tilt identical at sigma 0.1 and 9.9? {np.allclose(tilts_a, tilts_b)}")
print(
    "\nEvery polar angle is 0, so the trainable rotation is the identity for any"
    "\nsigma and the gradient is structurally zero. The validator accepts samples=1,"
    "\nthe summary still prints mosaicity as enabled, and IsotropicMosaicityRefined"
    "\nstill reports a sigma as though refinement had settled there."
)


# ---------------------------------------------------------------------------
# BUG 5  the finiteness guard does not reject +inf.
# ---------------------------------------------------------------------------
rule("BUG 5  IsotropicMosaicity accepts a non-finite sigma")

spec = IsotropicMosaicity(sigma_degrees=math.inf)  # specs.py:328 tests `>= 0.0`
print(f"accepted: {spec}")
t, w, _ = isotropic_mosaic_tilts(math.inf, 5, "golden_spiral")
print(f"resulting tilts finite? {np.isfinite(t).all()}   weights: {w}")
print(
    "\n`not (sigma >= 0.0)` catches NaN but inf passes, while the error text promises"
    "\n'finite and non-negative'. Config cannot reach this (resolve_dataset_mosaicity"
    "\nguards with np.isfinite), so it is a broken invariant rather than a live path."
)

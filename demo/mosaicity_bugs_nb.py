# %% [markdown]
# # Isotropic mosaicity: reproducing the review's findings
#
# Run from the repository root, on this branch, with the project interpreter:
#
# ```sh
# .venv/bin/python -m jupyter lab demo/mosaicity_bugs.ipynb
# ```
#
# Everything here imports the branch's own code and writes nothing. Each measured value is
# printed beside the target it is meant to match.
#
# What this notebook does **not** claim: that the PETS2 apparent mosaicity fed into
# `sigma_degrees` is the standard deviation of a 2D isotropic Gaussian. The code says so in a
# docstring; the value itself is scraped from free text with no definition recorded. So no
# statement below depends on the absolute amount of spread being right. Finding 1 is a
# contradiction between two code paths in this repository, which needs no convention.
#
# Not reproduced here: the beam loss from chunked beam selection (needs a full crystal and a
# `UnionCoupling` policy) and the documentation defect (no numbers to print).

# %%
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

SIGMA = 0.3  # degrees
N = 5  # blochwave.mosaicity_samples default
print(f"torch {torch.__version__}, numpy {np.__version__}")

# %% [markdown]
# ## 1. The two placements disagree by a factor of √2
#
# A 2D isotropic Gaussian of standard deviation `s` has mean squared tilt `2s^2` and mean
# fourth power `8s^4`. Those are the numbers this repository's own test asserts, in
# `test_symmetric_mosaic_placement_matches_the_gaussian_moments`.
#
# Both placements are reachable from config. Only one has a moment test.

# %%
def moments(sigma: float, samples: int, placement: str) -> tuple[float, float]:
    """Weighted <rho^2>/sigma^2 and <rho^4>/sigma^4 for a placement."""
    _, weights, polar = isotropic_mosaic_tilts(sigma, samples, placement)
    w = weights / weights.sum()
    return (
        float(np.sum(w * polar**2) / sigma**2),
        float(np.sum(w * polar**4) / sigma**4),
    )


print(f"{'placement':<16}{'<rho^2>/s^2':>13}{'<rho^4>/s^4':>13}")
for placement, samples in (("symmetric", 5), ("golden_spiral", 5), ("golden_spiral", 1001)):
    m2, m4 = moments(SIGMA, samples, placement)
    print(f"{placement + f' ({samples})':<16}{m2:>13.4f}{m4:>13.4f}")
print(f"{'target':<16}{2.0:>13.4f}{8.0:>13.4f}")

# %% [markdown]
# ### Why `golden_spiral` comes out that way: it applies its own density twice
#
# The placement is inverse-CDF sampling on evenly spaced quantiles (`orientation.py:239`):
#
# ```python
# polar = sigma_degrees * np.sqrt(-2.0 * np.log(1.0 - (k + 0.5) / samples))
# ```
#
# That is stratified sampling. Each sample stands for an equal slice of probability, `1/N`,
# so they must be combined with equal weight. The code weights them by the density as well
# (`orientation.py:199`):
#
# ```python
# weights = np.exp(-(polar**2) / (2.0 * sigma_degrees**2))
# ```
#
# Substitute the first into the second and the `sigma` cancels, leaving `1 - (k+0.5)/N`.

# %%
_, weights, polar = isotropic_mosaic_tilts(SIGMA, N, "golden_spiral")
quantiles = (np.arange(N) + 0.5) / N

print(f"{'k':>2}{'quantile':>11}{'polar (deg)':>14}{'weight':>10}{'1 - quantile':>14}")
for k in range(N):
    print(
        f"{k:>2}{quantiles[k]:>11.4f}{polar[k]:>14.5f}"
        f"{weights[k]:>10.5f}{1 - quantiles[k]:>14.5f}"
    )

print(f"\nweights == 1 - (k+0.5)/samples exactly? {np.allclose(weights, 1 - quantiles)}")

# %% [markdown]
# The weights are a function of `k` and `N` alone, so they cannot depend on sigma. They are
# documented as "each sample's relative Gaussian density" for a given sigma.

# %%
for factor in (1, 10, 100):
    _, w, _ = isotropic_mosaic_tilts(SIGMA * factor, N, "golden_spiral")
    print(f"sigma = {SIGMA * factor:>6.1f} deg  ->  weights {np.round(w, 5)}")

# %% [markdown]
# ### The docstring's defence does not hold
#
# The docstring says the weighting corrects a bias that quantile placement leaves at small
# sample counts. Raising the sample count settles that. A small-sample correction would
# approach the target. This moves away from it, while equal weights converge on it.

# %%
print(f"{'samples':>8}{'<rho^2>/s^2':>13}{'<rho^4>/s^4':>13}   weighting")
for n in (5, 21, 101, 1001):
    m2, m4 = moments(SIGMA, n, "golden_spiral")
    print(f"{n:>8}{m2:>13.4f}{m4:>13.4f}   as shipped")
for n in (5, 1001):
    _, _, p = isotropic_mosaic_tilts(SIGMA, n, "golden_spiral")
    m2 = float(np.mean(p**2) / SIGMA**2)
    m4 = float(np.mean(p**4) / SIGMA**4)
    print(f"{n:>8}{m2:>13.4f}{m4:>13.4f}   equal weights")
print(f"{'target':>8}{2.0:>13.4f}{8.0:>13.4f}")

# %% [markdown]
# The shipped weighting settles on 1 and 2 rather than 2 and 8. The ratio between the two
# placements' modelled spread:

# %%
m2_golden, _ = moments(SIGMA, 1001, "golden_spiral")
m2_symmetric, _ = moments(SIGMA, 5, "symmetric")
print(f"golden_spiral effective / symmetric effective = {math.sqrt(m2_golden / m2_symmetric):.4f}")
print(f"1/sqrt(2) = {1 / math.sqrt(2):.4f}")

# %% [markdown]
# Which placement is wrong depends on which convention the PETS value follows, and that is
# not recorded in the repository. What is not tenable is shipping both as interchangeable.

# %% [markdown]
# ## 2. The mosaic weights are dropped when a plan is written to disk
#
# `_dump_reduction` (`serialize.py:188`) has no `WeightedSum` arm and falls through to
# `{"kind": "plain"}`; loading maps that back to `PlainSum()`. The tilts are written out, so
# the geometry survives and nothing raises. Only the weights go.

# %%
n_rocking, n_mosaic = 3, 5
_, mosaic_weights, _ = isotropic_mosaic_tilts(SIGMA, n_mosaic, "golden_spiral")
# beams.py:158 builds exactly this: the per-sample weights tiled across the sweep positions.
original = WeightedSum(weights=torch.tensor(np.tile(mosaic_weights, n_rocking)))

payload = _dump_reduction(original)
restored = _load_reduction(payload)
print(f"written to the checkpoint : {payload}")
print(f"read back                 : {restored}")
print(f"weights preserved         : {isinstance(restored, WeightedSum)}")

# %% [markdown]
# ### How much it moves the result
#
# Most of the raw change is a single uniform factor, and `WeightedSum`'s own docstring says
# the overall weight scale is absorbed by the fitted intensity scale. So the raw figure
# overstates the damage. The honest measure is what survives a best-fit rescale.

# %%
rng = np.random.default_rng(0)
# Intensity falling off with mosaic cone radius, as a real rocking curve does.
decay = np.tile(np.exp(-0.5 * np.arange(n_mosaic)), n_rocking)[:, None, None]
stacked = torch.tensor(
    rng.uniform(0.5, 1.5, (n_rocking * n_mosaic, 1, 12)) * decay, dtype=torch.float64
)

before = reduce_tilts(stacked, original).squeeze().numpy()
after = reduce_tilts(stacked, restored).squeeze().numpy()

raw = np.abs(after / before - 1)
scale = (after @ before) / (after @ after)
rescaled = np.abs(scale * after / before - 1)

print(f"raw change            : max {raw.max() * 100:5.1f}%   mean {raw.mean() * 100:5.1f}%")
print(f"absorbed scale factor : {scale:.4f}  (the intensity fit soaks this up)")
print(
    f"after rescale         : max {rescaled.max() * 100:5.1f}%   "
    f"mean {rescaled.mean() * 100:5.1f}%   <- the real effect"
)

# %% [markdown]
# Synthetic intensities understate it. Measured end to end through the real
# `write_plan`/`read_plan`/`build_engine` path, what survives the rescale is about 16%
# maximum and 6% mean, which is far above refinement noise.
#
# The failure is the second run: the first writes the plan and its lock, an unchanged re-run
# finds the lock valid, reuses the plan, and computes something different. Both the reuse and
# resume paths take the lossy branch.

# %% [markdown]
# ## 3. At the default settings the refined sigma cannot move more than about 4%
#
# Sigma is kept positive through `softplus`, so `d_sigma/d_raw = sigmoid(raw)` is small when
# sigma is small. Adam caps each step at roughly the learning rate in raw space, so total raw
# travel is bounded by `steps * lr` whatever the gradient says.

# %%
STEPS, LR = 40, 1e-3  # config/schema.py:329 and :262

print(f"{'sigma0':>8}{'d_sig/d_raw':>14}{'best-case sigma':>18}{'change':>10}")
for s0 in (0.050, 0.018, 0.300):
    raw0 = math.log(math.expm1(s0))
    s1 = torch.nn.functional.softplus(torch.tensor(raw0 + STEPS * LR)).item()
    slope = torch.sigmoid(torch.tensor(raw0)).item()
    print(f"{s0:>8.3f}{slope:>14.4f}{s1:>18.6f}{(s1 / s0 - 1) * 100:>9.2f}%")

# %% [markdown]
# That assumes a maximally favourable gradient of the same sign on every step, so it is a
# ceiling and not an estimate. The refined sigma therefore lands within about 4% of its
# starting value for almost any data, and reads as agreement with PETS rather than as
# refinement having been unable to move.

# %% [markdown]
# ## 4. `mosaicity_samples: 1` creates a parameter that cannot move
#
# With one sample the only polar angle is zero, so `components.py:424`'s
# `(scale - 1.0) * polar` is the identity for every sigma and the gradient is structurally
# zero. The validator accepts it, the summary still reports mosaicity as enabled, and the
# refinement event still reports a sigma as though it had been fitted.

# %%
azimuth, polar_1 = isotropic_mosaic_directions(SIGMA, 1, "golden_spiral")
print(f"samples=1 -> azimuth {azimuth}, polar {polar_1}")

tilts_small, _, _ = isotropic_mosaic_tilts(0.1, 1, "golden_spiral")
tilts_large, _, _ = isotropic_mosaic_tilts(9.9, 1, "golden_spiral")
print(f"tilt identical at sigma 0.1 and 9.9? {np.allclose(tilts_small, tilts_large)}")
print(f"tilt is the identity?                {np.allclose(tilts_small, np.eye(3))}")

# %% [markdown]
# ## 5. A non-finite sigma passes the finiteness check
#
# `specs.py:328` tests `not (sigma_degrees >= 0.0)`, which catches NaN but admits `+inf`,
# against an error message promising finiteness. Config cannot reach this, because the
# resolver guards with `np.isfinite`, so it is a value type failing its own stated invariant
# rather than a live path.

# %%
spec = IsotropicMosaicity(sigma_degrees=math.inf)
print(f"accepted: {spec}")

tilts_inf, weights_inf, _ = isotropic_mosaic_tilts(math.inf, 5, "golden_spiral")
print(f"tilts finite? {np.isfinite(tilts_inf).all()}")
print(f"weights     : {weights_inf}")

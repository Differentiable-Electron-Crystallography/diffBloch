"""Coupling-policy forward-solver parity anchor: reproduce the private reference given its coupling.

This complements the from-scratch accuracy anchors in :mod:`test_anchor` with a **forward-solver
parity** check. It does not fit anything: it replays the ``diffBloch_private`` reference's *exact*
dynamical coupling (the ``union_splits=12`` non-adaptive tilt-segment union sets, cap ``|g|<2.05``,
``sg_max=0.01``, mean-inner-potential ``u0``) through our Bloch solver and reproduces the private
per-rotation ``R_obs`` from the hash-verified ``reference_results.json`` to ~1e-3 on every dumped
rotation.

Two things are reproduced together, because both were proven to be the private's behaviour:
- **Coupling** -- the per-segment union beam sets over disjoint tilt chunks (geometry proven
  byte-identical by ``compare_private_segments.py``; ``Fgb`` by ``compare_fgb.py``).
- **Tilt reduction (mosaicity)** -- the private smooths each reflection's rocking curve with a
  window-5 moving average before summing (config ``mosaicity: true``;
  ``diffraction_dataset.get_integrated_intensities`` -> ``moving_average``). This replay therefore
  gathers each matched reflection's per-tilt intensity into its **full** rocking curve across all
  segments, *then* applies the window-5 valid-mean smoothing -- the reduction must see the whole
  curve (the window exceeds a single segment's 3-4 tilts). That reassemble-then-smooth shape is
  exactly what a future ``SegmentedOrientationPlan`` will do inside the engine; here it is explicit.

Rot 61 was previously a "corner-beam outlier" under a plain-sum replay. That was a *reduction*
artifact, not a solve difference: its per-tilt ``|psi(4,0,5)|^2`` is byte-identical to the private,
and the window-5 moving-average's edge down-weighting (a sharp peak near the boundary of its coupled
range) accounts for the whole gap. With the faithful reduction it is no longer an outlier and joins
the tight set.

Opt-in ``e2e`` (excluded from ``just check``).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from diffBloch.config import load_experiment
from diffBloch.core.losses import optimal_scale, rbragg
from diffBloch.core.products import PatternBatch, align
from diffBloch.core.solver import Method
from diffBloch.engine import OrientationPlan, ScatteringGrid
from diffBloch.io import read_structure
from diffBloch.preprocess.experiment import RefinementSetup
from diffBloch.preprocess.plan import Plan
from diffBloch.preprocess.scoring import build_engine

pytestmark = pytest.mark.e2e

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "quartz_anchor"
REPLAY_ROOT = FIXTURE_ROOT / "parity_replay"

# Nominal 200 kV -- the energy the private ran at. Not wavelength2energy(0.0251)=199717: the PETS
# wavelength is rounded to 4 sig figs, and the near-cap beams are energy-sensitive, so the replay
# must use the private's exact beam energy to reproduce its numbers.
ENERGY_EV = 200_000.0

# The private's mosaicity moving-average window (config mosaicity: true hardcodes window_size=5).
MOSAICITY_WINDOW = 5

# Every dumped rotation reproduces the private R_obs to ~1e-3 once the faithful reduction is applied
# (the notebook chase matched all five to 4 decimals). The tolerance is loose enough for
# cross-platform eigensolver degeneracy while still catching a physics regression.
PARITY_ROTATIONS = (13, 27, 60, 61, 64)
PARITY_TOL = 2e-3


def _reference_r_obs() -> dict[int, float]:
    import json

    data = json.loads((FIXTURE_ROOT / "reference_results.json").read_text())
    return {int(d["rotation_idx"]): float(d["R_obs"]) for d in data["rotations"]}


def _smoothed_sum(curve: torch.Tensor) -> torch.Tensor:
    """Window-MOSAICITY_WINDOW valid-mean sum over the tilt axis: the private mosaicity integration.

    ``curve`` is ``(n_tilts, n_matched)``. The private applies a moving average (valid ``unfold``
    mean, zero-padded back) then sums; the zero-pad does not affect the sum, so this equals the sum
    of the ``n_tilts - window + 1`` window means. The edge windows (counts 1..window-1) down-weight
    a sharp peak sitting near the boundary of a reflection's coupled range -- the rot-61 effect.
    """
    if curve.shape[0] < MOSAICITY_WINDOW:
        return curve.sum(dim=0)
    return curve.unfold(0, MOSAICITY_WINDOW, 1).mean(dim=-1).sum(dim=0)


def _replay_r_obs(
    rotation: int,
    grid: ScatteringGrid,
    refinement: RefinementSetup,
    method: Method,
    tilts: np.ndarray,
) -> float:
    """Replay one rotation's private coupling + mosaicity through our solver; return Bragg R."""
    d = np.load(REPLAY_ROOT / f"rot_{rotation}.npz")
    observed = torch.tensor(d["exp_ints"], dtype=torch.float64)
    sigmas = torch.tensor(d["sigmas"], dtype=torch.float64)
    pattern = PatternBatch(
        hkl=torch.tensor(d["hkl_matched"], dtype=torch.int64), intensities=observed, sigmas=sigmas
    )
    thickness = float(np.asarray(d["thickness"]).reshape(-1)[-1])
    u0 = float(d["u0"])
    # Reassemble each matched reflection's full rocking curve: every tilt belongs to exactly one
    # segment (disjoint covers), and a reflection contributes at that tilt iff it is in the
    # segment's coupling union. One single-tilt solve per tilt places its matched intensities on the
    # full axis; the mosaicity smoothing then runs over the whole curve (never per-segment -- the
    # window exceeds a segment's tilt count).
    curve = torch.zeros(len(tilts), len(d["hkl_matched"]), dtype=torch.float64)
    for k in range(int(d["n_segments"])):
        seg_hkl = d[f"seg{k}_hkl"]
        for tilt_index in d[f"seg{k}_cover"]:
            op = OrientationPlan.build(
                grid,
                seg_hkl,
                pattern,
                energy=ENERGY_EV,
                thickness=(thickness,),
                orientation=d["orientation"],
                tilts=tilts[int(tilt_index)][None],
                u0=u0,
            )
            engine = build_engine(Plan(grid=grid, orientations=(op,)), refinement, method=method)
            with torch.no_grad():
                solution = engine.simulate(refinement.params)[0]
            aligned = align(solution, pattern, op.alignment)
            curve[int(tilt_index)].index_add_(0, op.alignment.pattern_index, aligned.calculated[0])
    _scale, r_obs = optimal_scale(_smoothed_sum(curve), observed, sigmas, metric=rbragg)
    return float(r_obs)


def test_quartz_coupling_parity() -> None:
    cfg, _lock = load_experiment(FIXTURE_ROOT)
    structure = read_structure(FIXTURE_ROOT / cfg.inputs.structure)
    grid = ScatteringGrid.from_cell(structure.unit_cell, g_max=cfg.numerics.g_max)
    refinement = RefinementSetup.from_structure(structure)
    tilts = np.load(REPLAY_ROOT / "tilts.npz")["tilts"]
    reference = _reference_r_obs()

    for rotation in PARITY_ROTATIONS:
        r_obs = _replay_r_obs(rotation, grid, refinement, cfg.solver.inference, tilts)
        assert r_obs == pytest.approx(
            reference[rotation], abs=PARITY_TOL
        ), f"rot {rotation}: replay R_obs {r_obs:.4f} vs reference {reference[rotation]:.4f}"

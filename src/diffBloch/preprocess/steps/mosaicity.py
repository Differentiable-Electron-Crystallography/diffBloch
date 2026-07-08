"""``mosaicity``: broaden the rocking-curve integration to model crystal mosaic spread.

A ``Plan -> Plan`` step that switches each rotation's tilt-axis reduction from a plain incoherent
sum (:class:`~diffBloch.core.products.PlainSum`) to a moving-average-broadened sum
(:class:`~diffBloch.core.products.MosaicSmoothed`). Crystal mosaicity smears each reflection's
rocking curve; the private models it as a ``window``-wide moving average of the per-tilt intensities
before the sum-over-tilts integration (``diffBloch_private`` ``DiffractionDataset.moving_average``).

It is a **modifier on top of** ``integrate_rocking_curve``
(:func:`~diffBloch.preprocess.steps.rocking_curve.integrate_rocking_curve`):
it only has meaning once the tilt set exists (a moving average over a single tilt is degenerate), so
it is ordered *after* that step and *before* the fits (fitting under the same integrated model used
at evaluation -- the fit/eval consistency invariant). Off by default: no ``mosaicity`` step = the
plain-sum reduction = today's behaviour, byte-identical.

Pure and geometry-preserving: the reduction is a per-orientation attribute, so this only
``dataclasses.replace``\\ s the reduction descriptor -- no beam plans are rebuilt (mirrors
``fit_thickness`` swapping only the thickness). Faithful to the private, which passes the mosaicity
setting into both the orientation fit and the final evaluation.
"""

from __future__ import annotations

from dataclasses import replace

from diffBloch.core.products import MosaicSmoothed
from diffBloch.engine.plan import OrientationPlan
from diffBloch.preprocess.pipeline import PlanStep, as_step
from diffBloch.preprocess.plan import Plan, require_orientation_plans
from diffBloch.specs import Mosaicity

__all__ = ["mosaicity"]


def mosaicity(spec: Mosaicity) -> PlanStep:
    """Return a ``Plan -> Plan`` step applying mosaicity broadening to the rocking-curve reduction.

    ``spec`` is a pre-validated :class:`~diffBloch.specs.Mosaicity` (``window >= 1`` is guaranteed,
    so this never re-validates that bound). Each orientation's tilt reduction becomes
    :class:`~diffBloch.core.products.MosaicSmoothed` with ``spec.window``. Because the broadening
    presupposes a rocking-curve tilt set, this raises if any orientation has fewer than two tilts
    (i.e. ``integrate_rocking_curve`` has not run) or if ``spec.window`` exceeds the tilt count --
    failing fast at plan construction rather than deep inside the forward pass.
    """

    def run(plan: Plan) -> Plan:
        orientations = tuple(_apply_one(op, spec.window) for op in require_orientation_plans(plan))
        return replace(plan, orientations=orientations)

    return as_step("mosaicity", spec, run)


def _apply_one(op: OrientationPlan, window: int) -> OrientationPlan:
    """Set one orientation's tilt reduction to the mosaicity sum, guarding the tilt count."""
    n_tilts = op.tilts.shape[0]
    if n_tilts < 2:
        raise ValueError(
            "mosaicity requires a rocking-curve tilt set; compose integrate_rocking_curve first "
            f"(found {n_tilts} tilt)"
        )
    if window > n_tilts:
        raise ValueError(f"mosaicity window {window} exceeds the {n_tilts} rocking-curve tilts")
    return replace(op, tilt_reduction=MosaicSmoothed(window))

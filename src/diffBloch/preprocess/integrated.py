"""Which reflections a continuous-rotation frame integrates: the Bragg crossing inside the sweep.

A frame of a continuous-rotation experiment integrates a reflection only when the crystal sweeps
it through its Bragg condition (``Sg = 0``) inside that frame's tilt range, with enough margin that
the rocking curve around the crossing lands inside the frame rather than spilling into the
neighbouring frames. These are the pure geometry pieces of that test: the tilt angle at which each
reflection crosses ``Sg = 0``, and the keep-mask given a rocking-curve half-width in ``Sg``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from diffBloch.core.dynamical.primitives import excitation_errors
from diffBloch.preprocess.orientation import rocking_curve_tilts
from diffBloch.specs import IntegrationGeometry

__all__ = ["bragg_crossing_angles", "integrated_mask", "sg_per_radian"]

# Excitation errors are sampled this finely across the sweep to locate the crossing; Sg is close
# to linear in the tilt over a frame, so linear interpolation between samples is accurate.
_CROSSING_SAMPLES = 21


def _require_rotation(integration: IntegrationGeometry) -> None:
    if integration.geometry != "continuous_rotation":
        raise ValueError(
            "integrated-reflection scoring is defined for continuous_rotation frames only, got "
            f"{integration.geometry!r}"
        )


def bragg_crossing_angles(
    g: NDArray[np.float64],
    *,
    energy: float,
    u0: float,
    semiangle: float,
) -> NDArray[np.float64]:
    """Tilt angle (degrees) at which each lab-frame ``g`` ``(N, 3)`` crosses ``Sg = 0``.

    The tilt sweeps ``[-semiangle, +semiangle]`` about the goniometer ``x`` axis exactly as
    :func:`~diffBloch.preprocess.orientation.rocking_curve_tilts` builds it. ``NaN`` where ``Sg``
    keeps one sign over the whole sweep (the reflection is never on its Bragg condition in this
    frame). With more than one crossing the one nearest the frame centre is returned.
    """
    g = np.asarray(g, dtype=np.float64)
    angles = np.linspace(-semiangle, semiangle, _CROSSING_SAMPLES)
    tilts = rocking_curve_tilts(semiangle, _CROSSING_SAMPLES)
    tilted = np.einsum("tij,nj->tni", tilts, g)  # g at each tilt: tilt @ g
    sg = excitation_errors(tilted.reshape(-1, 3), energy, u0=u0).reshape(_CROSSING_SAMPLES, -1)
    crossing = np.full(g.shape[0], np.nan)
    left, right = sg[:-1], sg[1:]
    changes = (np.sign(left) != np.sign(right)) | (left == 0.0)
    for n in np.flatnonzero(changes.any(axis=0)):
        steps = np.flatnonzero(changes[:, n])
        a, b = left[steps, n], right[steps, n]
        with np.errstate(divide="ignore", invalid="ignore"):
            fraction = np.where(a == b, 0.0, a / (a - b))
        candidates = angles[steps] + fraction * (angles[steps + 1] - angles[steps])
        crossing[n] = candidates[np.argmin(np.abs(candidates))]
    return crossing


def sg_per_radian(g: NDArray[np.float64], integration: IntegrationGeometry) -> NDArray[np.float64]:
    """How fast each ``g``'s excitation error changes with tilt, ``|dSg/dtheta|`` per radian.

    For a rotation about ``x`` this is the lever arm ``|(g_y, g_z)|``, the same quantity the Klar
    window used for ``sg_max``; it converts a rocking-curve width in ``Sg`` into a tilt angle.
    """
    _require_rotation(integration)
    return np.linalg.norm(np.asarray(g, dtype=np.float64)[:, 1:], axis=1)


def integrated_mask(
    g: NDArray[np.float64],
    *,
    energy: float,
    u0: float,
    integration: IntegrationGeometry,
    half_width_sg: float = 0.0,
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Keep-mask of the reflections ``g`` ``(N, 3)`` this frame integrates, and the margins.

    A reflection is integrated when it crosses ``Sg = 0`` inside the frame and the crossing sits at
    least ``half_width_sg`` (converted to a tilt angle by its own lever arm) inside both edges, so
    the rocking curve around it falls inside the frame. ``half_width_sg = 0`` is the bare crossing
    test. Also returns each reflection's margin in degrees -- the distance from the crossing to the
    nearer edge, less the half-width; ``-inf`` where it never crosses -- which ranks how centrally
    a frame integrates the reflection.
    """
    _require_rotation(integration)
    semiangle = integration.semiangle
    crossing = bragg_crossing_angles(g, energy=energy, u0=u0, semiangle=semiangle)
    lever = sg_per_radian(g, integration)
    with np.errstate(divide="ignore", invalid="ignore"):
        half_width_deg = np.where(lever > 0.0, np.rad2deg(half_width_sg / lever), np.inf)
    margin = semiangle - np.abs(crossing) - half_width_deg
    margin = np.where(np.isnan(crossing), -np.inf, margin)
    return margin >= 0.0, margin

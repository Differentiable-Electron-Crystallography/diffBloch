"""Observables derived from a propagated wavefunction.

The bridge from ``core.solver.propagate`` (complex exit-wave amplitudes) to the real, comparable
quantities a loss consumes. ``intensities`` is the elastic diffracted intensity ``|psi|^2``
ported from ``diffBloch_private/diffBloch/dynamical.py`` (``torch.abs(psi) ** 2`` at lines
737/802).

Typed product objects (``BlochSolution`` / ``PatternBatch``, retiring ``DiffractionDataset``) land
in the next slice; this slice is the pure observable.
"""

from __future__ import annotations

from torch import Tensor

__all__ = ["intensities"]


def intensities(amplitudes: Tensor) -> Tensor:
    """Elastic diffracted intensity ``|psi|^2`` of complex exit-wave ``amplitudes``.

    Shape-preserving; returns a real tensor (the real dtype matching the complex input).
    Differentiable in ``amplitudes`` (hence back through ``A`` / ``Fgb``).
    """
    return amplitudes.abs().square()

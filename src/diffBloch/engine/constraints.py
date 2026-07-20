"""Hard molecular constraints on the physical state.

A :class:`ConstraintTransform` is a hard *reparameterization* of the bounded
:class:`~diffBloch.params.PhysicalState`, distinct from a soft
:class:`~diffBloch.engine.refine.PenaltyTerm`: a constraint rewrites the state so an invariant holds
exactly at every optimizer step (e.g. deriving hydrogen positions from their parents), rather than
adding a cost the optimizer may trade against. Constraints are applied in the refinement objective
after ``constrain`` and before the diffraction term, so the diffraction *and* the penalties see the
transformed state.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from diffBloch.params import PhysicalState

__all__ = ["ConstraintTransform"]


@runtime_checkable
class ConstraintTransform(Protocol):
    """A hard constraint applied to the physical state during refinement.

    ``name`` identifies the transform for deterministic ordering and duplicate rejection; ``apply``
    returns a *new* :class:`~diffBloch.params.PhysicalState` with the constraint enforced (no
    in-place mutation), so gradients flow through the reparameterization to the free parameters it
    depends on.
    """

    name: str

    def apply(self, state: PhysicalState) -> PhysicalState:
        """Return a new physical state with this constraint enforced."""
        ...

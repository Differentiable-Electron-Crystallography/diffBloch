"""Validated parameter value-types for the preprocess calibration steps.

These are the *parsed* forms of the sweep parameters (parse, don't validate): each frozen dataclass
validates its own invariants in ``__post_init__``, so an invalid spec is unrepresentable and the
pure ``Plan -> Plan`` steps that consume them never re-validate. The pydantic config blocks at the
YAML edge (:mod:`diffBloch.config.schema`) parse into these via ``to_search`` / ``to_grid`` and
delegate their validation here -- one home for each rule, no drift between config and function.

They are plain frozen dataclasses (the codebase's one value-object vocabulary, like
:class:`~diffBloch.params.RefinableParams`), so the algorithm contract stays pydantic-free: pydantic
parses YAML at the edge but never rides into a step. A direct/test caller constructs them the same
way the config does and gets the same construction-time error.

Failures raise ``ValueError`` today (fail-fast; the only callers are config-load and direct
construction). When the ``app`` layer needs to surface validation errors *as values* (to a TUI /
batch runner), a thin boundary ``parse(...) -> Result[Spec, ValidationError]`` adapter will wrap
these raising constructors -- Result stays at that boundary and never enters a step. See
``design/decisions/stage11-convergence.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["HexagonalSearch", "ThicknessGrid"]


@dataclass(frozen=True)
class HexagonalSearch:
    """Validated bounds for the ``fit_orientation`` Palatinus hexagonal search (degrees).

    Defaults are the faithful ``diffBloch_private`` values (``configs/preprocess/base.yaml``).
    ``max_iterations`` has no private precedent (the private search has no cap); its default is an
    uncalibrated runaway guard (see ``KNOWN_ISSUES.md``), to be tuned once real-data convergence is
    known.
    """

    max_search_angle: float = 0.4  # largest tilt radius the search starts from
    min_search_angle: float = 0.001  # radius floor that terminates the search
    n_steps: int = 6  # hexagonal azimuths per ring (6 -> 0, 60, ..., 300 deg)
    max_iterations: int = 200  # runaway guard: max search passes per orientation (uncalibrated)

    def __post_init__(self) -> None:
        if self.min_search_angle <= 0.0 or self.max_search_angle <= 0.0:
            raise ValueError("search angles must be positive")
        if self.max_search_angle <= self.min_search_angle:
            raise ValueError("max_search_angle must exceed min_search_angle")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")


@dataclass(frozen=True)
class ThicknessGrid:
    """Validated grid of candidate thicknesses for ``fit_thickness`` (Angstroms).

    ``fit_thickness`` evaluates ``n_steps`` candidates spaced evenly from ``min_thickness`` to
    ``max_thickness`` (inclusive) and keeps the lowest-wR2 one. Defaults are the faithful
    ``diffBloch_private`` values (``configs/preprocess/base.yaml``: 5 A to 2000 A in 100 steps).
    """

    min_thickness: float = 5.0  # smallest candidate thickness
    max_thickness: float = 2000.0  # largest candidate thickness
    n_steps: int = 100  # number of evenly-spaced candidates (inclusive endpoints)

    def __post_init__(self) -> None:
        if self.min_thickness <= 0.0 or self.max_thickness <= 0.0:
            raise ValueError("thickness bounds must be positive")
        if self.max_thickness <= self.min_thickness:
            raise ValueError("max_thickness must exceed min_thickness")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")

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

__all__ = ["BeamSelection", "ConvergenceTolerance", "HexagonalSearch", "ThicknessGrid"]


@dataclass(frozen=True)
class BeamSelection:
    """Validated cutoffs for the ``select_beams`` Klar et al. (2023) active-set filter.

    The three knobs *jointly* define each orientation's active beam set, so they share one value:
    ``rsg`` (relative excitation-error cutoff -- a reflection is kept when ``|Sg| / sg_max < rsg``)
    and ``integration_semiangle`` (degrees, which scales ``sg_max``) must both be positive, since
    either at zero rejects every reflection. ``dsg`` (the absolute excitation-error margin in the
    ``sg_max - |Sg| > dsg`` test) carries no positivity invariant -- a negative margin legitimately
    loosens the cone -- so it is left unconstrained rather than fabricate a bound.

    Defaults are the faithful ``diffBloch_private`` values (``NumericsConfig``).
    """

    rsg: float = 0.9  # relative excitation-error cutoff: keep when |Sg| / sg_max < rsg
    dsg: float = 0.0015  # absolute excitation-error margin: keep when sg_max - |Sg| > dsg
    integration_semiangle: float = 1.0  # degrees: integration cone half-angle; scales sg_max

    def __post_init__(self) -> None:
        if self.rsg <= 0.0:
            raise ValueError("rsg must be positive")
        if self.integration_semiangle <= 0.0:
            raise ValueError("integration_semiangle must be positive")


@dataclass(frozen=True)
class ConvergenceTolerance:
    """Stopping rule for a convergence sweep: stability threshold + a runaway cap.

    A convergence sweep grows a simulation-accuracy knob and stops once *consecutive* simulations
    stop changing: ``r_factor_threshold`` is the largest consecutive-simulation R-factor still
    counted as "converged" (the private's ``r_factor_threshold = 0.005``). ``patience`` is how many
    *consecutive settled steps* are required before declaring convergence -- guarding the private's
    plateau bug, where one below-threshold step (often a no-op step that left the discrete beam set,
    hence the simulation, unchanged) stopped the sweep prematurely; a single dip no longer suffices.
    ``max_iterations`` is the hard cap on sweep steps before non-convergence is raised (the
    private's ``MAX_SWEEP_ITERATIONS = 100``); it also gives ``iterate_until``'s previously-bare cap
    a home.

    Defaults are the faithful ``diffBloch_private`` values
    (``configs/convergence_test/base.yaml`` + ``convergence_testing.MAX_SWEEP_ITERATIONS``), except
    ``patience`` which has no private precedent (the private stops on the first dip); its default of
    2 is the minimal "not a one-off" and is an uncalibrated target (see ``KNOWN_ISSUES.md``).
    """

    r_factor_threshold: float = 0.005  # converged once consecutive-sim R-factor < this
    patience: int = 2  # consecutive settled steps required before declaring convergence
    max_iterations: int = 100  # hard cap on sweep steps before raising non-convergence

    def __post_init__(self) -> None:
        if self.r_factor_threshold <= 0.0:
            raise ValueError("r_factor_threshold must be positive")
        if self.patience < 1:
            raise ValueError("patience must be >= 1")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")


@dataclass(frozen=True)
class HexagonalSearch:
    """Validated bounds for the ``fit_orientation`` Palatinus hexagonal search (degrees).

    Defaults are the faithful ``diffBloch_private`` values (``configs/preprocess/base.yaml``).
    ``max_iterations`` has no private precedent (the private search has no cap -- it relies on
    monotone wR2 descent plus the radius floor). Its default of ``600`` is **calibrated on the
    quartz anchor**: across its 99 rotations the slowest legitimate search converged in 526 passes,
    so 600 leaves headroom while still catching a genuine runaway. A dataset with shallower minima
    may need a larger cap -- raise it via ``preprocess.orientation.max_iterations`` (see
    ``KNOWN_ISSUES.md``).
    """

    max_search_angle: float = 0.4  # largest tilt radius the search starts from
    min_search_angle: float = 0.001  # radius floor that terminates the search
    n_steps: int = 6  # hexagonal azimuths per ring (6 -> 0, 60, ..., 300 deg)
    max_iterations: int = 600  # runaway guard: max search passes per orientation (quartz max 526)

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

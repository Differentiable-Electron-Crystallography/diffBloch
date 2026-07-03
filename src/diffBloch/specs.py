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
from typing import Literal

__all__ = [
    "BeamSelection",
    "ConvergenceTest",
    "ConvergenceTolerance",
    "HexagonalSearch",
    "Mosaicity",
    "RockingCurve",
    "ThicknessGrid",
]


@dataclass(frozen=True)
class BeamSelection:
    """Validated cutoffs for the ``select_beams`` Klar et al. (2023) active-set filter.

    The knobs *jointly* define each orientation's active beam set. ``rsg`` (relative excitation
    error cutoff -- a reflection is kept when ``|Sg| / sg_max < rsg``) and ``integration_semiangle``
    (degrees, which scales ``sg_max``) must both be positive, since either at zero rejects every
    reflection. ``dsg`` (the absolute excitation-error margin in the ``sg_max - |Sg| > dsg`` test)
    carries no positivity invariant -- a negative margin legitimately loosens the cone -- so it is
    left unconstrained rather than fabricate a bound. ``geometry`` is the data-collection geometry:
    it fixes which reflection component sets ``sg_max`` (the excitation-error span swept during
    integration) -- distance from the goniometer rock axis for ``continuous_rotation``, distance
    from the beam for ``precession`` -- so it must match the integrator's tilt geometry
    (:class:`RockingCurve`). It shares the ``data_collection_geometry`` of ``RockingCurve``.

    Defaults are the faithful ``diffBloch_private`` values (``NumericsConfig``).
    """

    rsg: float = 0.9  # relative excitation-error cutoff: keep when |Sg| / sg_max < rsg
    dsg: float = 0.0015  # absolute excitation-error margin: keep when sg_max - |Sg| > dsg
    integration_semiangle: float = 1.0  # degrees: integration cone half-angle; scales sg_max
    geometry: Literal["continuous_rotation", "precession"] = "continuous_rotation"

    def __post_init__(self) -> None:
        if self.rsg <= 0.0:
            raise ValueError("rsg must be positive")
        if self.integration_semiangle <= 0.0:
            raise ValueError("integration_semiangle must be positive")
        if self.geometry not in ("continuous_rotation", "precession"):
            raise ValueError("geometry must be 'continuous_rotation' or 'precession'")


@dataclass(frozen=True)
class ConvergenceTolerance:
    """Stopping rule for a convergence sweep: stability threshold + a runaway cap.

    A convergence sweep grows a simulation-accuracy knob and stops the first time *consecutive*
    simulations stop changing: ``r_factor_threshold`` is the largest consecutive-simulation R-factor
    still counted as "converged" (the private's ``r_factor_threshold = 0.005``). ``max_iterations``
    is the hard cap on sweep steps before non-convergence is raised (the private's
    ``MAX_SWEEP_ITERATIONS = 100``); it also gives ``iterate_until``'s previously-bare cap a home.

    Defaults are the faithful ``diffBloch_private`` values
    (``configs/convergence_test/base.yaml`` + ``convergence_testing.MAX_SWEEP_ITERATIONS``). The
    stopping rule is the private's exactly -- the first below-threshold step stops the sweep, with
    no patience and no null-step handling (see ``design/decisions/stage11-convergence.md``).
    """

    r_factor_threshold: float = 0.005  # converged once consecutive-sim R-factor < this
    max_iterations: int = 100  # hard cap on sweep steps before raising non-convergence

    def __post_init__(self) -> None:
        if self.r_factor_threshold <= 0.0:
            raise ValueError("r_factor_threshold must be positive")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")


@dataclass(frozen=True)
class ConvergenceTest:
    """Which convergence operation the preprocess driver runs, and how it sweeps the knobs.

    ``operation`` selects the driver's phase(s): ``coverage`` grows the beam pool + window to the
    minimum that maximises matched-reflection coverage (pure geometry, tilt untouched);
    ``self_stability`` grows the pool, window and rocking-curve tilt count until consecutive
    simulations stop changing; ``both`` runs coverage first and seeds self-stability from its
    settled scalars (the private's ``initial_*`` handoff). ``start_g_max_refine`` is the pool
    sweep's start
    radius -- the window and tilt starts come from :class:`BeamSelection.integration_semiangle` and
    :class:`RockingCurve.sampling`, so they are not duplicated here. ``pool_step`` / ``window_step``
    / ``tilt_step`` are the per-knob increments; ``num_passes`` is the fixed self-stability
    coordinate-sweep count (each pass revisits every knob after the others moved). The R-factor
    stopping rule + runaway cap live on :class:`ConvergenceTolerance`, not here (single
    responsibility: this type is *what to sweep*, that one is *when to stop*).

    ``operation`` and ``num_passes`` are faithful to ``diffBloch_private`` ``convergence_test``
    (branch ``pattern-vis-convergence-testing``: ``operation in {initial_minimum_param_sweep,
    hyperparams_optimization, both}`` renamed to the 2.0 phase names; ``num_passes`` the e2e's 2).
    The step magnitudes are 2.0 defaults -- the private branch's config yaml was not captured, so
    they are calibrated for the convergence tutorial rather than ported verbatim (tune per dataset).
    """

    operation: Literal["coverage", "self_stability", "both"] = "both"
    start_g_max_refine: float = 0.5  # pool sweep start radius (window/tilt starts from the specs)
    pool_step: float = 0.1  # g_max_refine increment per sweep step
    window_step: float = 0.2  # integration_semiangle increment per sweep step (degrees)
    tilt_step: float = 2.0  # rocking_curve_sampling increment per sweep step (tilt count)
    num_passes: int = 2  # fixed self-stability coordinate-sweep passes (per-pass order-swap)

    def __post_init__(self) -> None:
        if self.operation not in ("coverage", "self_stability", "both"):
            raise ValueError("operation must be 'coverage', 'self_stability', or 'both'")
        if self.start_g_max_refine <= 0.0:
            raise ValueError("start_g_max_refine must be positive")
        if self.pool_step <= 0.0 or self.window_step <= 0.0 or self.tilt_step <= 0.0:
            raise ValueError("pool_step, window_step and tilt_step must be positive")
        if self.num_passes < 1:
            raise ValueError("num_passes must be >= 1")


@dataclass(frozen=True)
class HexagonalSearch:
    """Validated bounds for the ``fit_orientation`` Palatinus hexagonal search (degrees).

    Defaults are the faithful ``diffBloch_private`` values (``configs/preprocess/base.yaml``).
    ``max_iterations`` has no private precedent (the private search has no cap -- it relies on
    monotone wR2 descent plus the radius floor). Its default of ``2000`` is **calibrated on the
    quartz anchor under the integrated recipe**: every one of its 99 rotations still terminates by
    the radius floor, but the bumpier rocking-curve-integrated landscape needs many more passes than
    the static fit -- the slowest legitimate search took 1288 passes (526 without integration), so
    2000 leaves cross-platform headroom while still catching a genuine runaway. A dataset with
    shallower minima may need a larger cap -- raise it via ``preprocess.orientation.max_iterations``
    (see ``KNOWN_ISSUES.md``).
    """

    max_search_angle: float = 0.4  # largest tilt radius the search starts from
    min_search_angle: float = 0.001  # radius floor that terminates the search
    n_steps: int = 6  # hexagonal azimuths per ring (6 -> 0, 60, ..., 300 deg)
    max_iterations: int = 2000  # runaway guard: max search passes (integrated quartz max 1288)

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
class RockingCurve:
    """Validated geometry for rocking-curve integration (tilts as sub-orientations).

    A rotation-electron-diffraction frame integrates each reflection's intensity as the crystal
    sweeps through the Ewald sphere, so the forward model samples ``sampling`` slightly-tilted
    sub-orientations spanning +/- ``semiangle`` and sums their intensities. ``semiangle`` (degrees)
    is the tilt half-width -- the same physical angular integration range as the Klar beam-selection
    window (``BeamSelection.integration_semiangle``), here setting the tilt span, so the two share
    one value. ``sampling`` is the number of tilts; ``sampling = 1`` is the identity (a single
    static solve), which is how the integration composes off by default. ``geometry`` selects the
    sweep: ``continuous_rotation`` (goniometer x-axis tilts, implemented) or ``precession`` (a cone;
    a deferred discriminated mode -- see the decision doc).

    Faithful to ``diffBloch_private`` (``integration_semiangle`` / ``rocking_curve_sampling`` /
    ``data_collection_geometry``; ``rotation_dataset.generate_integration_rotation_matrices``); see
    ``design/decisions/stage11-rocking-curve.md``.
    """

    semiangle: float = 1.0  # degrees: tilt half-width (shares BeamSelection.integration_semiangle)
    sampling: int = 42  # number of tilts across +/- semiangle; 1 = single static solve (identity)
    geometry: Literal["continuous_rotation", "precession"] = "continuous_rotation"

    def __post_init__(self) -> None:
        if self.semiangle <= 0.0:
            raise ValueError("semiangle must be positive")
        if self.sampling < 1:
            raise ValueError("sampling must be >= 1")
        if self.geometry not in ("continuous_rotation", "precession"):
            raise ValueError("geometry must be 'continuous_rotation' or 'precession'")


@dataclass(frozen=True)
class Mosaicity:
    """Mosaicity broadening of the rocking curve: a moving-average window over the tilt axis.

    Crystal mosaic spread smears each reflection's rocking curve; the private models it as a
    ``window``-wide moving average of the per-tilt intensities before the sum-over-tilts
    integration. ``window`` is the number of consecutive tilts averaged; it must be ``>= 1`` and, at
    reduction time, ``<= sampling`` (the tilt count). ``window = 1`` is the identity (no
    broadening), so composing the ``mosaicity`` step with it is a no-op. This is a modifier on top
    of the rocking-curve integration (:class:`RockingCurve`) -- it only has meaning once the tilt
    set exists, so the ``mosaicity`` step is ordered after ``integrate_rocking_curve``.

    **Divergence from ``diffBloch_private`` (recorded):** the private hardcodes the moving-average
    ``window_size = 5`` and uses its ``mosaicity_num_frames`` config only as an on/off flag (the
    frame count never reaches the window). 2.0 keeps the faithful **default of 5** but exposes
    ``window`` as a real, tunable config parameter -- a principled fix of the private quirk (the
    config field name implied a tunable window the code ignored). See ``DIVERGENCE.md``.
    """

    window: int = 5  # tilts averaged per sliding window; faithful private default (hardcoded there)

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window must be >= 1")


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

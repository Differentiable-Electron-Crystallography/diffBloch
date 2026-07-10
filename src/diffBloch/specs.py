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
these raising constructors -- Result stays at that boundary and never enters a step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "BeamSelection",
    "ConvergenceTest",
    "ConvergenceTolerance",
    "HexagonalSearch",
    "IntegrationGeometry",
    "Mosaicity",
    "RockingCurve",
    "ScoredSelection",
    "ThicknessGrid",
    "TiltIndependent",
    "TiltSegmentUnion",
    "TrialCoupling",
]


@dataclass(frozen=True)
class IntegrationGeometry:
    """The angular integration range a rotation frame sweeps -- one shared physical value.

    As the crystal rocks, each reflection is integrated over an angular range. That range sets BOTH
    the Klar beam-selection window (the excitation-error span ``sg_max``) and the rocking-curve tilt
    span, because it is the *same physical angle*. Modelling it once here -- shared by
    :class:`BeamSelection` and :class:`RockingCurve` rather than each declaring its own -- makes it
    impossible to give the two consumers different values (the drift a duplicated field invites).

    ``semiangle`` (degrees) is the tilt half-width / integration cone half-angle; it must be
    positive (zero rejects every reflection). ``geometry`` is the data-collection sweep -- distance
    from the goniometer rock axis for ``continuous_rotation``, distance from the beam for
    ``precession`` -- which fixes the ``sg_max`` lever arm and the tilt axis, so both consumers must
    agree on it too.

    Faithful to ``diffBloch_private`` (``integration_semiangle`` / ``data_collection_geometry``).
    """

    semiangle: float = 1.0  # degrees: tilt half-width / integration cone half-angle; scales sg_max
    geometry: Literal["continuous_rotation", "precession"] = "continuous_rotation"

    def __post_init__(self) -> None:
        if self.semiangle <= 0.0:
            raise ValueError("semiangle must be positive")
        if self.geometry not in ("continuous_rotation", "precession"):
            raise ValueError("geometry must be 'continuous_rotation' or 'precession'")


@dataclass(frozen=True)
class BeamSelection:
    """Validated cutoffs for the ``select_beams`` Klar et al. (2023) active-set filter.

    The knobs *jointly* define each orientation's active beam set. ``rsg`` (relative excitation
    error cutoff -- a reflection is kept when ``|Sg| / sg_max < rsg``) must be positive, since at
    zero it rejects every reflection. ``dsg`` (the absolute excitation-error margin in the
    ``sg_max - |Sg| > dsg`` test) carries no positivity invariant -- a negative margin legitimately
    loosens the cone -- so it is left unconstrained rather than fabricate a bound. ``integration``
    (an :class:`IntegrationGeometry`) supplies the ``semiangle`` that scales ``sg_max`` and the
    ``geometry`` that fixes its lever arm; it is *shared* with the :class:`RockingCurve` integrator
    (one physical angle, so the two cannot disagree).

    Defaults are the faithful ``diffBloch_private`` values (``NumericsConfig``).
    """

    rsg: float = 0.9  # relative excitation-error cutoff: keep when |Sg| / sg_max < rsg
    dsg: float = 0.0015  # absolute excitation-error margin: keep when sg_max - |Sg| > dsg
    integration: IntegrationGeometry = field(default_factory=IntegrationGeometry)

    def __post_init__(self) -> None:
        if self.rsg <= 0.0:
            raise ValueError("rsg must be positive")


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
    no patience and no null-step handling.
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
    radius -- the window and tilt starts come from :class:`IntegrationGeometry.semiangle` and
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
    shallower minima may need a larger cap -- raise it via
    ``preprocess.orientation.max_iterations``.

    **Recalibrating for another compound.** The cap is a runaway guard, so set it comfortably above
    the slowest *legitimate* search on that compound, measured under the production recipe:

    1. Run ``fit_orientation`` under the **integrated recipe** you will use in production (the
       rocking-curve-integrated landscape is bumpier and needs far more passes than a static fit --
       calibrating on the static fit under-sizes the cap), on **all** rotations, with a generous cap
       (e.g. ``10_000``) and a :class:`~diffBloch.observability.RecordingLogger`.
    2. Read the per-rotation :class:`~diffBloch.observability.OrientationFitted` events. Each
       carries ``n_passes`` (the sweeps that search took) and ``pass_cap`` (the cap in force).
    3. **Confirm every rotation converged by the radius floor, not by the cap** -- i.e. no
       rotation raised, and ``max(n_passes) < pass_cap`` with margin. A rotation that runs to the
       cap is *signal*: either a genuinely (near-)degenerate landscape to investigate, or a cap
       that is still too low -- never silently raise the cap to make it disappear.
    4. Set ``max_iterations = ceil(headroom * max(n_passes))`` with ``headroom`` ~1.5 (quartz:
       ``1288 -> 2000``). Record the calibrating dataset + recipe alongside the value.

    The ``../notebooks/iain`` calibration notebook automates steps 1-4 (runs the search, plots the
    per-rotation ``n_passes`` distribution against ``pass_cap``, flags any cap-hitters, and prints
    the recommended cap).
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
    sub-orientations spanning +/- the integration ``semiangle`` and sums their intensities.
    ``sampling`` is the number of tilts; ``sampling = 1`` is the identity (a single static solve),
    which is how the integration composes off by default. ``integration`` (an
    :class:`IntegrationGeometry`) supplies the tilt half-width ``semiangle`` -- the *same* physical
    angular range as the Klar beam-selection window, shared with :class:`BeamSelection` so the two
    cannot disagree -- and the ``geometry`` that selects the sweep (``continuous_rotation``,
    goniometer x-axis tilts, implemented; or ``precession``, a deferred cone mode).

    Faithful to ``diffBloch_private`` (``integration_semiangle`` / ``rocking_curve_sampling`` /
    ``data_collection_geometry``; ``rotation_dataset.generate_integration_rotation_matrices``).
    """

    sampling: int = 42  # number of tilts across +/- semiangle; 1 = single static solve (identity)
    integration: IntegrationGeometry = field(default_factory=IntegrationGeometry)

    def __post_init__(self) -> None:
        if self.sampling < 1:
            raise ValueError("sampling must be >= 1")


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

    **Divergence from ``diffBloch_private``:** the private hardcodes the moving-average
    ``window_size = 5`` and uses its ``mosaicity_num_frames`` config only as an on/off flag (the
    frame count never reaches the window). 2.0 keeps the faithful **default of 5** but exposes
    ``window`` as a real, tunable config parameter -- a principled fix of the private quirk (the
    config field name implied a tunable window the code ignored).
    """

    window: int = 5  # tilts averaged per sliding window; faithful private default (hardcoded there)

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window must be >= 1")


@dataclass(frozen=True)
class TiltSegmentUnion:
    """Tilt-segment-union beam coupling: per-tilt-chunk beam sets, not one set for the whole curve.

    The ``diffBloch_private`` coupling policy for rocking-curve integration. It partitions the
    ``B`` tilts into ``n_splits`` contiguous, disjoint chunks and gives each chunk its own coupled
    beam set: the **union** of the excited-beam masks at the chunk's two boundary tilts. A beam is
    excited at a tilt when ``|Sg| < sg_max`` *and* ``|g| < g_max - cap_margin`` (a hard
    excitation-error + coupling-radius cutoff, distinct from the Klar relative filter of
    :class:`BeamSelection`). Because a sharp reflection drifts through the Ewald sphere as the
    crystal rocks, the excited set genuinely differs across the curve; one tilt-independent set
    either over-couples (slow) or drops beams a later tilt needs. The per-chunk union is the
    private's compromise. Each reflection's full rocking curve is later reassembled across chunks
    before the mosaicity reduction (the window spans more tilts than one chunk holds).

    ``g_max`` is the coupling radius (the private's ``g_max_sf / 2 = 4.5 / 2``) and ``cap_margin``
    the safety margin subtracted from it (the private's hardcoded ``0.2``), so the effective cap is
    ``g_max - cap_margin = 2.05``. ``sg_max`` is the excitation-error cutoff. The mean-inner-
    potential ``u0`` and beam energy are experiment quantities threaded in at build time, not policy
    knobs. The ``union_adaptive`` recursive-bisection variant is deferred (only the fixed even-split
    ``union_adaptive = False`` path is ported).

    Defaults are the faithful ``diffBloch_private`` values (``config.union_splits = 12``,
    ``self.g_max = 4.5 / 2``, cap margin ``0.2``, ``self.sg_max = 0.01``).
    """

    n_splits: int = 12  # contiguous tilt chunks; each gets its own boundary-union beam set
    g_max: float = (
        2.25  # coupling radius (private g_max_sf / 2); effective cap = g_max - cap_margin
    )
    cap_margin: float = 0.2  # subtracted from g_max for the coupling cap (private hardcoded 0.2)
    sg_max: float = 0.01  # excitation-error cutoff: a beam couples at a tilt when |Sg| < sg_max

    def __post_init__(self) -> None:
        if self.n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if self.g_max <= 0.0 or self.sg_max <= 0.0:
            raise ValueError("g_max and sg_max must be positive")
        if self.g_max - self.cap_margin <= 0.0:
            raise ValueError("coupling cap g_max - cap_margin must be positive")


@dataclass(frozen=True)
class TiltIndependent:
    """The default coupling: one beam set shared across every rocking-curve tilt.

    The 2.0 baseline (and the ``diffBloch_private`` ``union_splits <= 1`` degenerate case): the
    active beam set ``select_beams`` picks for the nominal orientation is reused, unchanged, at
    every
    tilt of the rocking curve. Fieldless because it carries no policy of its own -- the shared set
    is
    already fixed on the plan; it is the identity member of the coupling discriminated union, chosen
    by construction when a run does *not* want the tilt-dependent per-chunk re-selection.
    """


# How a rocking curve couples beams across its tilts: one shared set (:class:`TiltIndependent`) or
# the private's per-tilt-chunk boundary unions (:class:`TiltSegmentUnion`). A discriminated union
# the
# ``couple_beams`` step matches on, not a boolean toggle -- the faithful policy carries its own
# parameters, the default carries none.
CouplingPolicy = TiltIndependent | TiltSegmentUnion


@dataclass(frozen=True)
class ScoredSelection:
    """The SCORED selector: the Klar window intersected with a scoring-resolution cap.

    When a fit re-derives its reflection sets per trial under a coupling policy, the *scored* set is
    not the solve union -- it is the union filtered back down to the reflections actually compared
    against the observed pattern. That is two filters, mirroring ``diffBloch_private``'s
    ``filter_hkls`` (the Klar relative-excitation window) followed by ``resolution_filter`` (a
    radial ``|g|`` cap): ``klar`` supplies the former (:class:`BeamSelection` -- ``rsg`` / ``dsg`` +
    the shared :class:`IntegrationGeometry`), and ``g_max`` the latter. The private's
    ``g_min_refine`` is ``0.0`` for every dataset (verified), so no lower-shell bound is modelled;
    add one when a dataset needs it.

    ``g_max`` is the *scoring*-resolution cap (the private's ``g_max_refine`` in its scored-set
    role), given its own named home here so it is distinct from the seed beam-pool radius -- the two
    are numerically equal today (both ``1.6`` on quartz) but are separate quantities. It must be
    positive.
    """

    klar: BeamSelection = field(default_factory=BeamSelection)
    g_max: float = 1.6  # scoring-resolution cap on |g| (the private's g_max_refine scored role)

    def __post_init__(self) -> None:
        if self.g_max <= 0.0:
            raise ValueError("g_max must be positive")


@dataclass(frozen=True)
class TrialCoupling:
    """Per-trial re-derivation of both reflection sets during an orientation fit.

    The faithful ``diffBloch_private`` orientation objective re-runs the whole forward at every
    trial orientation: it re-couples the SOLVE union (the excitation coupling of ``policy``) *and*
    re-selects the SCORED set (``scored``) from that fresh union, so both sets track the trial
    orientation rather than staying pinned to the seed. Passed to
    :func:`~diffBloch.preprocess.steps.fit_orientation.fit_orientation` (``coupling=...``) to opt a
    fit into that behaviour; its absence (``None``) keeps the tilt-independent fit (one fixed beam
    set across the search).

    Bundling both selectors makes the invalid state -- coupling active but a selector missing --
    unrepresentable, so the fit takes one optional parameter with no cross-parameter guard.
    """

    policy: TiltSegmentUnion  # SOLVE: the per-tilt-segment excitation union
    scored: ScoredSelection  # SCORED: the Klar window + resolution cap, re-selected per trial


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

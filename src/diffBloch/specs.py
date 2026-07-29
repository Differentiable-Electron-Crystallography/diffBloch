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

Failures raise ``ValueError`` (fail-fast): the callers are config-load and direct construction. A
boundary adapter that needs to surface validation errors *as values* rather than as exceptions (for
a TUI or batch runner) can wrap these raising constructors without changing the step contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "BeamSelection",
    "ConvergenceTest",
    "ConvergenceTolerance",
    "FrameSelection",
    "HexagonalSearch",
    "IntegrationGeometry",
    "Mosaicity",
    "OrientationSelection",
    "RockingCurve",
    "ScoredHklSelection",
    "ThicknessGrid",
    "TiltIndependent",
    "SegmentedUnionCoupling",
    "TrialCoupling",
    "assert_grid_covers_coupling",
]


@dataclass(frozen=True)
class OrientationSelection:
    """Original PETS rotation indices excluded from every downstream experiment stage.

    Indices are zero-based in the source ``.cif_pets`` rotation order. Filtering happens before
    the train/validation split, beam construction, orientation/thickness fitting, inference, and
    structure refinement. Duplicate or negative indices are rejected so the recorded experiment
    selection has one unambiguous identity; the experiment boundary checks the data-dependent upper
    bound once the number of PETS rotations is known.
    """

    ignore_orientations: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if any(index < 0 for index in self.ignore_orientations):
            raise ValueError("ignore_orientations indices must be non-negative")
        if len(set(self.ignore_orientations)) != len(self.ignore_orientations):
            raise ValueError("ignore_orientations must not contain duplicate indices")


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

    The default cutoffs are the values used by the default preprocess path.
    """

    rsg: float = 0.9  # relative excitation-error cutoff: keep when |Sg| / sg_max < rsg
    dsg: float = 0.0015  # absolute excitation-error margin: keep when sg_max - |Sg| > dsg
    integration: IntegrationGeometry = field(default_factory=IntegrationGeometry)

    def __post_init__(self) -> None:
        if self.rsg <= 0.0:
            raise ValueError("rsg must be positive")


@dataclass(frozen=True)
class FrameSelection:
    """Validated criterion for the ``select_frames`` per-rotation (whole-frame) drop.

    The sibling of :class:`BeamSelection`: where that prunes *reflections within* a frame,
    ``select_frames`` drops *whole frames* whose observed pattern is too sparse to inform the fit,
    for the beam-damaged tail of a rotation scan. ``min_observed`` is the fewest *strong* observed
    reflections (``intensity > 3 * sigma``, strict) a frame must carry to be kept; frames below it
    are dropped. The count is **model-independent** -- it reads the observed pattern only, never the
    calculated fit -- so it cannot circularly keep the frames the current model already explains.

    ``min_observed == 0`` keeps every frame (the disabled / no-op default -- opting in requires a
    positive threshold); a negative count is meaningless and rejected. The drop is derived from a
    data-quality floor rather than a hard-coded list of frame indices.
    """

    min_observed: int = 0  # keep a frame iff its strong-reflection count (I > 3 sigma) >= this

    def __post_init__(self) -> None:
        if self.min_observed < 0:
            raise ValueError("min_observed must be >= 0")


@dataclass(frozen=True)
class ConvergenceTolerance:
    """Stopping rule for a convergence sweep: stability threshold + a runaway cap.

    A convergence sweep grows a simulation-accuracy knob and stops the first time *consecutive*
    simulations stop changing: ``r_factor_threshold`` is the largest consecutive-simulation R-factor
    still counted as "converged". ``max_iterations`` is the hard cap on sweep steps before
    non-convergence is raised; it also gives ``iterate_until``'s otherwise-bare cap a home.

    The stopping rule is deliberately simple: the first below-threshold step stops the sweep -- no
    patience window and no null-step handling.
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
    """Step sizes for simulation-convergence testing.

    The test increases ``g_max``, ``sg_max``, and the number of rocking-curve tilt steps. Each
    control is advanced until consecutive simulations differ by less than the requested tolerance.
    Multiple passes revisit the controls after the others have changed.
    """

    g_max_step: float = 0.1
    sg_max_step: float = 0.005
    tilt_steps_step: int = 2
    num_passes: int = 2

    def __post_init__(self) -> None:
        if self.g_max_step <= 0.0 or self.sg_max_step <= 0.0 or self.tilt_steps_step < 1:
            raise ValueError("g_max_step, sg_max_step, and tilt_steps_step must be positive")
        if self.num_passes < 1:
            raise ValueError("num_passes must be >= 1")


@dataclass(frozen=True)
class HexagonalSearch:
    """Validated bounds for the ``fit_orientation`` hexagonal search (degrees, Palatinus et al. 2013).

    ``max_iterations`` is a runaway guard on an otherwise uncapped search: the search terminates on
    its own by monotone wR2 descent plus the radius floor, and the cap only catches a genuine
    non-terminating case. Its default of ``2000`` is **calibrated on the quartz anchor under the
    integrated recipe**: every one of its 99 rotations still terminates by
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

    Steps 1-4 are mechanical and can be automated: run the search, plot the per-rotation
    ``n_passes`` distribution against ``pass_cap``, flag any cap-hitters, and read off the
    recommended cap.
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
    """

    sampling: int = 42  # number of tilts across +/- semiangle; 1 = single static solve (identity)
    integration: IntegrationGeometry = field(default_factory=IntegrationGeometry)

    def __post_init__(self) -> None:
        if self.sampling < 1:
            raise ValueError("sampling must be >= 1")


@dataclass(frozen=True)
class Mosaicity:
    """Mosaicity broadening of the rocking curve: a moving-average window over the tilt axis.

    Crystal mosaic spread smears each reflection's rocking curve; diffBloch models it as a
    ``window``-wide moving average of the per-tilt intensities before the sum-over-tilts
    integration. ``window`` is the number of consecutive tilts averaged; it must be ``>= 1`` and, at
    reduction time, ``<= sampling`` (the tilt count). ``window = 1`` is the identity (no
    broadening), so composing the ``mosaicity`` step with it is a no-op. This is a modifier on top
    of the rocking-curve integration (:class:`RockingCurve`) -- it only has meaning once the tilt
    set exists, so the ``mosaicity`` step is ordered after ``integrate_rocking_curve``.

    ``window`` is a tunable config parameter; its default of 5 is the standard moving-average width.
    """

    window: int = 5  # tilts averaged per sliding window (standard moving-average width)

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window must be >= 1")


@dataclass(frozen=True)
class SegmentedUnionCoupling:
    """Tilt-segment-union beam coupling: per-tilt-chunk beam sets, not one set for the whole curve.

    The coupling policy for rocking-curve integration: it partitions the
    ``B`` tilts into ``fixed_n_segments`` contiguous, disjoint chunks and gives each chunk its own
    coupled beam set: the **union** of the excited-beam masks at the chunk's two boundary tilts.
    A beam is excited at a tilt when ``|Sg| < sg_max`` *and* ``|g| < g_max`` (a hard
    excitation-error + coupling-radius cutoff, distinct from the Klar relative filter of
    :class:`BeamSelection`).
    Because a sharp reflection drifts through the Ewald sphere as the crystal rocks, the excited set
    genuinely differs across the curve; one tilt-independent set either over-couples (slow) or drops
    beams a later tilt needs. The per-chunk union is the compromise this policy strikes. Each
    reflection's full rocking curve is later reassembled across chunks before the mosaicity
    reduction (the window spans more tilts than one chunk holds).

    ``g_max`` is the coupling radius: a beam couples when ``|g| < g_max``. The cutoff is the
    physical solve radius, with no additional margin.
    ``sg_max`` is the excitation-error cutoff. The mean-inner-potential ``u0`` and beam energy are
    experiment quantities threaded in at build time, not policy knobs.

    ``union_adaptive`` chooses how the chunk boundaries are placed. ``False`` uses
    ``fixed_n_segments`` fixed even-sized chunks. ``True`` places boundaries by recursive bisection:
    a tilt range is split further only while its midpoint adds more than ``union_max_new_beams_pct``
    of the boundary union's beams (else the range is frozen as one chunk), so segments are dense
    where the excited set drifts and sparse where it is stable. In the adaptive mode
    ``fixed_n_segments`` is ignored.

    The defaults suit the standard rocking-curve recipe (12 fixed even-sized chunks, fixed mode).
    """

    fixed_n_segments: int = (
        12  # contiguous tilt chunks (fixed mode); each gets a boundary-union set
    )
    g_max: float = 2.25  # coupling radius: a beam couples at a tilt when |g| < g_max
    sg_max: float = 0.01  # excitation-error cutoff: a beam couples at a tilt when |Sg| < sg_max
    union_adaptive: bool = True  # place chunk boundaries by recursive bisection, not even splits
    union_max_new_beams_pct: float = 0.01  # adaptive: split while a midpoint adds > this fraction

    def __post_init__(self) -> None:
        if self.fixed_n_segments < 1:
            raise ValueError("fixed_n_segments must be >= 1")
        if self.g_max <= 0.0 or self.sg_max <= 0.0:
            raise ValueError("g_max and sg_max must be positive")
        if not 0.0 < self.union_max_new_beams_pct <= 1.0:
            raise ValueError("union_max_new_beams_pct must be in (0, 1]")


def assert_grid_covers_coupling(policy: SegmentedUnionCoupling, grid_g_max: float) -> None:
    """Guarantee the ``|g| <= grid_g_max`` grid sphere spans every coupled beam difference (O(1)).

    A coupled solve union admits only beams with ``|g| < g_max``, so any pairwise difference is
    ``|g_j - g_i| < 2 * g_max`` (triangle inequality). When ``2 * g_max <= grid_g_max`` the dense
    integer ``structure_factor_hkl`` sphere therefore contains every difference, so the per-segment
    gathers
    cannot address a reflection outside it -- exactly the condition that makes
    :func:`~diffBloch.core.dynamical.build_structure_factor_gather` ``validate=False`` sound on the
    coupled fit path (it closes the silent-zero coverage gap the O(N^2) integrity checks otherwise
    catch). The radius is orientation-independent, so this one scalar comparison covers every trial
    of every rotation -- checked at fit setup, failing loudly before any solve rather than silently
    gathering zeros deep in the search. The default recipe derives the grid as ``2 * g_max``, so it
    only bites a programmatic caller that hand-builds a grid smaller than its coupling radius needs.
    """
    if 2.0 * policy.g_max > grid_g_max:
        raise ValueError(
            f"coupling radius g_max {policy.g_max:.4g} needs a structure-factor grid "
            f"g_max >= {2.0 * policy.g_max:.4g} to span the beam-difference support, but the grid "
            f"g_max is {grid_g_max}. Widen the grid or shrink the coupling radius; else a coupled "
            "gather silently gathers zeros under validate=False."
        )


@dataclass(frozen=True)
class TiltIndependent:
    """The default coupling: one beam set shared across every rocking-curve tilt.

    The baseline: the active beam set ``select_beams`` picks for the nominal orientation is reused,
    unchanged, at every tilt of the rocking curve. Fieldless because it carries no policy of its own
    -- the shared set is already fixed on the plan; it is the identity member of the coupling
    discriminated union, chosen by construction when a run does *not* want the tilt-dependent
    per-chunk re-selection.
    """


# How a rocking curve couples beams across its tilts: one shared set (:class:`TiltIndependent`) or
# per-tilt-chunk boundary unions (:class:`SegmentedUnionCoupling`). A discriminated union the
# ``couple_beams`` step matches on, not a boolean toggle -- the tilt-dependent policy carries its
# own parameters, the default carries none.
CouplingPolicy = TiltIndependent | SegmentedUnionCoupling


@dataclass(frozen=True)
class ScoredHklSelection:
    """The SCORED selector: the Klar window intersected with a scoring-resolution cap.

    When a fit re-derives its reflection sets per trial under a coupling policy, the *scored* set is
    not the solve union -- it is the union filtered back down to the reflections actually compared
    against the observed pattern. That is two filters: the Klar relative-excitation window followed
    by a radial ``|g|`` cap. ``klar`` supplies the former (:class:`BeamSelection` -- ``rsg`` /
    ``dsg`` + the shared :class:`IntegrationGeometry`), and ``g_max`` the latter. No lower-shell
    bound is modelled; add one when a dataset needs it.

    ``g_max`` is the *scoring*-resolution cap, given its own named home here so it is distinct from
    the seed beam-pool radius -- the two may be numerically equal but are separate quantities. It
    must be positive.
    """

    klar: BeamSelection = field(default_factory=BeamSelection)
    g_max: float = 1.6  # scoring-resolution cap on |g|

    def __post_init__(self) -> None:
        if self.g_max <= 0.0:
            raise ValueError("g_max must be positive")


@dataclass(frozen=True)
class TrialCoupling:
    """Per-trial re-derivation of both reflection sets during an orientation fit.

    The orientation objective under this coupling re-runs the whole forward at every
    trial orientation: it re-couples the SOLVE union (the excitation coupling of ``policy``) *and*
    re-selects the SCORED set (``scored``) from that fresh union, so both sets track the trial
    orientation rather than staying pinned to the seed. Passed to
    :func:`~diffBloch.preprocess.steps.fit_orientation.fit_orientation` (``coupling=...``) to opt a
    fit into that behaviour; its absence (``None``) keeps the tilt-independent fit (one fixed beam
    set across the search).

    Bundling both selectors makes the invalid state -- coupling active but a selector missing --
    unrepresentable, so the fit takes one optional parameter with no cross-parameter guard.
    """

    policy: SegmentedUnionCoupling  # SOLVE: the per-tilt-segment excitation union
    scored: ScoredHklSelection  # SCORED: the Klar window + resolution cap, re-selected per trial


@dataclass(frozen=True)
class ThicknessGrid:
    """Validated grid of candidate thicknesses for ``fit_thickness`` (Angstroms).

    ``fit_thickness`` evaluates ``n_steps`` candidates spaced evenly from ``min_thickness`` to
    ``max_thickness`` (inclusive) and keeps the lowest-wR2 one. The defaults span 5 A to 2000 A in
    100 steps.
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

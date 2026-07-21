"""Pydantic configuration schema for a diffBloch experiment.

Config is validated at the boundary: no Hydra, and no ``DictConfig`` reaches the core. Every field
carries a sensible default ("defaults as code"), so an ``experiment.yaml`` only needs to specify
input references and overrides.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from diffBloch.core.solver import Method
from diffBloch.engine.losses import scaled_w_rbragg_loss, weighted_mse_loss
from diffBloch.engine.refine import AtomSelection, TrainableSpec

if TYPE_CHECKING:
    from diffBloch.engine.forward import LossFn
from diffBloch.specs import (
    BeamSelection,
    HexagonalSearch,
    IntegrationGeometry,
    Mosaicity,
    RockingCurve,
    ThicknessGrid,
    TiltSegmentUnion,
)

# The preprocess config classes below are 1:1 YAML edges over their value-types; their field
# defaults derive from these default instances so the boundary value cannot drift from the
# value-type it parses into. The value-type in ``specs`` is the single source of truth for both the
# default value and its validation rules (e.g. the quartz-calibrated ``max_iterations`` lives once,
# in ``HexagonalSearch``).
_HEXAGONAL_SEARCH_DEFAULTS = HexagonalSearch()
_THICKNESS_GRID_DEFAULTS = ThicknessGrid()


class _StrictConfig(BaseModel):
    """Base for every config model: reject unknown YAML keys at the boundary (the allowlist guard).

    ``extra="forbid"`` turns an unrecognised key into a load-time ``ValidationError`` instead of
    pydantic's default silent drop. Without it, a stale or misspelled key (or a field removed from
    the schema but left in a YAML) is ignored unnoticed -- the exact hole that let the dead
    ``g_max_sf`` / ``sg_max`` keys linger in the fixtures. It is the Ecto-``cast`` / NimbleOptions
    allowlist: config carries only what a consumer reads, enforced at parse time. It does not catch
    a *declared* field with no reader (whole-program analysis would); pairs with keeping each config
    block close to the value-type its fields feed.
    """

    model_config = ConfigDict(extra="forbid")


class SolverConfig(_StrictConfig):
    """Which dynamical solver to use for each phase.

    Both fields are typed as the solver's own :data:`~diffBloch.core.solver.Method` literal (the
    single source of truth), so an unknown method fails fast at config load rather than deep in the
    forward model.
    """

    refine: Method = "matrix_exp"  # gradient-safe default for the refinement (backprop) path
    inference: Method = "bloch_eigen"  # pinned to match existing e2e references


class NumericsConfig(_StrictConfig):
    """Stage-3 numerical-accuracy controls, frozen into the simulation spec.

    ``g_max`` (structure-factor grid radius) and ``g_max_refine`` (seed beam-pool radius) are grid
    primitives consumed directly. ``rsg`` / ``dsg`` are the Klar beam-selection cutoffs and
    ``rocking_curve_sampling`` the tilt count -- the parts of :class:`BeamSelection` /
    :class:`RockingCurve` those value-types do *not* share. ``integration`` is the shared
    :class:`IntegrationGeometry` (one physical angle + geometry feeding both), carried once as its
    own value-type so it cannot be given two values; ``mosaicity`` is the :class:`Mosaicity`
    reduction. The last two are the value-types themselves (identity, not a projected copy), so
    pydantic validates them and forbids unknown keys inside them too.
    """

    g_max: float = 4.5
    g_max_refine: float = 1.6
    rsg: float = 0.9
    dsg: float = 0.0015
    rocking_curve_sampling: int = 42
    integration: IntegrationGeometry = Field(default_factory=IntegrationGeometry)
    mosaicity: Mosaicity = Field(default_factory=Mosaicity)

    def to_beam_selection(self) -> BeamSelection:
        """Assemble the ``select_beams`` value-type: the Klar cutoffs + the shared integration."""
        return BeamSelection(rsg=self.rsg, dsg=self.dsg, integration=self.integration)

    def to_rocking_curve(self) -> RockingCurve:
        """Assemble the ``integrate_rocking_curve`` value-type: tilt count + the shared integration.

        The tilt span and geometry come from the *same* :class:`IntegrationGeometry` as
        ``to_beam_selection``, so the beam window and the tilt sweep cannot disagree.
        """
        return RockingCurve(sampling=self.rocking_curve_sampling, integration=self.integration)

    @model_validator(mode="after")
    def _parse_fails_fast(self) -> NumericsConfig:
        self.to_beam_selection()  # the rules live in BeamSelection; fail fast at config load
        self.to_rocking_curve()  # the rules live in RockingCurve; fail fast at config load
        return self


class SampleConfig(_StrictConfig):
    """Fixed sample properties.

    Thickness is captured here because it is a sample/nuisance parameter, not a numerical-accuracy
    knob. A later refinement stage can make it refinable without splitting its config home.
    """

    thicknesses: tuple[float, ...] = (820.0,)

    @field_validator("thicknesses")
    @classmethod
    def _positive_thicknesses(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value:
            raise ValueError("thicknesses must contain at least one value")
        if any(thickness <= 0.0 for thickness in value):
            raise ValueError("thicknesses must be positive")
        return value


class DataSplitConfig(_StrictConfig):
    """Required train/validation split declaration.

    The concrete selector language is intentionally small for Stage 1: it records the fixed split
    policy that later dataset code will materialize into ``data_used``.
    """

    train: str = "all_except_validation"
    validation: str = "every_10th_rotation"


class ObjectiveConfig(_StrictConfig):
    """The differentiable data loss for the default refinement path.

    ``data_term`` parses (via :meth:`to_loss`) into the
    :data:`~diffBloch.engine.forward.LossFn` ``build_engine`` consumes. Only implemented terms are
    admissible (a Poisson NLL and a Gauss-Newton least-squares backend are deferred). Only knobs the
    default path actually consumes live here -- outlier rejection, penalty/nuisance weighting, and
    gradient-norm reporting are deferred until wired, not accepted-but-ignored (cf. penalties, which
    are Python/API composition, not config).
    """

    data_term: Literal["weighted_r", "least_squares"] = "weighted_r"

    def to_loss(self) -> LossFn:
        """Parse the data term into the ``LossFn`` the engine scores with."""
        return {
            "weighted_r": scaled_w_rbragg_loss,
            "least_squares": weighted_mse_loss,
        }[self.data_term]


class OptimizerConfig(_StrictConfig):
    """Explicit optimizer backend for a refinement stage (matches ``OptimizerName``)."""

    name: Literal["lbfgs", "adam", "adamw"] = "lbfgs"
    lr: float = 1e-3


def _atom_selection(mode: Literal["all", "none"]) -> AtomSelection:
    return AtomSelection.all() if mode == "all" else AtomSelection.none()


class TrainableConfig(_StrictConfig):
    """Whole-group trainable selections for a refinement stage.

    A 1:1 edge over :class:`~diffBloch.engine.refine.TrainableSpec`: each group is ``all`` or
    ``none`` and parses (via :meth:`to_spec`) into an ``AtomSelection``. Element-filtered selections
    (e.g. freeze H) are not config: they are Python/API composition (see
    :func:`~diffBloch.engine.with_hydrogen_riding`).
    """

    positions: Literal["all", "none"] = "all"
    adp: Literal["all", "none"] = "all"
    occupancy: Literal["all", "none"] = "none"
    fgb: Literal["all", "none"] = "none"

    def to_spec(self) -> TrainableSpec:
        """Parse into the ``TrainableSpec`` the refinement optimizer consumes."""
        return TrainableSpec(
            positions=_atom_selection(self.positions),
            adp=_atom_selection(self.adp),
            occupancy=_atom_selection(self.occupancy),
            fgb=_atom_selection(self.fgb),
        )


class RefinementConfig(_StrictConfig):
    """Stable execution knobs for the *default* single-stage app refinement (``run refine``).

    These tune the default path; they do not author a scientific program. Scientific composition
    (hard constraints such as hydrogen riding, soft penalties, freeze-H masks, multi-stage
    workflows) is expressed as typed Python/API values -- see
    :func:`~diffBloch.engine.build_refinement_model`,
    :func:`~diffBloch.engine.build_refinement_problem`, and
    :func:`~diffBloch.engine.with_hydrogen_riding` -- and is promoted to config only once the
    default recipe commits to it as stable public behaviour.
    """

    steps: int = 500
    trainable: TrainableConfig = Field(default_factory=TrainableConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)
    split: DataSplitConfig = Field(default_factory=DataSplitConfig)


class OrientationFitConfig(_StrictConfig):
    """Bounds for the ``fit_orientation`` Palatinus hexagonal search (preprocess).

    The YAML edge: parses (via :meth:`to_search`) into the validated
    :class:`~diffBloch.specs.HexagonalSearch` value-type the pure ``fit_orientation`` consumes, and
    delegates all validation there (one rule home, no drift). Defaults derive from that value-type
    (``_HEXAGONAL_SEARCH_DEFAULTS``), so the boundary value cannot drift from it either.
    """

    max_search_angle: float = _HEXAGONAL_SEARCH_DEFAULTS.max_search_angle  # degrees
    min_search_angle: float = _HEXAGONAL_SEARCH_DEFAULTS.min_search_angle  # degrees
    n_steps: int = _HEXAGONAL_SEARCH_DEFAULTS.n_steps  # hexagonal azimuths per ring
    max_iterations: int = _HEXAGONAL_SEARCH_DEFAULTS.max_iterations  # runaway guard

    def to_search(self) -> HexagonalSearch:
        """Parse into the validated value-type the pure ``fit_orientation`` consumes."""
        return HexagonalSearch(
            max_search_angle=self.max_search_angle,
            min_search_angle=self.min_search_angle,
            n_steps=self.n_steps,
            max_iterations=self.max_iterations,
        )

    @model_validator(mode="after")
    def _parse_fails_fast(self) -> OrientationFitConfig:
        self.to_search()  # the rules live in HexagonalSearch; fail fast at config load
        return self


class ThicknessFitConfig(_StrictConfig):
    """Bounds for the ``fit_thickness`` per-rotation grid search (preprocess).

    The YAML edge: parses (via :meth:`to_grid`) into the validated
    :class:`~diffBloch.specs.ThicknessGrid` value-type the pure ``fit_thickness`` consumes, and
    delegates all validation there (one rule home, no drift). Defaults derive from that value-type
    (``_THICKNESS_GRID_DEFAULTS``), so the boundary value cannot drift from it either.
    """

    min_thickness: float = _THICKNESS_GRID_DEFAULTS.min_thickness  # Angstroms
    max_thickness: float = _THICKNESS_GRID_DEFAULTS.max_thickness  # Angstroms
    n_steps: int = _THICKNESS_GRID_DEFAULTS.n_steps  # evenly-spaced candidates

    def to_grid(self) -> ThicknessGrid:
        """Parse into the validated value-type the pure ``fit_thickness`` consumes."""
        return ThicknessGrid(
            min_thickness=self.min_thickness,
            max_thickness=self.max_thickness,
            n_steps=self.n_steps,
        )

    @model_validator(mode="after")
    def _parse_fails_fast(self) -> ThicknessFitConfig:
        self.to_grid()  # the rules live in ThicknessGrid; fail fast at config load
        return self


class CouplingConfig(_StrictConfig):
    """Per-trial beam coupling policy for ``fit_orientation`` (preprocess).

    The YAML edge: parses (via :meth:`to_policy`) into the validated
    :class:`~diffBloch.specs.TiltSegmentUnion` value-type the coupled fit consumes, and delegates
    all validation there (one rule home, no drift).

    Unlike the numerical preprocess blocks, coupling carries **no defaults**: it determines the
    physics (the per-trial SOLVE union) and is experiment-specific, so a silent faithful-default
    would let a forgotten policy pass as a deliberate one. All four fields are required when the
    block is present, and the block itself is optional only for experiments that never run the
    coupled fit (see :class:`PreprocessConfig`); composing the fit without it raises. The value-type
    keeps its own defaults for programmatic pipeline authors -- only the config edge is explicit.
    """

    n_splits: int  # contiguous tilt chunks
    g_max: float  # coupling radius (1/Angstrom)
    cap_margin: float  # subtracted from g_max for the coupling cap
    sg_max: float  # excitation-error cutoff

    def to_policy(self) -> TiltSegmentUnion:
        """Parse into the validated value-type the coupled ``fit_orientation`` consumes."""
        return TiltSegmentUnion(
            n_splits=self.n_splits,
            g_max=self.g_max,
            cap_margin=self.cap_margin,
            sg_max=self.sg_max,
        )

    @model_validator(mode="after")
    def _parse_fails_fast(self) -> CouplingConfig:
        self.to_policy()  # the rules live in TiltSegmentUnion; fail fast at config load
        return self


class PreprocessConfig(_StrictConfig):
    """Preprocess-stage configuration (the ``Plan -> Plan`` calibration pipeline).

    Grouping, not composition: each block configures one preprocess step. Only steps the default run
    composes get a config block here: ``fit_orientation`` (its search bounds under ``orientation``
    and its per-trial ``coupling`` policy) and ``fit_thickness``. The optional ``converge_numerics``
    driver is *not* in the default recipe, so it has no config block -- a caller that composes it
    constructs :class:`~diffBloch.specs.ConvergenceTest` /
    :class:`~diffBloch.specs.ConvergenceTolerance` at the composition site (their defaults are the
    faithful values). Opt-in step config lives with the step, not in an always-present block.

    ``coupling`` is ``None`` unless declared: it has no faithful default (see
    :class:`CouplingConfig`), so an experiment that runs the coupled orientation fit must declare it
    or the recipe build raises. An experiment that never runs the fit may leave it unset.
    """

    orientation: OrientationFitConfig = Field(default_factory=OrientationFitConfig)
    coupling: CouplingConfig | None = None
    thickness: ThicknessFitConfig = Field(default_factory=ThicknessFitConfig)


class Inputs(_StrictConfig):
    """Input references — relative to the experiment directory only (no project-root paths)."""

    structure: str
    observations: str
    load_hydrogens: bool = False  # include hydrogen atom sites (molecular crystals; off by default)

    @field_validator("structure", "observations")
    @classmethod
    def _relative_path_only(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                "input references must be relative paths within the experiment directory"
            )
        return value


class ExperimentConfig(_StrictConfig):
    """A whole experiment, validated at load. No Hydra, no ``DictConfig``."""

    name: str
    inputs: Inputs
    sample: SampleConfig = Field(default_factory=SampleConfig)
    numerics: NumericsConfig = Field(default_factory=NumericsConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    """Parse and validate one ``experiment.yaml``.

    Fails fast with a ``pydantic.ValidationError`` at the boundary, rather than a deferred runtime
    surprise deep in the pipeline.
    """
    data = yaml.safe_load(Path(path).read_text())
    return ExperimentConfig.model_validate(data)

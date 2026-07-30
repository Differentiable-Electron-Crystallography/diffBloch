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

from diffBloch.core.solver import FloatFormat, SolverMethod
from diffBloch.engine.losses import weighted_mse_loss, wr2_loss
from diffBloch.engine.refine import AtomSelection, TrainableSpec

if TYPE_CHECKING:
    from diffBloch.engine.forward import LossFn
from diffBloch.specs import (
    Absorption,
    ApparentThicknessNetwork,
    BeamSelection,
    HexagonalSearch,
    IntegrationGeometry,
    Mosaicity,
    OrientationSelection,
    PerTiltCoupling,
    RockingCurve,
    ThicknessGrid,
    UnionCoupling,
)

# The preprocess config classes below are 1:1 YAML edges over their value-types; their field
# defaults derive from these default instances so the boundary value cannot drift from the
# value-type it parses into. The value-type in ``specs`` is the single source of truth for both the
# default value and its validation rules (e.g. the calibrated ``max_iterations`` lives once,
# in ``HexagonalSearch``).
_HEXAGONAL_SEARCH_DEFAULTS = HexagonalSearch()
_THICKNESS_GRID_DEFAULTS = ThicknessGrid()
_THICKNESS_NN_DEFAULTS = ApparentThicknessNetwork()


class _StrictConfig(BaseModel):
    """Base for every config model: reject unknown YAML keys at the boundary (the allowlist guard).

    ``extra="forbid"`` turns an unrecognised key into a load-time ``ValidationError`` instead of
    pydantic's default silent drop. Without it, a stale or misspelled key (or a field removed from
    the schema but left in a YAML) is ignored unnoticed. It is an allowlist: config carries only
    what a consumer reads, enforced at parse time. It does not catch
    a *declared* field with no reader (whole-program analysis would); pairs with keeping each config
    block close to the value-type its fields feed.
    """

    model_config = ConfigDict(extra="forbid")


class SolverConfig(_StrictConfig):
    """Which dynamical solver to use for each phase.

    Both fields are typed as the solver's own :data:`~diffBloch.core.solver.SolverMethod` literal (the
    single source of truth), so an unknown method fails fast at config load rather than deep in the
    forward model.
    """

    refine: SolverMethod = "matrix_exp"  # gradient-safe default for the refinement (backprop) path
    inference: SolverMethod = "matrix_exp"


class BlochwaveConfig(_StrictConfig):
    """Numerical-accuracy controls, frozen into the simulation spec.

    ``g_max_refine`` (seed beam-pool / scoring-resolution radius) is a grid primitive consumed
    directly. The structure-factor support grid is *not* a config field: it is derived as ``2x`` the
    solve cutoff (``g_max``), because a beam set bounded by ``|g| <= cutoff`` produces ``F(g - h)`` terms
    reaching ``2 * cutoff`` -- so declaring both cutoff and support would let them contradict
    (one beam cutoff, support derived). ``rsg`` / ``dsg`` are
    the Klar beam-selection cutoffs and ``rocking_curve_sampling`` the tilt count. The shared
    integration semi-angle is read from the PETS observations rather than configured.
    ``mosaicity`` is the :class:`Mosaicity` reduction.
    """

    solver: SolverConfig = Field(default_factory=SolverConfig)
    absorption: bool = False
    g_max_refine: float = 1.6
    rsg: float = 0.9
    dsg: float = 0.0015
    rocking_curve_sampling: int = 42
    mosaicity: Mosaicity = Field(default_factory=Mosaicity)
    fixed_n_segments: int = 12
    coupling_mode: Literal["union", "per_tilt"] = "union"
    g_max: float = 2.25
    sg_max: float = 0.01
    union_adaptive: bool = True
    union_max_new_beams_pct: float = 0.01
    ignore_orientations: tuple[int, ...] = ()

    def to_absorption(self) -> Absorption:
        """Parse the absorption switch into its typed scientific value."""
        return Absorption(enabled=self.absorption)

    def to_beam_selection(self, integration: IntegrationGeometry) -> BeamSelection:
        """Assemble the ``select_beams`` value-type: the Klar cutoffs + the shared integration."""
        return BeamSelection(rsg=self.rsg, dsg=self.dsg, integration=integration)

    def to_rocking_curve(self, integration: IntegrationGeometry) -> RockingCurve:
        """Assemble the ``integrate_rocking_curve`` value-type: tilt count + the shared integration.

        The caller passes the PETS-derived value to both this method and
        :meth:`to_beam_selection`, so the beam window and tilt sweep cannot disagree.
        """
        return RockingCurve(sampling=self.rocking_curve_sampling, integration=integration)

    def to_policy(self) -> UnionCoupling | PerTiltCoupling:
        """Assemble the selected tilt-dependent Bloch-wave beam-coupling policy."""
        if self.coupling_mode == "per_tilt":
            return PerTiltCoupling(g_max=self.g_max, sg_max=self.sg_max)
        return UnionCoupling(
            fixed_n_segments=self.fixed_n_segments,
            g_max=self.g_max,
            sg_max=self.sg_max,
            union_adaptive=self.union_adaptive,
            union_max_new_beams_pct=self.union_max_new_beams_pct,
        )

    def to_orientation_selection(self) -> OrientationSelection:
        """Parse zero-based source PETS indices excluded from the whole Bloch experiment."""
        return OrientationSelection(ignore_orientations=self.ignore_orientations)

    @model_validator(mode="after")
    def _parse_fails_fast(self) -> BlochwaveConfig:
        if self.rsg <= 0.0:
            raise ValueError("rsg must be positive")
        if self.rocking_curve_sampling < 1:
            raise ValueError("rocking_curve_sampling must be >= 1")
        self.to_policy()
        self.to_orientation_selection()
        self.to_absorption()
        if self.absorption and (
            self.solver.refine == "bloch_eigen" or self.solver.inference == "bloch_eigen"
        ):
            raise ValueError("absorption requires the non-Hermitian-safe 'matrix_exp' solver")
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

    The concrete selector language is intentionally small: it records the fixed split
    policy that dataset code materializes into ``data_used``.
    """

    train: str = "all_except_validation"
    validation: str = "every_10th_rotation"


class ObjectiveConfig(_StrictConfig):
    """The differentiable data loss for the default refinement path.

    ``data_term`` parses (via :meth:`to_loss`) into the
    :data:`~diffBloch.engine.forward.LossFn` ``build_engine`` consumes; only implemented terms are
    admissible. Only knobs the default path actually consumes live here -- outlier rejection,
    penalty/nuisance weighting, and gradient-norm reporting are not accepted config keys until a
    consumer reads them, rather than accepted-but-ignored (cf. penalties, which are Python/API
    composition, not config).
    """

    data_term: Literal["wr2", "least_squares"] = "wr2"

    def to_loss(self) -> LossFn:
        """Parse the data term into the ``LossFn`` the engine scores with."""
        return {
            "wr2": wr2_loss,
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

    def to_spec(self) -> TrainableSpec:
        """Parse into the ``TrainableSpec`` the refinement optimizer consumes."""
        return TrainableSpec(
            positions=_atom_selection(self.positions),
            adp=_atom_selection(self.adp),
            occupancy=_atom_selection(self.occupancy),
        )


class ThicknessNNConfig(_StrictConfig):
    """Recorded apparent-thickness neural network used by the default refinement path."""

    enabled: bool = _THICKNESS_NN_DEFAULTS.enabled
    num_samples: int = _THICKNESS_NN_DEFAULTS.num_samples
    sample_thickness: bool = _THICKNESS_NN_DEFAULTS.sample_thickness
    form: Literal["min_thickness"] = _THICKNESS_NN_DEFAULTS.form
    min_thickness: float = _THICKNESS_NN_DEFAULTS.min_thickness
    max_thickness: float = _THICKNESS_NN_DEFAULTS.max_thickness
    init_seed: int = _THICKNESS_NN_DEFAULTS.init_seed

    def to_spec(self) -> ApparentThicknessNetwork:
        """Parse the YAML block into its validated value-type."""
        return ApparentThicknessNetwork(**self.model_dump())

    @model_validator(mode="after")
    def _parse_fails_fast(self) -> ThicknessNNConfig:
        self.to_spec()
        return self


class RefinementConfig(_StrictConfig):
    """Stable execution knobs for the *default* single-stage app refinement (``run refine``).

    These tune the default path; they do not author a scientific program. Scientific composition
    (hard constraints such as hydrogen riding, soft penalties, freeze-H masks, multi-stage
    workflows) is expressed as typed Python/API values -- see
    :func:`~diffBloch.engine.build_refinement_model`,
    :func:`~diffBloch.engine.build_refinement_problem`, and
    :func:`~diffBloch.engine.with_hydrogen_riding` -- and is promoted to config only once the
    default recipe commits to it as stable public behaviour. ``precision`` selects the refinement
    solve's numeric field: ``"fp64"`` keeps the complex128 default; ``"fp32"`` runs the Bloch solve
    in complex64 for faster/lower-memory refines at reduced decimal precision.
    """

    steps: int = 40
    trainable: TrainableConfig = Field(default_factory=TrainableConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)
    precision: FloatFormat = "fp64"
    split: DataSplitConfig = Field(default_factory=DataSplitConfig)
    thickness_nn: ThicknessNNConfig = Field(default_factory=ThicknessNNConfig)


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


class PreprocessConfig(_StrictConfig):
    """Preprocess-stage configuration (the ``Plan -> Plan`` calibration pipeline).

    Grouping, not composition: each block configures one preprocess step. Only steps the default run
    composes get a config block here: ``fit_orientation`` under ``orientation`` and
    ``fit_thickness`` under ``thickness``. The optional
    ``converge_numerics``
    driver is *not* in the default recipe, so it has no config block -- a caller that composes it
    constructs :class:`~diffBloch.specs.ConvergenceTest` /
    :class:`~diffBloch.specs.ConvergenceTolerance` at the composition site (which carry their own
    defaults). Opt-in step config lives with the step, not in an always-present block.
    """

    optimize_orientation: bool = True
    optimize_thickness: bool = True
    orientation: OrientationFitConfig = Field(default_factory=OrientationFitConfig)
    thickness: ThicknessFitConfig = Field(default_factory=ThicknessFitConfig)


class Inputs(_StrictConfig):
    """Input references — relative to the experiment directory only (no project-root paths)."""

    structure: str
    exp_data: str
    orientations: str | None = None
    load_hydrogens: bool = False  # include hydrogen atom sites (molecular crystals; off by default)

    @field_validator("structure", "exp_data", "orientations")
    @classmethod
    def _relative_path_only(cls, value: str | None) -> str | None:
        if value is None:
            return None
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
    blochwave: BlochwaveConfig = Field(default_factory=BlochwaveConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    """Parse and validate one ``experiment.yaml``.

    Fails fast with a ``pydantic.ValidationError`` at the boundary, rather than a deferred runtime
    surprise deep in the pipeline.
    """
    data = yaml.safe_load(Path(path).read_text())
    return ExperimentConfig.model_validate(data)

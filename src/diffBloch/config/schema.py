"""Pydantic configuration schema for a diffBloch experiment.

Config is validated at the boundary: no Hydra, and no ``DictConfig`` reaches the core. Every field
carries a sensible default ("defaults as code"), so an ``experiment.yaml`` only needs to specify
input references and overrides.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from diffBloch.core.solver import SolverMethod
from diffBloch.engine.losses import (
    least_squares_scores,
    rbragg_loss,
    robs_scores,
    weighted_mse_loss,
    wr2_loss,
    wr2_scores,
)
from diffBloch.engine.refine import AtomSelection, TrainableSpec

if TYPE_CHECKING:
    from diffBloch.engine.forward import LossFn, ScoresFn
from diffBloch.observability import ExperimentDeclared
from diffBloch.specs import (
    Absorption,
    ApparentThicknessNetwork,
    BeamSelection,
    IntegrationGeometry,
    Mosaicity,
    NelderMeadSearch,
    OrientationSelection,
    PerTiltCoupling,
    RockingCurve,
    ThicknessGrid,
    UnionCoupling,
)

# The preprocess config classes below are 1:1 YAML edges over their value-types; their field
# defaults derive from these default instances so the boundary value cannot drift from the
# value-type it parses into. The value-type in ``specs`` is the single source of truth for both the
# default value and its validation rules.
_NELDER_MEAD_DEFAULTS = NelderMeadSearch()
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

    The structure-factor support grid is *not* a config field: it is derived as ``2x`` the
    solve cutoff (``g_max``), because a beam set bounded by ``|g| <= cutoff`` produces ``F(g - h)`` terms
    reaching ``2 * cutoff`` -- so declaring both cutoff and support would let them contradict
    (one beam cutoff, support derived). The orientation-independent seed pool
    (``from_experiment``'s difference-safe candidate set) and the scored-reflection cap
    (``optimize_orientation``'s trial-coupling window) both read ``g_max`` directly, rather than a
    separate smaller radius. ``rsg`` / ``dsg`` are
    the Klar beam-selection cutoffs and ``rocking_curve_sampling`` the tilt count. The shared
    integration semi-angle is read from the PETS experimental data rather than configured.
    ``mosaicity`` is the :class:`Mosaicity` reduction.
    """

    solver: SolverConfig = Field(default_factory=SolverConfig)
    absorption: bool = False
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
    """Train/validation split declaration.

    ``train_test=False`` (default) trains on every rotation -- no held-out validation set, matching
    the behaviour of an experiment.yaml that never mentions ``split``. ``train_test=True`` excludes
    an evenly spaced ``val_frac`` fraction of rotations from the refinement objective (preprocessing
    still fits their orientation/thickness) and reports their held-out wR2/R_obs alongside the
    training-set numbers -- opt in per experiment, since it means training sees less data.
    """

    train_test: bool = False
    val_frac: float = 0.2

    @field_validator("val_frac")
    @classmethod
    def _val_frac_in_range(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError("val_frac must be between 0 and 1 (exclusive)")
        return value


class LossMetricsConfig(_StrictConfig):
    """The one residual driving the whole pipeline: preprocess search AND refinement.

    ``residual`` parses into a matching ``LossFn``/``ScoresFn`` pair (:meth:`to_loss` /
    :meth:`to_scores`) -- the scalar the gradient refinement minimises and the per-thickness vector
    ``optimize_orientation``/``optimize_thickness`` search, off the *same* metric (see
    :mod:`diffBloch.engine.losses`). A top-level ``ExperimentConfig`` field (not scoped under
    ``refinement``) because it governs preprocessing too, not just the gradient stage; only
    implemented terms are admissible. Only knobs the default path actually consumes live here --
    outlier rejection, penalty/nuisance weighting, and gradient-norm reporting are not accepted
    config keys until a consumer reads them, rather than accepted-but-ignored (cf. penalties, which
    are Python/API composition, not config).
    """

    residual: Literal["wr2", "least_squares", "robs"] = "wr2"

    def to_loss(self) -> LossFn:
        """Parse the residual into the scalar ``LossFn`` the gradient refinement minimises."""
        return {
            "wr2": wr2_loss,
            "least_squares": weighted_mse_loss,
            "robs": rbragg_loss,
        }[self.residual]

    def to_scores(self) -> ScoresFn:
        """Parse the residual into the per-thickness ``ScoresFn`` the preprocessing search uses.

        The exact per-thickness form :meth:`to_loss` sums to a scalar -- see
        :mod:`diffBloch.engine.losses` -- so the two always agree on one metric.
        """
        return {
            "wr2": wr2_scores,
            "least_squares": least_squares_scores,
            "robs": robs_scores,
        }[self.residual]


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
    default recipe commits to it as stable public behaviour.
    """

    steps: int = 40
    trainable: TrainableConfig = Field(default_factory=TrainableConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    split: DataSplitConfig = Field(default_factory=DataSplitConfig)
    thickness_nn: ThicknessNNConfig = Field(default_factory=ThicknessNNConfig)


class NelderMeadOptimizationConfig(_StrictConfig):
    """Bounds for the ``optimize_orientation`` local Nelder-Mead search (preprocess).

    The YAML edge: parses (via :meth:`to_search`) into the validated
    :class:`~diffBloch.specs.NelderMeadSearch` value-type.
    """

    step_size: float = _NELDER_MEAD_DEFAULTS.step_size  # degrees
    max_iterations: int = _NELDER_MEAD_DEFAULTS.max_iterations  # scipy `maxiter`
    x_tolerance: float = _NELDER_MEAD_DEFAULTS.x_tolerance  # scipy `xatol`
    f_tolerance: float = _NELDER_MEAD_DEFAULTS.f_tolerance  # scipy `fatol`
    penalize_fewer_reflections: bool = _NELDER_MEAD_DEFAULTS.penalize_fewer_reflections

    def to_search(self) -> NelderMeadSearch:
        """Parse into the validated value-type the pure ``optimize_orientation`` consumes."""
        return NelderMeadSearch(
            step_size=self.step_size,
            max_iterations=self.max_iterations,
            x_tolerance=self.x_tolerance,
            f_tolerance=self.f_tolerance,
            penalize_fewer_reflections=self.penalize_fewer_reflections,
        )

    @model_validator(mode="after")
    def _parse_fails_fast(self) -> NelderMeadOptimizationConfig:
        self.to_search()  # the rules live in NelderMeadSearch; fail fast at config load
        return self


class OrientationOptimizationConfig(_StrictConfig):
    """Bounds for the ``optimize_orientation`` orientation search (preprocess).

    The YAML edge: parses (via :meth:`to_search`) into the validated
    :class:`~diffBloch.specs.NelderMeadSearch` value-type the pure ``optimize_orientation`` consumes
    (the ``nelder_mead`` block), and delegates all validation there (one rule home, no drift).
    """

    nelder_mead: NelderMeadOptimizationConfig = Field(default_factory=NelderMeadOptimizationConfig)

    def to_search(self) -> NelderMeadSearch:
        """Parse into the validated value-type the pure ``optimize_orientation`` consumes."""
        return self.nelder_mead.to_search()

    @model_validator(mode="after")
    def _parse_fails_fast(self) -> OrientationOptimizationConfig:
        self.to_search()  # the rules live in NelderMeadSearch; fail fast at load
        return self


class ThicknessOptimizationConfig(_StrictConfig):
    """Bounds for the ``optimize_thickness`` per-rotation grid search (preprocess).

    The YAML edge: parses (via :meth:`to_grid`) into the validated
    :class:`~diffBloch.specs.ThicknessGrid` value-type the pure ``optimize_thickness`` consumes, and
    delegates all validation there (one rule home, no drift). Defaults derive from that value-type
    (``_THICKNESS_GRID_DEFAULTS``), so the boundary value cannot drift from it either.

    ``plot`` is reporting-only -- it selects whether the CLI attaches a
    :class:`~diffBloch.app.loggers.plotting.ThicknessPlotLogger` (one wR2-vs-thickness PNG per
    rotation, default ``<inputs.structure's directory>/thickness_optim``); it never changes the
    fitted ``Plan``, so :func:`~diffBloch.config.manifest.dataset_config_digest` excludes it explicitly even
    when the rest of this block is in scope.
    """

    min_thickness: float = _THICKNESS_GRID_DEFAULTS.min_thickness  # Angstroms
    max_thickness: float = _THICKNESS_GRID_DEFAULTS.max_thickness  # Angstroms
    n_steps: int = _THICKNESS_GRID_DEFAULTS.n_steps  # evenly-spaced candidates
    plot: bool = False

    def to_grid(self) -> ThicknessGrid:
        """Parse into the validated value-type the pure ``optimize_thickness`` consumes."""
        return ThicknessGrid(
            min_thickness=self.min_thickness,
            max_thickness=self.max_thickness,
            n_steps=self.n_steps,
        )

    @model_validator(mode="after")
    def _parse_fails_fast(self) -> ThicknessOptimizationConfig:
        self.to_grid()  # the rules live in ThicknessGrid; fail fast at config load
        return self


class PreprocessConfig(_StrictConfig):
    """Preprocess-stage configuration (the ``Plan -> Plan`` calibration pipeline).

    Grouping, not composition: each block configures one preprocess step. Only steps the default run
    composes get a config block here: ``optimize_orientation`` under ``orientation`` and
    ``optimize_thickness`` under ``thickness``. The optional
    ``converge_numerics``
    driver is *not* in the default recipe, so it has no config block -- a caller that composes it
    constructs :class:`~diffBloch.specs.ConvergenceTest` /
    :class:`~diffBloch.specs.ConvergenceTolerance` at the composition site (which carry their own
    defaults). Opt-in step config lives with the step, not in an always-present block.
    """

    optimize_orientation: bool = True
    optimize_thickness: bool = True
    # Fitting stage order when both are enabled: orientation then thickness (default) fits
    # orientation against the seed thickness, then thickness against the fitted orientation;
    # "thickness_first" reverses that, fitting thickness against the seed orientation first.
    stage_order: Literal["orientation_first", "thickness_first"] = "orientation_first"
    orientation: OrientationOptimizationConfig = Field(
        default_factory=OrientationOptimizationConfig
    )
    thickness: ThicknessOptimizationConfig = Field(default_factory=ThicknessOptimizationConfig)
    # Path (relative to the experiment directory) to a 'Rotation Index'/'Orientation Matrix' CSV
    # that overwrites every candidate's orientation before the recipe's fitting steps run -- see
    # diffBloch.preprocess.import_orientations. None (default) skips the import; optimize_orientation
    # still controls whether the search refines from the imported seed or is skipped so the CSV's
    # orientations are used as-is. The CLI's --orientations-csv flag overrides this when given.
    orientations_csv: str | None = None


def dataset_checkpoint_stem(ref: str) -> str:
    """The per-dataset checkpoint name stem for an ``exp_data`` ref: ``plan.<stem>.npz``.

    Path separators become ``__`` and a ``.cif_pets`` suffix is dropped, e.g.
    ``undamaged/frame_1.cif_pets -> undamaged__frame_1``. Checkpoint identity follows the *file*,
    not its position in ``exp_data``, so reordering or inserting datasets never restales another
    dataset's checkpoint. Two refs may sanitize to the same stem (``a/b`` vs ``a__b``);
    :class:`Inputs` rejects such configs up front rather than letting two datasets share one
    checkpoint on disk.
    """
    stem = ref.removesuffix(".cif_pets")
    return "__".join(Path(stem).parts)


def _relative_path_only(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("input references must be relative paths within the experiment directory")
    return value


class Inputs(_StrictConfig):
    """Input references — relative to the experiment directory only (no project-root paths)."""

    structure: str
    exp_data: str | list[str]
    # Pool rotations from every file in exp_data into one experiment. False (default) keeps
    # exp_data a single path. True requires exp_data to be a list of 2+ distinct paths. Each file
    # is preprocessed and checkpointed on its own, with its own integration geometry (precession
    # angles may differ), and the settled per-file plans are pooled in memory before refinement;
    # the files' wavelength-derived beam energies must snap to the same voltage, since the engine
    # solves the whole pooled experiment at one energy (see preprocess.pool).
    multi_dataset: bool = False
    load_hydrogens: bool = False  # include hydrogen atom sites (molecular crystals; off by default)

    @field_validator("structure")
    @classmethod
    def _structure_relative_path(cls, value: str) -> str:
        return _relative_path_only(value)

    @field_validator("exp_data")
    @classmethod
    def _exp_data_relative_paths(cls, value: str | list[str]) -> str | list[str]:
        if isinstance(value, list):
            return [_relative_path_only(v) for v in value]
        return _relative_path_only(value)

    @model_validator(mode="after")
    def _multi_dataset_shape(self) -> Inputs:
        if self.multi_dataset:
            if not isinstance(self.exp_data, list) or len(self.exp_data) < 2:
                raise ValueError(
                    "inputs.multi_dataset=true requires inputs.exp_data to be a list of 2+ paths"
                )
            if len(set(self.exp_data)) != len(self.exp_data):
                raise ValueError(
                    "inputs.exp_data lists the same dataset more than once; pooling a file twice "
                    "would double-weight its reflections in the refinement"
                )
            stems: dict[str, str] = {}
            for ref in self.exp_data:
                stem = dataset_checkpoint_stem(ref)
                if stem in stems:
                    raise ValueError(
                        f"inputs.exp_data entries {stems[stem]!r} and {ref!r} map to the same "
                        f"checkpoint name plan.{stem}.npz; rename one so per-dataset checkpoints "
                        "stay distinct"
                    )
                stems[stem] = ref
        elif isinstance(self.exp_data, list):
            raise ValueError(
                "inputs.exp_data is a list but inputs.multi_dataset is false -- set "
                "multi_dataset=true to pool multiple datasets, or use a single path"
            )
        return self


class ExperimentConfig(_StrictConfig):
    """A whole experiment, validated at load. No Hydra, no ``DictConfig``."""

    name: str
    inputs: Inputs
    sample: SampleConfig = Field(default_factory=SampleConfig)
    blochwave: BlochwaveConfig = Field(default_factory=BlochwaveConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    # Top-level, not under `refinement`: this metric drives BOTH the orientation/thickness
    # preprocessing search and the gradient refinement loss (see LossMetricsConfig), so it isn't
    # scoped to either stage.
    loss_metrics: LossMetricsConfig = Field(default_factory=LossMetricsConfig)
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)

    def to_declaration(self, integrations: Sequence[IntegrationGeometry]) -> ExperimentDeclared:
        """Project the result-determining knobs onto the run's declaration event.

        One more ``to_*`` edge alongside :meth:`BlochwaveConfig.to_policy` /
        :meth:`~BlochwaveConfig.to_absorption`: the config already owns every value here, so mapping
        them lives with it rather than in whichever caller happens to emit the event. Any backend
        (W&B/Comet hyperparameters, the written summary) reads the run's settings from this one
        event instead of being handed the config object. ``integrations`` is one PETS-derived
        geometry per ``inputs.exp_data`` entry, in that order.
        """
        experimental_data = (
            ", ".join(self.inputs.exp_data)
            if isinstance(self.inputs.exp_data, list)
            else self.inputs.exp_data
        )
        return ExperimentDeclared(
            name=self.name,
            structure=self.inputs.structure,
            experimental_data=experimental_data,
            optimizer=self.refinement.optimizer.name,
            seed_thicknesses=tuple(self.sample.thicknesses),
            integration_semiangles=tuple(integration.semiangle for integration in integrations),
            rocking_curve_sampling=self.blochwave.rocking_curve_sampling,
            dsg=self.blochwave.dsg,
            rsg=self.blochwave.rsg,
            solve_g_max=self.blochwave.g_max,
            sg_max=self.blochwave.sg_max,
            absorption=self.blochwave.absorption,
            steps=self.refinement.steps,
            learning_rate=self.refinement.optimizer.lr,
        )


def load_config(path: str | Path) -> ExperimentConfig:
    """Parse and validate one ``experiment.yaml``.

    Fails fast with a ``pydantic.ValidationError`` at the boundary, rather than a deferred runtime
    surprise deep in the pipeline.
    """
    data = yaml.safe_load(Path(path).read_text())
    return ExperimentConfig.model_validate(data)

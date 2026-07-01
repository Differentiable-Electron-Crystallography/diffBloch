"""Pydantic configuration schema for a diffBloch experiment.

Config is validated at the boundary: no Hydra, and no ``DictConfig`` reaches the core. Every field
carries a sensible default ("defaults as code"), so an ``experiment.yaml`` only needs to specify
input references and overrides. See the synthesis notebook
(``notebooks/iain/principled_refactor_synthesis.ipynb``) §5.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from diffBloch.specs import BeamSelection, HexagonalSearch, RockingCurve, ThicknessGrid

# The preprocess config classes below are 1:1 YAML edges over their value-types; their field
# defaults derive from these default instances so the boundary value cannot drift from the
# value-type it parses into. The value-type in ``specs`` is the single source of truth for both the
# default value and its validation rules (e.g. the quartz-calibrated ``max_iterations`` lives once,
# in ``HexagonalSearch``).
_HEXAGONAL_SEARCH_DEFAULTS = HexagonalSearch()
_THICKNESS_GRID_DEFAULTS = ThicknessGrid()


class SolverConfig(BaseModel):
    """Which dynamical solver to use for each phase."""

    refine: str = "matrix_exp"  # gradient-safe default for the refinement (backprop) path
    inference: str = "bloch_eigen"  # pinned to match existing e2e references


class NumericsConfig(BaseModel):
    """Stage-3 numerical-accuracy controls, frozen into the simulation spec."""

    g_max: float = 4.5
    g_max_sf: float = 4.5
    g_max_refine: float = 1.6
    sg_max: float = 0.01
    rocking_curve_sampling: int = 42
    dsg: float = 0.0015
    rsg: float = 0.9
    integration_semiangle: float = 1.0

    def to_beam_selection(self) -> BeamSelection:
        """Parse the beam-selection subset into the value-type ``select_beams`` consumes."""
        return BeamSelection(
            rsg=self.rsg, dsg=self.dsg, integration_semiangle=self.integration_semiangle
        )

    def to_rocking_curve(self) -> RockingCurve:
        """Parse the rocking-curve subset into the value-type ``integrate_rocking_curve`` consumes.

        ``integration_semiangle`` doubles as the tilt half-width (one physical angular integration
        range shared with the Klar beam window; see the decision doc), and
        ``rocking_curve_sampling`` is the tilt count. Geometry defaults to continuous rotation until
        ``data_collection_geometry`` is surfaced from the PETS reader (a deferred discriminated
        mode).
        """
        return RockingCurve(
            semiangle=self.integration_semiangle, sampling=self.rocking_curve_sampling
        )

    @model_validator(mode="after")
    def _parse_fails_fast(self) -> NumericsConfig:
        self.to_beam_selection()  # the rules live in BeamSelection; fail fast at config load
        self.to_rocking_curve()  # the rules live in RockingCurve; fail fast at config load
        return self


class SampleConfig(BaseModel):
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


class DataSplitConfig(BaseModel):
    """Required train/validation split declaration.

    The concrete selector language is intentionally small for Stage 1: it records the fixed split
    policy that later dataset code will materialize into ``data_used``.
    """

    train: str = "all_except_validation"
    validation: str = "every_10th_rotation"


class ObjectiveConfig(BaseModel):
    """First-class target composition, not one opaque scalar loss."""

    data_term: Literal["weighted_r", "poisson_nll", "least_squares"] = "weighted_r"
    outlier_rejection: Literal["none", "tukey", "sigma_clip"] = "none"
    restraints_weight: float = 1.0
    nuisance_weight: float = 1.0
    report_gradient_norms: bool = True


class OptimizerConfig(BaseModel):
    """Explicit optimizer backend for a refinement stage."""

    name: Literal["lbfgs", "adam", "adamw", "least_squares"] = "lbfgs"
    lr: float = 1e-3
    max_line_search_steps: int = 20


class BeamDamageConfig(BaseModel):
    """Optional inline beam-damage step (off by default; see synthesis §17).

    ``activate`` must be true (here or via ``engine.activate("beam_damage")``) before ``b_dose`` can
    be a refinement target — target selection alone never activates an optional component.
    """

    activate: bool = False
    model: str = "analytic"  # "analytic" | "nn"
    b_dose_init: float = 0.0


class ObservationConfig(BaseModel):
    """Observation-model components applied to simulated intensities before the loss."""

    beam_damage: BeamDamageConfig = Field(default_factory=BeamDamageConfig)


class RefinementConfig(BaseModel):
    """Default refinement-stage hyperparameters."""

    steps: int = 500
    targets: tuple[str, ...] = ("positions", "adp")
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)
    split: DataSplitConfig = Field(default_factory=DataSplitConfig)


class OrientationFitConfig(BaseModel):
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


class ThicknessFitConfig(BaseModel):
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


class PreprocessConfig(BaseModel):
    """Preprocess-stage configuration (the ``Plan -> Plan`` calibration pipeline).

    Grouping, not composition: each block configures one preprocess step. ``fit_orientation`` and
    ``fit_thickness`` are wired today; the ``converge_numerics`` block joins here when it lands.
    """

    orientation: OrientationFitConfig = Field(default_factory=OrientationFitConfig)
    thickness: ThicknessFitConfig = Field(default_factory=ThicknessFitConfig)


class Inputs(BaseModel):
    """Input references — relative to the experiment directory only (no project-root paths)."""

    structure: str
    observations: str
    orientations: str | None = None

    @field_validator("structure", "observations", "orientations")
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


class ExperimentConfig(BaseModel):
    """A whole experiment, validated at load. No Hydra, no ``DictConfig``."""

    name: str
    inputs: Inputs
    sample: SampleConfig = Field(default_factory=SampleConfig)
    numerics: NumericsConfig = Field(default_factory=NumericsConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    observation: ObservationConfig = Field(default_factory=ObservationConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    """Parse and validate one ``experiment.yaml``.

    Fails fast with a ``pydantic.ValidationError`` at the boundary, rather than a deferred runtime
    surprise deep in the pipeline.
    """
    data = yaml.safe_load(Path(path).read_text())
    return ExperimentConfig.model_validate(data)

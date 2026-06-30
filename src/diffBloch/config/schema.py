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

    Declarative, sweepable home at the boundary; the preprocess driver unpacks these into the plain
    keyword arguments of :func:`diffBloch.preprocess.fit_orientation` (no pydantic model reaches the
    pure function). Defaults are the faithful ``diffBloch_private`` values
    (``configs/preprocess/base.yaml``); ``max_iterations`` has no private precedent (the private
    search has no cap) and its default is an uncalibrated runaway guard (see KNOWN_ISSUES.md), to be
    tuned once real-data convergence is known.
    """

    max_search_angle: float = 0.4  # degrees: largest tilt radius the search starts from
    min_search_angle: float = 0.001  # degrees: radius floor that terminates the search
    n_steps: int = 6  # hexagonal azimuths per ring (6 -> 0, 60, ..., 300 deg)
    max_iterations: int = 200  # runaway guard: max search passes per orientation (uncalibrated)

    @field_validator("min_search_angle", "max_search_angle")
    @classmethod
    def _positive_angles(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("search angles must be positive")
        return value

    @field_validator("n_steps")
    @classmethod
    def _at_least_one_step(cls, value: int) -> int:
        if value < 1:
            raise ValueError("n_steps must be >= 1")
        return value

    @field_validator("max_iterations")
    @classmethod
    def _at_least_one_iteration(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_iterations must be >= 1")
        return value

    @model_validator(mode="after")
    def _max_exceeds_min(self) -> OrientationFitConfig:
        if self.max_search_angle <= self.min_search_angle:
            raise ValueError("max_search_angle must exceed min_search_angle")
        return self


class ThicknessFitConfig(BaseModel):
    """Bounds for the ``fit_thickness`` per-rotation grid search (preprocess).

    Declarative, sweepable home at the boundary; the preprocess driver unpacks these into the plain
    keyword arguments of :func:`diffBloch.preprocess.fit_thickness` (no pydantic model reaches the
    pure function). For each rotation the step evaluates ``n_steps`` candidate thicknesses spaced
    evenly from ``min_thickness`` to ``max_thickness`` (inclusive) and keeps the one with the lowest
    weighted R-factor. Defaults are the faithful ``diffBloch_private`` values
    (``configs/preprocess/base.yaml``: 5 A to 2000 A in 100 steps).
    """

    min_thickness: float = 5.0  # Angstroms: smallest candidate thickness
    max_thickness: float = 2000.0  # Angstroms: largest candidate thickness
    n_steps: int = 100  # number of evenly-spaced candidates (inclusive endpoints)

    @field_validator("min_thickness", "max_thickness")
    @classmethod
    def _positive_thickness(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("thickness bounds must be positive")
        return value

    @field_validator("n_steps")
    @classmethod
    def _at_least_one_step(cls, value: int) -> int:
        if value < 1:
            raise ValueError("n_steps must be >= 1")
        return value

    @model_validator(mode="after")
    def _max_exceeds_min(self) -> ThicknessFitConfig:
        if self.max_thickness <= self.min_thickness:
            raise ValueError("max_thickness must exceed min_thickness")
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

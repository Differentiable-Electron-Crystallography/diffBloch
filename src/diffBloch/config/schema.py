"""Pydantic configuration schema for a diffBloch experiment.

Config is validated at the boundary: no Hydra, and no ``DictConfig`` reaches the core. Every field
carries a sensible default ("defaults as code"), so an ``experiment.yaml`` only needs to specify
input references and overrides. See the synthesis notebook
(``notebooks/iain/principled_refactor_synthesis.ipynb``) §5.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


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
    lr: float = 1e-3
    targets: tuple[str, ...] = ("positions", "adp")


class Inputs(BaseModel):
    """Input references — relative to the experiment directory only (no project-root paths)."""

    structure: str
    observations: str
    orientations: str | None = None


class ExperimentConfig(BaseModel):
    """A whole experiment, validated at load. No Hydra, no ``DictConfig``."""

    name: str
    inputs: Inputs
    numerics: NumericsConfig = Field(default_factory=NumericsConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    observation: ObservationConfig = Field(default_factory=ObservationConfig)
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    """Parse and validate one ``experiment.yaml``.

    Fails fast with a ``pydantic.ValidationError`` at the boundary, rather than a deferred runtime
    surprise deep in the pipeline.
    """
    data = yaml.safe_load(Path(path).read_text())
    return ExperimentConfig.model_validate(data)

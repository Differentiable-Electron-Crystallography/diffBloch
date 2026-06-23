"""Experiment configuration (Pydantic v2; validated at the boundary, no Hydra)."""

from diffBloch.config.schema import ExperimentConfig, load_config

__all__ = ["ExperimentConfig", "load_config"]

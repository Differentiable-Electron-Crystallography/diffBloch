# Config

The experiment configuration schema. Pydantic validates the config at the boundary — no Hydra, no
`DictConfig` reaches the core. Every field has a sensible default, so an `experiment.yaml` only
specifies input references and overrides.

::: diffBloch.config.schema

# Config And Manifests

The experiment configuration schema. Pydantic validates the config at the boundary — no Hydra, no
`DictConfig` reaches the core. Every field has a sensible default, so an `experiment.yaml` only
specifies input references and overrides.

::: diffBloch.config.schema

Experiment locks hash input bytes only; run manifests hash generated artifacts. The working run
format is a directory, with zip/tar/BagIt/RO-Crate as export formats.

::: diffBloch.config.manifest

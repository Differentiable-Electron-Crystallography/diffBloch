# Config And Manifests

The experiment configuration schema. Pydantic validates the config at the boundary — no Hydra, no
`DictConfig` reaches the core. Every field has a sensible default, so an `experiment.yaml` only
specifies input references and overrides.

```{eval-rst}
.. automodule:: diffBloch.config.schema
```

Experiment locks hash input bytes only; the preprocess and refinement locks hash generated
artifacts.

```{eval-rst}
.. automodule:: diffBloch.config.manifest
```

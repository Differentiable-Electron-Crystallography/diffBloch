# App

The imperative shell: the `diffbloch` CLI, the default experiment runner it delegates to, and the
logger backends (console and structured report always available; Weights & Biases and Comet behind optional extras —
vendor SDKs never reach the core).

```{eval-rst}
.. automodule:: diffBloch.app.cli
```

```{eval-rst}
.. automodule:: diffBloch.app.program
```

```{eval-rst}
.. automodule:: diffBloch.app.loggers
```

```{eval-rst}
.. automodule:: diffBloch.app.loggers.wandb
```

```{eval-rst}
.. automodule:: diffBloch.app.loggers.comet
```

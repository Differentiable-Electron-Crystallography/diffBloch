# Observability

Domain observations as typed events with pluggable logger sinks. The pure core *emits* events as
plain values; a `Logger` attached at the app boundary interprets them (the null default discards
them). Solver diagnostics ride stdlib `logging` instead — two channels, two mechanisms.

```{eval-rst}
.. automodule:: diffBloch.observability
```

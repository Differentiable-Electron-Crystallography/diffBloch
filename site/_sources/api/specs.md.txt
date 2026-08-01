# Specs

Validated value-types for the preprocess algorithms — beam selection, rocking-curve geometry,
mosaicity, orientation search, convergence sweeps. Pydantic parses YAML at the boundary and hands
these frozen dataclasses to the steps, so the algorithm contract stays pydantic-free; invalid
bounds are unrepresentable by construction.

```{eval-rst}
.. automodule:: diffBloch.specs
```

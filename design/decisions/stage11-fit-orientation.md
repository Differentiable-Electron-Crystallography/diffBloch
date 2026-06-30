# Decision: `fit_orientation` — objective, parameterization, and optimizer

**Status:** accepted (slice 11 (5/n), pre-implementation).
**Context:** per-rotation refinement of the crystal orientation in the preprocess pipeline.

## Objective — wR2 from the forward simulation (not a geometry-only fit)

The faithful objective (private `programs/preprocess.py::orientation_optim`) is **wR2**
(`w_rbragg`) computed from the **full forward Bloch simulation** with the trial orientation, scaled
against the observed intensities. An earlier sketch assumed a geometry-only `Σ|Sg|` least-squares;
that is rejected — it ignores the dynamical-intensity matching that is the entire reason to refine
orientation against electron-diffraction data.

## Parameterization — right-multiplied delta rotation

Per-rotation correction `new_orientation = orientation @ goniometer_rotation(α, β, ω)`
(right-multiplied; private convention `rotation @ construct_rotation_matrix(...)`). The three
correction angles are the only refined quantities per rotation. Because the delta is a true
rotation, `det` and the non-orthonormal `U = UB·B⁻¹` measured-cell correction are preserved exactly
— this is how the **re-orthonormalization trap** (`KNOWN_ISSUES.md`) is avoided *by construction*,
not by a post-hoc guard. Reuses the existing `preprocess.goniometer_rotation`.

## Shape — a `PlanStep` factory over a captured `RefinementSetup` (the Reader pattern)

The objective needs the forward kernel + structure + observed data, so `fit_orientation` is a
`PlanStep` **factory that closes over the read-only `RefinementSetup`**:

```python
fit_orientation(refinement, *, loss, optimizer) -> (Plan -> Plan)
```

A functional-references pass (Elm, OCaml, Elixir, the functional-core/imperative-shell pattern)
settles the "is a simulation inside a `Plan -> Plan` step a side effect?" question:

- A **deterministic** simulation is **not** an effect. Elm reserves `Cmd`/`Sub` for the *outside
  world / non-determinism* (HTTP, random, time); deterministic transformation "stays in the world of
  Elm, writing functions." OCaml 5 effect handlers are for *control*, not pure compute.
  Functional-core/imperative-shell (Bernhardt) keeps IO in the shell; pure compute is the core.
  Our forward sim is referentially transparent (same `Plan` + structure → same corrected `Plan`),
  so it is **pure-but-expensive core compute**, not a side effect.
- The extra read-only context is supplied by the **Reader / environment monad**: "a value in an
  environment monad is equivalent to a function with an additional, anonymous argument." Capturing
  `RefinementSetup` by partial application and returning `Plan -> Plan` is exactly this. Elixir's own
  anti-pattern guidance echoes it: split arguments into "data that may change" vs "read-only".

So the `Plan` is the changing data threaded through the pipeline; `RefinementSetup` is captured
environment. Only the **observation** of the search (trace logging, whole-`Plan` checkpoint, typed
domain events) is an effect, routed to the imperative shell per
`design/decisions/effects-and-observability.md` (realized in stage 12).

## Optimizer — A chosen, B recorded

- **(A, chosen) Palatinus modified simplex** — gradient-free hexagonal search with shrinking radius
  (Palatinus et al., *Acta Cryst.* A69, 171–188, 2013). Robust on the non-convex orientation
  landscape, matches the private implementation and the cited method, and lets us anchor against the
  real `R_obs = 0.0438`.
- **(B, recorded option) Gradient-based on the 3 angles** — leverages the differentiable core and is
  simpler, but risks local minima on a problem the original authors deliberately solved
  gradient-free. If adopted later it is a divergence from the original and earns a `DIVERGENCE.md`
  entry.

## Sequencing

`fit_orientation` needs a reusable **wR2-scoring seam** first: `(trial orientation,
RefinementSetup) → forward sim → integrate intensities → scale against observed → wR2`. Build that
pure scoring function (**5a**), then the Palatinus search on top (**5b**).

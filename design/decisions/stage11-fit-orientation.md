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

Per-rotation correction `new_orientation = orientation @ delta` (**right-multiplied**; private
convention). Because `delta` is a true rotation (`det = 1`), `det(orientation)` and the
non-orthonormal `U = UB·B⁻¹` measured-cell correction are preserved exactly — this is how the
**re-orthonormalization trap** (`KNOWN_ISSUES.md`) is avoided *by construction*, not by a post-hoc
guard. The form of `delta` is tied to the optimizer (the two are not interchangeable):

- **(A, chosen — Palatinus) hexagonal tilt** `delta = R_z(φ)·R_x(θ)·R_z(-φ)` — a tilt of magnitude
  `θ` (the current search radius) about an in-plane axis at azimuth `φ ∈ {0,60,…,300}°`. **2 DOF**
  per step (`φ, θ`). Private `generate_new_tilt`. This is what `fit_orientation` uses.
- **(B, recorded option — gradient/continuous) 3-angle goniometer**
  `delta = goniometer_rotation(α, β, ω)` — the continuous parameterization the private's Nelder-Mead
  / Bayesian objective (`orientation_optim`, `construct_rotation_matrix`) refines. Pairs with a
  continuous optimizer, not the hexagonal search.

## Shape — a `PlanStep` factory that captures the read-only `RefinementSetup`

The objective needs the forward kernel + structure + observed data, so `fit_orientation` is a
`PlanStep` **factory that closes over the read-only `RefinementSetup`**:

```python
fit_orientation(refinement, *, loss, optimizer) -> (Plan -> Plan)
```

A survey of how purely-functional languages draw the line (Elm, OCaml, Elixir, the
functional-core/imperative-shell pattern) settles the "is a simulation inside a `Plan -> Plan` step
a side effect?" question. The short answer: no. A side effect is something that touches the outside
world or is non-deterministic; a calculation that only reads its inputs and returns a value is not,
no matter how expensive.

- A **deterministic** simulation is **not** an effect. Elm reserves `Cmd`/`Sub` for the *outside
  world / non-determinism* (HTTP, random, time); deterministic transformation "stays in the world of
  Elm, writing functions." OCaml 5 effect handlers are for *control*, not pure compute.
  Functional-core/imperative-shell (Bernhardt) keeps IO in the shell; pure compute is the core.
  Our forward sim depends only on its inputs (same `Plan` + structure -> same corrected `Plan`),
  so it is **expensive-but-pure calculation**, not a side effect.
- The extra read-only context is supplied by **capturing it as a hidden argument**: a function that
  needs some unchanging background data can either take it as an explicit parameter every call, or
  be built once with that data baked in. Capturing `RefinementSetup` by partial application and
  returning `Plan -> Plan` is the second option. (In functional-programming terms this is the
  "Reader" or "environment" idea -- "a value in an environment is equivalent to a function with an
  additional, anonymous argument" -- and Elixir's own guidance echoes it: split arguments into
  "data that may change" vs "read-only".)

So the `Plan` is the changing data threaded through the pipeline; `RefinementSetup` is the captured
background data. Only the **observation** of the search (trace logging, whole-`Plan` checkpoint,
typed domain events) is an effect, routed to the imperative shell per
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

`fit_orientation` needs a reusable **wR2-scoring step** first: `(trial orientation,
RefinementSetup) → forward sim → integrate intensities → scale against observed → wR2`. Build that
pure scoring function (**5a**), then the Palatinus search on top (**5b**).

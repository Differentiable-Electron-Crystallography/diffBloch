# Stage 10 — the refinement loop: an imperative island in a pure core

## Context

Stages 1–9 are pure, differentiable, frozen-dataclass code. Stage 10's `RefinementEngine.objective`
is the same: `params -> simulated diffraction -> scalar loss`, differentiable end to end. But
*refinement* — actually minimising that loss — needs a `torch.optim` optimizer, and torch optimizers
are irreducibly imperative: they mutate `.grad` and leaf tensors in place and carry internal state
(LBFGS history, Adam moments). A training loop cannot be a pure function.

This is the first deliberately-stateful corner of the codebase. The decision is *where* the seam
lives and how the imperativeness is contained.

## Decision

**Option A (chosen): one `engine/` package, the imperative loop quarantined in its own module.**

```
engine/
  __init__.py   # re-exports the public surface
  plan.py       # ScatteringGrid, OrientationPlan        (refinement-invariant geometry)
  forward.py    # RefinementEngine.objective / simulate    (pure, differentiable spine)
  refine.py     # run_refinement + RefinementResult        (the torch.optim loop)
```

- The invariant is **method-level, not module-graph-level**: `simulate()` and `objective()` are
  optimizer-free pure functions, `refine()` is the single imperative entry point, and `core/` never
  imports `torch.optim`. `forward.py` *does* import `refine.py` at the top level (the
  `RefinementEngine.refine` facade delegates to `run_refinement`) — and that is fine: there is no
  import cycle (`refine.py` imports only `params` + torch), and the property worth protecting is the
  purity of `simulate`/`objective`, which holds regardless of the import edge. We deliberately do
  *not* hide that edge behind a lazy in-method import: doing so would advertise a module decoupling
  that does not exist (the class still calls the optimizer loop) at the cost of comprehension.
- `RefinementEngine.refine(...)` stays a method (faithful to the roadmap's chainable-engine design)
  but is a thin delegate: it hands its own pure `objective` callable to
  `engine.refine.run_refinement`, which owns all the mutation.

### The functional contract over the imperative core

`run_refinement(objective, params, ...)` does **not** mutate the caller's `params`:

1. The selected *target* fields are cloned into fresh `requires_grad=True` leaves; every non-target
   field becomes a detached constant clone (`_to_leaves`).
2. A backend (`adam`/`adamw`/`lbfgs`) steps those leaves for `steps` iterations via a single
   `closure` — which unifies LBFGS' multi-evaluation line search with Adam/AdamW's single step.
3. A new, fully-detached `RefinementResult` is returned (`params`, the `losses` trajectory,
   `best_params`/`best_step`).

So the public surface reads functionally (`params_in -> result`); the in-place mutation is sealed
inside one function over throwaway leaves.

### Loss/curve semantics

`losses[i]` is the objective **before** step `i`'s update (the value `optimizer.step(closure)`
returns — `orig_loss` for LBFGS, the single evaluation for Adam). `best_params` snapshots the params
that produced the lowest recorded loss, paired correctly by snapshotting *before* each step.

## Rejected / deferred

- **Option B — a separate top-level `optimize/` (or `/diff`) package with a free `refine(engine,…)`
  function** (optax-style pure-model + separate driver). Clean, but it contradicts the documented
  chainable `engine.refine(...)` method and the engine-owned `OptimizerState`/history/`activate`
  design. `/diff` was also rejected on naming grounds (ambiguous against *diffraction* /
  *difference*). Revisit only as a deliberate move to an optax-style architecture.
- **Per-group learning rates** (private `init_optim` uses distinct LRs for positions vs ADP vs
  thickness). Deferred: single shared `lr` until there's evidence it's needed. The `_TARGET_FIELDS`
  mapping already groups by target, so per-group LRs slot in without restructuring.
- **`least_squares` (Gauss–Newton / Levenberg–Marquardt)** — a different algorithm, not a
  `torch.optim` backend. Deferred; the `OptimizerName` literal is the extension point.
- **Component `activate(...)`** (beam damage, thickness NN). The config rule "target selection never
  silently activates a component" still holds; there is simply nothing optional to activate yet
  (those components are deferred), so `activate` is not built. `b_dose` is intentionally absent from
  `_TARGET_FIELDS` for this reason.
- **`OptimizerState`, threaded `torch.Generator`, `snapshot`/`history`, `from_*` constructors** —
  the stateful/resumable surface from the roadmap. `RefinementResult` is the minimal honest result
  for now; the richer state is a later slice.
- **Multi-thickness reduction policy** beyond summation, and refinable thickness wiring.
- **Thickness placement is provisional.** `thicknesses` lives on the engine as one shared tensor
  across orientations; this is adequate for the synthetic/anchor case but not physically right for
  real multi-rotation data (an irregular specimen presents a different path length per rotation).
  Stage 11 moves thickness *per-rotation* into `OrientationPlan` (fit in preprocess, frozen into the
  `Plan`) — see ROADMAP stage 11.

## Sensitivity note (for test authors)

`refine` loss-decrease tests must use a *strongly scattering* synthetic case. A thin/weak crystal
gives absolute loss gradients ~1e-6; LBFGS (which uses gradient *magnitude*) then barely moves while
Adam (scale-invariant) progresses, producing a misleading "lbfgs doesn't work" failure. The engine
tests use thickness ~300 Å so the system is genuinely dynamical (diffracted intensity ~0.1, O(1)
gradients) and both optimizers demonstrably reduce the loss. Occupancy is a clean lever (it scales
every `F` linearly); ADP is a poor lever at small `|g|` (the Debye–Waller argument is ~1e-6 there).

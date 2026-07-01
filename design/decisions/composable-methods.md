# Decision: experimental methods are composed from swappable typed units

**Status:** accepted (stage 11).
**Context:** this is scientific software. A modeller must be able to make claims of the form
*"enabling
mosaicity improved R_obs"*, *"the bloch_eigen solver agrees with matrix_exp here"*, *"this loss /
this refinement approach changed the outcome"* — and support them by **toggling one thing and
re-running**, not by editing the model. So forward-model knobs (mosaicity, rocking-curve sampling),
preprocess steps, the solver, the loss, and refinement *approaches* should all be **composable,
swappable units**, never baked into a monolith.
**Reference:** the convergence decision (`stage11-convergence.md`, "the convergence engine is a
higher-order component") and the mosaicity knob (`stage11-rocking-curve.md`, decision 4) are two
instances of this one principle; this records the general rule.

## The principle

> An experimental *method* is a **composition of swappable, named, typed units** — not a subclass, a
> config-reflection plugin registry, or a god-object with flags. You measure the effect of a unit by
> composing it in or out and comparing results.

This is the same guardrail as the convergence HOF: abstract at the **typed-closure / composition**
level, not at a config-path-reflection level. Prefer building on the project's own combinators over
adopting a framework.

## It is already the architecture (map to existing seams)

The composable units exist today as small typed seams; nothing new is needed to honour the
principle:

| Unit (swap to run an experiment) | Seam | Compose via |
|---|---|---|
| Solver (`matrix_exp` refine · `bloch_eigen` eval) | `type Method` (`core.solver`) | `method=` argument |
| Loss / objective | `type LossFn` (`engine.forward`) | `loss=` on the engine |
| Preprocess step (`select_beams`, `fit_orientation`, `fit_thickness`, `converge_*`) | `type PlanStep = Plan -> Plan` (`preprocess.pipeline`) | `pipeline([...])`, `iterate_until` |
| Rocking-curve mosaicity (off by default) | a tilt-reduction `PlanStep` / value-type toggle (`stage11-rocking-curve.md` §4) | add/remove the composed step |
| Convergence over a knob | `converge_scalar` HOF over a typed closure | `converge_beams`, coordinate descent |

A *refinement approach* is therefore a **composition** of: preprocess `PlanStep`s (the
`Plan -> Plan`
transformers) → a terminal that runs the forward model with a chosen `method` + `loss` +
optimizer/targets. Swapping any unit is an experiment; the composition *is* the method.

## Terminals: inference and refinement are the same family

The pipeline's terminal (ROADMAP stage 11: transformers + a terminal estimator) has two members that
share the forward spine:

- **`run_inference`** (`preprocess`): eval-only — run the forward model once per rotation under
  `no_grad`, score (`R_obs`/`wR2`), return per-rotation metrics. The 2.0 analog of the private
  `evaluate_over_rotations`. This is the C1 terminal (`ROADMAP.md`, plan C).
- **`engine.refine`** (`engine`): optimize structure targets over the same forward `objective`.

Both take the same swappable `method`; `refine` additionally takes `loss`, optimizer, and targets.
Modelling refinement *approaches* as composed/swappable methods means: no new plugin machinery — a
new
approach is a new composition of these typed units (a preprocess pipeline + a terminal parameterised
by `method`/`loss`/targets), added and measured by composition.

## Guardrail

- Compose **typed units**, not config strings reflected into behaviour (no Hydra-style
  `_target_` instantiation in core).
- A unit is **off/identity by default** where that keeps the invariant (mosaicity off; `preprocess`
  optional). Turning it on is an explicit compose, so its effect is attributable.
- Persistence stays value-based (checkpoint the whole `Plan`; see `effects-and-observability.md`) —
  a composed method is reproduced by replaying the composition over the same inputs, not by a saved
  registry state.

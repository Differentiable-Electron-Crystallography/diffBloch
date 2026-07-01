# Decision (deferred): encode the pipeline as config feature flags, not a string-reflection registry

**Status:** deferred (direction agreed; implementation held).
**Context:** the preprocess/refinement pipeline is composed in code today (the anchor and tutorials
build `pipeline([select_beams(...), fit_orientation(...), ...])` explicitly). Question raised:
should the *composition* — which steps are enabled, with which parameters — also live in the
experiment config, so a run is reproducible and a researcher can compose approaches in/out
declaratively?

## The tension

Two recorded principles pull against a naive "pipeline in config":

- **Composable methods** (`composable-methods.md`): *compose typed units, not config-string
  reflection*. A generic `steps: ["select_beams", "fit_orientation"]` list resolved through a
  registry / `getattr` is the Hydra-style instantiate-from-string anti-pattern — it loses type
  safety and invites arbitrary, unvalidated composition.
- **Anti-Hydra config** (`AGENTS`/config convention): one validated pydantic `ExperimentConfig` at
  the boundary, **defaults-as-code**, no `DictConfig` in core.

## Decision (direction): feature flags + an explicit typed builder

The industry-standard **feature-flag** shape reconciles reproducibility with both principles:

- **Config carries per-step feature flags** — `enabled: bool` plus that step's parameters (the
  parameters already exist as the `*Config` blocks / value-types). Off by default.
- **An explicit builder maps validated flags → typed units** in a **canonical, code-owned order**,
  e.g. `if cfg.rocking_curve.enabled: steps.append(integrate_rocking_curve(...))`.
  This is a plain `match`/if-chain over a **closed** set — not reflection, not a registry.
- **Off/identity by default preserves the invariant**: no flag on → the canonical single-solve
  pipeline, byte-identical.
- **Dependencies validated at the boundary**: e.g. `mosaicity.enabled` requires
  `rocking_curve.enabled` (mosaicity modifies the tilt reduction; see the rocking-curve doc).
- **Reproducible**: the resolved flags + parameters are what the run records
  (`run_manifest`/`experiment.lock`), so a run replays from config.

### Why feature flags over an ordered discriminated-union step list

An ordered `pipeline: [{step: ..., ...}, ...]` list would let config author *arbitrary* order and
composition — more powerful, but that is precisely the Hydra-like "config authors the whole program"
we avoid, and it moves the canonical order out of code (against defaults-as-code). Feature flags
keep the **order and canonical shape in code** and let config toggle **known optional steps**
on/off with their parameters — which is exactly the scientific use case: *measure the effect of
enabling one step*, not re-wire the pipeline.

## Why deferred

Until the set of composable optional steps is larger (rocking-curve integration, mosaicity,
alternative losses / refinement approaches), the code-level `pipeline([...])` composition is clearer
and the config surface would be speculative. Revisit when there are enough toggleable steps to
warrant the flag surface; the shape above is the agreed target. Until then, reproducibility rests on
the versioned config parameters plus the code version (and the run manifest recording what ran).

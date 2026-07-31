# `preprocess/steps/` — the composable `Plan → Plan` units

This package holds the **steps**: each is a pure `Plan → Plan` transform (a scikit-learn-style
transformer over the shared [`Plan`](../plan.py) value). The parent package holds the **spine** (the
value + combinators + setup) and the **orchestrators** (pipeline composition + the convergence
driver). Steps are the swappable units you compose *in or out* to run an experiment; the parent
decides *how* they are composed.

## The pipeline

```
from_experiment ──▶ Plan ──▶ [ steps… ] ──▶ Plan ──▶ refine
   (spine)                     (this dir)             (terminal Plan → Result)
```

`from_experiment` (spine) seeds the initial complete `Plan`; each step returns a sharpened `Plan`;
`refine` consumes the final one. Convergence is **entirely optional by construction** — a full run
needs no `converge_*`/`cover_*` step at all.

## The steps

| Module | Public step(s) | What it fits / does |
|---|---|---|
| `beams.py` | `select_beams` | prune to the active Klar-window beam set (`klar_beam_mask`) |
| `optimize_orientation.py` | `optimize_orientation` | refine each rotation's crystal orientation to the data |
| `optimize_thickness.py` | `optimize_thickness` | fit the sample thickness to the data |
| `rocking_curve.py` | `integrate_rocking_curve` | bake rocking-curve tilt integration into the geometry |
| `convergence.py` | `converge_beams` / `converge_pool` / `converge_sampling` | grow a numerics knob until consecutive *simulations* stop moving (self-stability) |
| `coverage.py` | `cover_beams` / `cover_pool` | grow a beam knob until it stops recovering new *observed* reflections (match-count) |

`convergence.py` and `coverage.py` also hold the parameter-agnostic drivers their adapters use
(`converge_scalar`, `maximize_scalar`) and the measures (`simulation_rfactor` / `plan_coverage`).

## Layering (why there are no import cycles)

```
spine   (plan, pipeline, experiment, scoring, orientation)   ← imported by steps
steps   (this dir)                                           ← imported by orchestrators
orchestrators (pipeline composition, driver)
```

A step imports **down** into the spine (the `PlanStep` type, `RefinementSetup`, `build_engine`,
`seed_beam_hkl`) and sideways to sibling steps; nothing in the spine imports a step. The parent
`driver.py` / pipeline composition import **up** from steps. The cross-lever convergence driver
lives at the parent level, not here, because it threads scalar state *across* steps — a step owns
no state beyond its `Plan → Plan` contract.

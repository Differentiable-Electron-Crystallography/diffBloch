# Decision: the cross-lever convergence fixpoint lives in the driver, not in `pipeline`

**Status:** accepted (stage 11, pool lever landed `10fd0f9`).
**Generalised by:** `plan-composition-shapes.md` promotes the driver used here to a named pattern
(the three composition shapes; the driver as a hand-rolled `State`/`StateT`). This ADR is the worked
convergence instance of that pattern.
**Context:** beam-set convergence has two coupled levers — the Klar *window*
(`integration_semiangle`, `converge_beams`) and the candidate *pool* (`g_max_refine`,
`converge_pool`). `stage11-convergence.md` sketched the joint fixpoint as block coordinate descent
composed with the project's own combinators: `iterate_until(pipeline([window, pool]), until=...)`.
Landing `converge_pool` proved that naive composition is **wrong**. This decision records why, and
where the joint fixpoint belongs instead.
**Reference:** `stage11-convergence.md` (the two levers, the `converge_scalar` HOF, `iterate_until`);
`plan-shape-and-step-ordering.md` (what a `Plan` carries); the private
`programs/convergence_testing.py` block-descent loop.

## The finding: `iterate_until(pipeline([converge_beams, converge_pool]))` does not compose

Two independent obstructions surfaced when the pool lever landed, both structural rather than
tuning:

1. **Seed/pruned mismatch.** `converge_beams` re-selects the window *from an unpruned seed* each
   step — it must, or widening the window could never recover a beam a narrower window clipped
   (`converge_beams` treats its input as the candidate pool). But `converge_pool` *emits a
   window-pruned* `Plan` (active set = `seed(g_max_refine) ∩ window(selection)`). Feed a pruned
   plan into `converge_beams` and it treats the already-clipped set as its seed, so widening the
   window recovers nothing. The two steps disagree on what "the input Plan" means (candidate pool
   vs. active set).

2. **Shared scalar state the `Plan` does not carry.** The two levers are coupled through two
   scalars — the window `integration_semiangle` and the pool `g_max_refine`. Block coordinate
   descent holds one fixed while it converges the other, then swaps. But a `Plan` stores neither
   scalar (it carries the grid + per-orientation active beam sets, not the numerics knobs that
   produced them). A fixed-spec `pipeline([converge_beams(sel), converge_pool(sel)])` therefore
   bakes a *stale* value of each scalar into the other lever: the pool step keeps applying the
   starting window while the window step has already moved on, and vice versa.

Neither is fixable by re-ordering or re-tuning `pipeline`/`iterate_until`. Both stem from the same
root: **the coordinate-descent state (the unpruned candidate pool + the two live scalars) is not the
`Plan`.**

## Decision

The individual levers (`converge_beams`, `converge_pool`) ship as **standalone** `Plan -> Plan`
steps, each self-contained and independently testable. The **cross-lever fixpoint is the preprocess
driver's responsibility**, not a `pipeline` composition. The driver — the same later slice that owns
the operation discriminated union and the default pipeline assembly — holds the coordinate-descent
state explicitly:

- the **unpruned candidate pool** (so each lever re-selects from a stable seed, honouring
  obstruction 1);
- the **two live scalars** `integration_semiangle` and `g_max_refine` (so each lever is
  reconstructed with the *other* lever's just-settled value, honouring obstruction 2);

and runs the private's fixed multi-pass coordinate sweep over them: converge one lever, read its
settled scalar, feed it to the other lever's spec, and repeat for a fixed `num_passes` (default 2)
with the private's per-pass order-swap. The
`simulation_rfactor` / `simulation_converged` check still referees stability, and the grid-`g_max`
dependent-sizing partial order (`stage11-convergence.md`, `KNOWN_ISSUES.md`) is enforced there too.

## Why not carry the scalars on the `Plan`?

Considered and rejected. Storing `integration_semiangle` / `g_max_refine` (and an unpruned-pool
handle) on every `Plan` would let the levers self-thread — but it pollutes the `Plan`, which is a
*geometry + active-beams* value consumed by the forward model and every downstream fit, with
transient convergence-loop bookkeeping that only the driver cares about. That inverts
`plan-shape-and-step-ordering.md` (the `Plan` carries what the physics needs, nothing more) and
would leak convergence state into `fit_orientation` / `fit_thickness` / scoring, which have no
business reading it. The state is loop-local; it belongs to the loop's owner (the driver), not to
the value the loop transforms.

## Consequence for sequencing

The "pool lever + cross-lever fixpoint" item in `stage11-convergence.md` splits: the **pool lever**
is done (`converge_pool`, `10fd0f9`); the **cross-lever fixpoint** moves into the driver slice
(alongside the operation discriminated union and the default-pipeline assembly). Until the driver
lands, the two levers are used individually or under a hand-written descent; there is no supported
`pipeline([window, pool])` fixpoint, by design.

# Decision: the train/validation split is by rotation and mostly dormant — cross-validate reflections

**Status:** accepted (stage 11).
**Context:** `from_experiment` splits the rotations into a `train` / `validation` `Plan` pair
(config `DataSplitConfig`: `train = all_except_validation`, `validation = every_10th_rotation`), and
`PlanSplit` documents that "`refine` fits on `train` while scoring the held-out `validation`". The
executable anchor (plan C) needs to evaluate *all* rotations to match the private reference, which
prompted the question: does a whole-rotation train/validation split actually make sense here?
**Reference:** `diffBloch_private` `rotation_dataset.py` (`cfg.dataloader.val_prop`; a
`train_test_split` over rotations, used only when `val_prop < 1.0`; the quartz e2e config uses
`val_prop = 1.0`, i.e. *no* holdout). Cross-validation reference point: the free R-factor,
Brünger (1992), Nature 355, 472-475 (`REFERENCES.md`).

## What is actually being fit

Refinement optimizes a **single global structural model** — for quartz, 2 ASU atoms, ~a dozen
refinable numbers (positions + ADPs) — against **thousands of reflections over 99 orientations**.
Each rotation is a different *view* of the *same* crystal, not an independent draw from a
distribution. The fit is over-determined by orders of magnitude.

## Why a whole-rotation split is a weak cross-validation guard for this

- **The ML rationale does not transfer.** Train/validation splits estimate the generalization of a
  *high-capacity* learner that can memorize its training set. A handful of global physical
  parameters has no such capacity; there is essentially nothing to overfit in the ML sense.
- **The crystallographic standard holds out reflections, not orientations.** The free R-factor
  (R_free, Brünger 1992) reserves a random subset of *reflections* (typically 5-10%), refines on the
  rest, and monitors R_free as an unbiased check on model bias / over-parameterization / restraint
  mis-weighting. It is finer-grained and every orientation still contributes.
- **Whole-rotation holdout is lumpy and throws away signal.** Removing a rotation drops all its
  reflections at once (high-variance, blocky cross-validation) and discards that orientation's
  unique dynamical/geometric information — yet multi-view orientation coverage is exactly what
  constrains the structure.
- **A "held-out" rotation is not truly held out.** The per-rotation nuisances (orientation,
  thickness) are fit on *every* rotation during preprocess, including the validation ones, so the
  split does not isolate structural generalization.
- **The private did not hold out here.** The quartz experiment ran `val_prop = 1.0` (everything is
  validation), and the reference `R_obs` is aggregated over all 99 rotations.

## When a rotation split *does* make sense

When a **high-capacity learned component** that can overfit per-rotation is present — the private's
`ThicknessNN` (a learned `theta -> thickness` map) or learned structure factors. A learned
per-orientation function *can* memorize per-rotation quirks, so holding out whole rotations then
tests whether it generalizes to unseen orientations. That is the future learned-thickness / learned
mode (ROADMAP stage 11 thickness modes); the rotation split earns its keep there.

## Decision

- **Inference and physics refinement evaluate over all rotations** (`PlanSplit.combined`). The
  executable anchor and `run_inference` use `combined`; the whole-rotation split is not treated as a
  meaningful cross-validation guard for pure physics refinement.
- **Keep the split machinery, dormant and forward-compatible**, for the learned modes where a
  rotation-level holdout is genuinely informative. It costs nothing to construct and gives those
  modes a ready seam.
- **If a cross-validation metric is wanted for physics refinement, adopt R_free-style reflection
  holdout** (a random per-reflection mask across all orientations), not rotation holdout — finer,
  preserves orientation coverage, and matches crystallographic practice.

## Design smell surfaced (rule 3)

The split is currently **constructed but unconsumed**: nothing downstream distinguishes `train` from
`validation` (both `refine` and `run_inference` take a single `Plan`), yet `PlanSplit` is documented
as if the split were live ("`refine` fits on `train` while scoring the held-out `validation`"). Two
honest resolutions, to pick up when refinement wiring lands: either wire the split to its real use
(the learned modes, and/or an R_free-style reflection holdout for physics refinement), or stop
advertising a behaviour that is not implemented. Recorded here rather than silently fixed, since it
crosses the config / preprocess / engine layers.

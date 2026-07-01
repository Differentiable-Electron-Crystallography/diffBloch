# Decision (planned): explicit rotation exclusion, replacing the private's implicit index-commenting

**Status:** planned (stage 11+), not yet implemented.
**Context:** some rotations should be dropped from an experiment -- bad frames, misindexed
orientations, outliers, or a genuinely non-converging orientation search. In `diffBloch_private`
this is done *implicitly*: `cfg.dataloader.ignore_orientations` is a list of rotation indices
excluded from the training set, and researchers hand-edit it (the quartz config carries a large
commented-out list, the fossil of that manual workflow). We want the same capability, but
**explicit, reproducible, and justified**.
**Reference:** `diffBloch_private` `rotation_dataset.py` (~line 483):
`train_indices = [i for i in range(len(dataset)) if i not in cfg.dataloader.ignore_orientations]`,
and `tests/e2e/inference/quartz/config.yml` (`ignore_orientations: []  #[1,2,3,...]`).

## Decision

Add an **explicit exclusion list to the experiment config** (e.g. `observation.exclude_rotations` or
`refinement.exclude_rotations`: a list of rotation / zone-axis indices), which `from_experiment`
honours by omitting those `OrientationPlan`s from the constructed `Plan`(s). Consequences:

- **Reproducible.** The exclusion lives in the versioned `experiment.yaml` and is covered by
  `experiment.lock` -- the run is reproduced by replaying the config, never by editing code or
  commenting out lines. (Contrast the private's edit-a-Python/config-list workflow.)
- **Explicit and attributable.** A reader sees exactly which rotations were dropped. Excluding a
  rotation must be *justified* (a bad-frame / misindex reason), not a silent lever to lower `R` --
  ideally the config carries a short reason per excluded index, and reporting states how many were
  excluded and why.
- **Composable / orthogonal.** Exclusion is a data-curation concern, independent of the fit
  pipeline; applying it at `from_experiment` keeps it out of `select_beams` / `fit_orientation` /
  `fit_thickness`. It composes with the train/validation split
  (`design/decisions/train-validation-split.md`).

## What this is *not* (guardrails)

- **Not exclusion that optimizes the fit metric.** The line that must never be crossed: you may not
  choose the excluded set to minimize the quantity you report (aggregate `R`, wR2, ...). Excluding
  whatever rotation raises `R` always lowers `R` by discarding hard data -- data-dredging that
  inflates apparent quality. Any exclusion criterion must be *independent* of the objective (see the
  automatic mechanism below).
- **Not auto-exclusion of non-convergers.** Auto-dropping any orientation whose `fit_orientation`
  search hits `max_iterations` is dangerous: a non-converging search is a *signal* (a bad seed
  orientation, a beam-selection issue, a landscape problem) -- not an outlier -- and silently
  dropping it hides the problem. Search non-convergence is *report-only*. (The quartz non-convergers
  were a
  *false* signal from an under-set cap, fixed by calibrating `max_iterations` to 600 -- see that
  commit / `KNOWN_ISSUES.md`. The private does not exclude them, and neither do we.)
- **Not a fit knob.** Exclusion changes *which data* the model sees, not *how* it is fit. It must be
  set before and independently of any R-factor it would move.

## Automatic exclusion as a robust-outlier fixpoint (planned)

Exclusion can be made *automatic* without crossing the line above -- and it fits the project's
existing convergence family. It is **not** a `converge_scalar` (that is a monotone scalar sweep);
it is a **fixpoint over a discrete set**, which is exactly `iterate_until` (`Plan -> Plan` to a
fixpoint). The step is *flag statistical outliers, drop them, refit*; the fixpoint is *the excluded
set stops changing* (iterative trimmed / robust estimation):

    iterate_until(excluded_set_stable, pipeline([flag_outliers_and_exclude, refit]))

The one non-negotiable design constraint is the **criterion**:

- **Independent robustness, not the objective.** A rotation is flagged when its per-rotation
  residual is a *statistical* outlier versus the cohort -- e.g. `> k * MAD` from the median
  per-rotation `R`, or a Grubbs test -- *not* when dropping it lowers the aggregate `R`. The measure
  is a robustness statistic of the residual *distribution*, orthogonal to minimizing that residual.
- **Conservative and report-first.** Default to *reporting* flagged candidates for a human to
  confirm; automatic exclusion, if enabled, uses a conservative threshold and still records what it
  did. Guard against removing too much (a trimming fraction cap).
- **Recorded and reproducible.** The resolved excluded set (plus the criterion and the values that
  triggered it) is written back into the versioned config / `experiment.lock`, so the run reproduces
  by replay -- an automatic decision becomes an explicit, auditable artifact, not a hidden one.
- **Validated against the split.** Ideally exclusion decisions are stable under, or improve, the
  held-out metric (`design/decisions/train-validation-split.md`) -- so the exclusion itself is not
  overfit.

## Status / sequencing

Deferred: the executable quartz anchor needs **no** exclusions (all 99 rotations are kept; the
`max_iterations` calibration removed the only reason a rotation failed). This records the approach
so that when a dataset genuinely needs curation, the mechanism is explicit config, not a code edit.
See `ROADMAP.md`.

# Observability and loggers

diffBloch reports what it's doing as a stream of typed events -- a rotation's orientation score, a
preprocessing stage's beam counts, a refinement epoch's wR2 -- rather than ad hoc print statements.
Every backend (console, TUI, CSV, W&B, Comet) consumes the same stream, so adding a new sink never
requires touching the core.

## The event/logger pattern

An event is anything with a `channel` (its stable name), `measurements` (metric name -> value), and
`step` (its position on a run's x-axis, or `None` for a run-level fact). A logger is any object with
one method:

```python
def report(self, event: Event) -> None: ...
```

The core hands events to whatever logger it was given and knows nothing about backends; the default
is `NULL_LOGGER`, which discards everything, so a run with no logger attached costs nothing. Writing
a new backend is exactly one method -- see `diffBloch.app.loggers` for the built-in ones.

## What gets reported

### Preprocessing

A fresh preprocess run reports the plan's shape as the recipe reshapes it: one `PlanSeeded` for the
incoming plan, then one `PlanStepCompleted` per step (`select_beams`, `optimize_orientation`, ...),
each carrying the orientation count, structure-factor support size, solve-beam totals, and
observed/matched reflection counts. Read consecutively these are per-stage *survival* counts -- how
many beams and reflections each filter left behind -- which is why the seed baseline (`PlanSeeded`)
is reported at all: without it, the first stage's numbers are absolutes with nothing to compare
against. Reusing a checkpoint skips this runner entirely, so these only fire on a fresh run.

`n_matched_hkl` is absent rather than zero before `build_orientation_plans` runs, since a candidate
plan has no alignment yet and a reported `0` would misleadingly mean "matched nothing".

### Orientation and thickness optimization

`OrientationOptimizationStarted`/`ThicknessOptimizationStarted` report the rotation count once,
before either search begins. Each rotation then reports one `OrientationOptimized`/
`ThicknessOptimized` as it finishes, carrying that rotation's index, its residual, and (for
thickness) the full candidate-thickness/score grid the search evaluated -- not just the winner, so
a plot backend (see `ThicknessPlotLogger` below) can show the whole curve. An
`OrientationOptimizationSummary` closes out the stage with the mean score and deduplicated matched
HKL count across every rotation.

### Refinement

Before the first step, a refinement run declares its whole objective once: `ObjectiveManifest`
names every composed penalty with its weight, every hard constraint, and every model component --
so what's active is stated up front rather than inferred from the per-epoch stream. It prints
`none` for an empty category, because the default CLI path composes no penalties and that is a fact
worth reading, not an omission.

Each epoch then reports one `RefinementStep`: the epoch-mean wR2, R_obs, and diffraction loss, plus
every objective term the run actually composed as `{term}/raw`, `{term}/weight`, and
`{term}/contribution` -- so a weighted restraint's scientific value stays legible next to the
number the optimizer actually minimises. A restraint that was *not* composed into the objective
produces no measurement at all; that absence is what distinguishes an inactive term from a
satisfied one.

Each epoch mean carries its own denominator, e.g. `wR2 0.050000 [97/99]`: the mean covers only the
rotations that produced a finite score, and wR2/R_obs are filtered independently, so a run that
quietly evaluates fewer rotations cannot pass as a run that got better.

`RefinementCompleted` closes the run: the best epoch, which objective selected it
(`best_training_loss` vs `best_validation_loss` -- see [Refinement](refinement.md#refinement-outputs)),
and `RefinedRotationMetrics` gives the settled per-rotation wR2/R_obs/reflection counts for the
final report.

## Backends

| Backend | Where | What it does |
|---|---|---|
| `ConsoleLogger` | `diffBloch.app.loggers` | Bridges every event onto stdlib `logging` as a scrolling log. The CLI's default sink. |
| `TuiLogger` | `diffBloch.app.loggers.tui` (`diffBloch[tui]` extra, `--tui`) | Repaints the same event stream as a live terminal dashboard: experiment/objective panels, a per-phase progress bar, the epoch table with its denominators, the settled per-rotation table. |
| `CSVLogger` | `diffBloch.app.loggers` (`--csv PATH`) | Appends every event's measurements as long-format rows (`channel, step, metric, value`) -- one flat, tailable table for a heterogeneous stream, ready to filter by channel or pivot by step. |
| `WandbLogger` | `diffBloch.app.loggers.wandb` | Logs each event's measurements to Weights & Biases as `{channel}/{metric}` series. |
| `CometLogger` | `diffBloch.app.loggers.comet` | Logs each event's measurements to a Comet ML experiment. |
| `ThicknessPlotLogger` | `diffBloch.app.loggers.plotting` (`--plot-thickness`) | Saves one residual-vs-thickness PNG per rotation from `ThicknessOptimized`'s full candidate grid; a no-op on every other event. |
| `MultiLogger` | `diffBloch.observability` | Fans one event stream out to several loggers at once (e.g. console and W&B together). |

`ConsoleLogger` and `TuiLogger` are alternatives, never both at once -- a live terminal display owns
the terminal, so composing them would interleave garbled output. File and vendor sinks (`CSVLogger`,
`WandbLogger`, `CometLogger`, `ThicknessPlotLogger`) compose freely with either, and with each
other, via `MultiLogger`.

## Writing your own backend

```python
from diffBloch.observability import Event

class PrintLogger:
    def report(self, event: Event) -> None:
        print(f"{event.channel}[{event.step}] {dict(event.measurements)}")
```

Pass an instance wherever a `logger:` parameter is accepted (the preprocessing/refinement drivers,
`from_experiment`, the CLI's programmatic entry points), or compose it with the built-in backends
via `MultiLogger((ConsoleLogger(), PrintLogger()))`.

See [`diffBloch.observability`](api/observability.md) for the full event list and every event's
exact fields.

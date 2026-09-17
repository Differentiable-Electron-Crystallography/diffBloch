# diffBloch event report

A visualizer consumer, not part of the diffBloch core library. It reads the canonical JSONL reports
produced by CLI commands like `diffbloch refine`.

| module | role |
| --- | --- |
| `reader.py` | find, parse, and slice a report; `events_of` rebuilds records as the library's event dataclasses |
| `style.py` | the shared palette and chrome; colour assigned by job, applied per figure |
| `tables.py` | run-summary tables over the parsed records, rendered to Markdown |
| `figures.py` | matplotlib figures over the parsed records, grouped into stage sections |
| `event_report.ipynb` | interactive viewer; a thin driver over `reader` + `tables` + `figures` |

Table and plotting logic live in `tables.py` / `figures.py` rather than in notebook cells so they can
be imported and tested (`tests/unit/test_event_report_tool.py`). The notebook holds no rendering
logic of its own, and its outputs are not committed.

The notebook is the only rendering surface. The JSONL report itself is the machine-readable
artifact — anything wanting a different presentation reads the report directly rather than going
through a renderer here.

## The contract

A report line is an `EventRecord` envelope around one event from `diffBloch.observability`, and
`series | payload` is exactly that event's fields. Readers here never index the payload dict:
`reader.events_of(records, OrientationOptimized)` hands back real `OrientationOptimized` objects
via `diffBloch.observability.event_from_record`, so field names are checked by the type checker,
tuples come back as tuples, and JSON's `"NaN"` comes back as a float. A report that no longer fits
the events this checkout defines — a required field renamed or removed, an unknown event type —
raises `ReportSchemaError` naming the record and field, rather than a figure quietly going blank.
Fields *added* by a newer writer are ignored, so an older reader survives additive changes.

`diffBloch.observability.EVENT_TYPES` is the registry: every event class, keyed by the name
`event_type` records. Defining an event registers it.

The contract is pinned to disk as `tests/fixtures/reports/golden-v1.jsonl`, a small run carrying
every event type; `tests/unit/test_golden_report.py` reads it with the current code and renders
every table and figure from it. After a deliberate schema change, regenerate it with
`uv run python tests/fixtures/reports/build_golden.py` and commit the diff — that diff is the
review of the change.

## Example workflow

Install the project with dev/tooling dependencies:

```bash
uv sync --dev
```

Run a fresh refinement of the smallest bundled example:

```bash
EXPERIMENT=examples/Colmey_et_al_2026/data/quartz-no-abs

uv run diffbloch refine "$EXPERIMENT" --refresh
REPORT=$(ls -t "$EXPERIMENT"/reproducibility/reports/report-*.jsonl | head -n 1)
```

The report path is also printed in the command's output. The `REPORT=$(...)` line just captures the
newest timestamped report for the commands below. `--refresh` matters for what the report can show:
without it a valid checkpoint is reused, no orientation or thickness search runs, and the report
carries no events for those figures — the preprocess sections are simply absent. The same pattern
works for preprocess/infer/converge:

```bash
uv run diffbloch infer "$EXPERIMENT"
REPORT=$(ls -t "$EXPERIMENT"/reproducibility/reports/report-*.jsonl | head -n 1)
```

A command that *fails* still leaves its report, under `report-<stamp>-failed.jsonl` — the stage
events in it are how you see where the run stopped.

## The notebook

`event_report.ipynb` has three cells: setup, render, export. It takes one input, the path of the
report to render, and does nothing else.

**Choosing the report.** The setup cell sets `REPORT`, in this order of precedence:

1. edit the `REPORT = ...` line to any path (absolute, or relative to the working directory or the
   checkout root — both are tried);
2. otherwise the `DIFFBLOCH_EVENT_LOG` environment variable, if set when Jupyter was launched;
3. otherwise the newest `report-*.jsonl` under `./reproducibility/` or any bundled example.

The cell prints the path it settled on. To look at a different report, change `REPORT` and re-run
the render cell.

```bash
DIFFBLOCH_EVENT_LOG="$REPORT" \
  uv run jupyter lab tools/event_report/event_report.ipynb
```

**Rendering.** The render cell reads the report, prints how many records, tables and figures it
found, then shows the summary tables followed by the figures under one heading per stage, in the
order the run produced them. A report from a run that never entered a stage (an `infer` run, a
checkpoint-reuse `refine`) has fewer sections, not empty ones. A report written by an older
checkout whose events no longer fit the current ones stops here with a `ReportSchemaError` naming
the record and field — see *The contract* above.

**Exporting.** The export cell is opt-in: set `EXPORT_FIGURES = True` and re-run it to write every
figure to `EXPORT_DIR` (default `event_report_figures/` beside the notebook) in each of
`EXPORT_FORMATS` (default SVG, which stays vector; add `"png"` for a raster copy at 300 dpi). The
file names are the figure names in the table below. This is the only place the tool writes images.

**Headless.** The notebook needs no interaction, so it also runs unattended — a SLURM post-step, or
a quick check that a report renders:

```bash
DIFFBLOCH_EVENT_LOG="$REPORT" MPLBACKEND=Agg \
  uv run jupyter nbconvert --to html --execute tools/event_report/event_report.ipynb \
  --output-dir "$EXPERIMENT/reproducibility/reports"
```

That leaves a self-contained `event_report.html` beside the report. Executed notebooks and their
outputs are not committed.

## Tables

`tables.build_tables(records)` returns `(title, markdown)` pairs, shown above the figures. Like the
figures, a table whose events are absent is dropped rather than shown empty.

| table | from | shows |
| --- | --- | --- |
| Preprocess | `PreprocessCompleted` | rotation/stage/HKL counts, and whether orientation and thickness optimization ran with their own settings (`search.*`, `grid.*`) |
| Refinement summary | `ExperimentDeclared`, `RefinementStep`, `RefinementCompleted` | the selected epoch, the objective it was selected on, optimizer settings, train (and validation) wR2/R_obs means with their `[evaluated/total]` counts, diffraction loss, matched HKLs (I>3σ/total) |

Every mean carries the rotation count it was taken over, because a mean over fewer rotations is a
different quantity rather than a better one. A report written before `PreprocessCompleted` recorded
its `steps` says the stage settings were *not recorded*, never that the stages did not run.

## Figures

`figures.build_sections(records)` returns the figures grouped under stage headings — the notebook
renders one heading per section — and `figures.build_figures(records)` is the flat view that
`figures.export_figures(...)` names files from. Section *order* is derived from the report rather
than hardcoded, so a run configured `preprocess.stage_order: thickness_first` heads its thickness
section before its orientation one. A section whose figures all declined to render is dropped, not
shown empty.

| figure | from | shows |
| --- | --- | --- |
| `convergence_sweeps` | `converge` | one panel per control, R-factor *between consecutive settings* against candidate value, with the pass threshold as a rule and the crossing marked |
| `orientation_optimization` | `preprocess` | seed vs fitted score per rotation, with the fitted goniometer deltas |
| `orientation_search_trace` | `preprocess` | the scored Nelder-Mead path of the longest search |
| `thickness_grids` | `preprocess` | every rotation's thickness-vs-score curve overlaid, selected thickness marked — shows the *shape* of each minimum |
| `thickness_heatmap` | `preprocess` | the same grids as rotation × thickness → score, one panel per dataset, fitted thickness traced — shows whether the fit drifts smoothly with tilt or jumps between minima |
| `coupling_geometry`, `coupling_segment_heatmap` | `preprocess` | coupled-solve shape per rotation and per segment |
| `epoch_curve` | `refine` | train/validation wR2 and R_obs per epoch |
| `refined_rotation_scores` | `refine` | final per-rotation scores, held-out rotations marked |
| `per_dataset_summary` | `refine` | mean final wR2/R_obs per dataset, train and validation rotations split, with their counts (pooled runs only) |
| `thickness_model` | `refine` | the learned `ApparentThicknessNN` curve per dataset |

**The two thickness figures are different quantities.** `thickness_grids` / `thickness_heatmap` are
the *preprocess* stage's per-rotation grid search — one fitted scalar per rotation, picked by argmin
over a `linspace`. `thickness_model` is the *refinement* stage's `ApparentThicknessNN` — a trained
function of tilt angle, one per dataset, evaluated after the loop. They carry different names and
sit under separate headings for that reason.

The two coupling figures are emitted during **preprocess**, not refinement, even though what they
describe is the geometry the refinement loop repeats every step. They fire on checkpoint-reuse runs
and on `preprocess` / `infer` too, which never enter a refine stage at all.

Note that `convergence_sweeps` plots a *difference*, not a quality: a `ConvergenceTrial` simulates
at `previous` and again at `candidate` and reports the R-factor between the two, so the question it
answers is "has the answer stopped changing?" A curve that never crosses the threshold rule is a
sweep that ran out of range rather than one that converged.

The JSONL report is the durable contract. Runtime diffBloch code emits structured events and app
loggers persist only their declared output; report tools decide how to render them. Image export
belongs here, never in the core library or the runtime loggers.

## Style

`style.py` holds one palette for every figure, assigned by the job the colour does:

- **Categorical** (identity — a series, a dataset, a pass) draws from a fixed slot order that
  passes the adjacent-pair colourblind gate (worst OKLab ΔE 9.1, target ≥ 8) and the normal-vision
  floor (worst ΔE 22.9, floor 15). Slots are assigned by position and never cycled — matplotlib's
  default would paint a hundred single-population curves in ten hues and imply a grouping that
  isn't in the data.
- **Sequential** (continuous magnitude — the heatmaps) is `viridis`: monotonic in lightness by
  construction, so it survives greyscale printing and all three dichromacies. A ramp interpolated
  through arbitrary hex steps is not perceptually uniform and injects structure the data doesn't
  have. [Crameri's scientific colour maps](https://www.fabiocrameri.ch/colourmaps/) are the other
  defensible family if these figures ever go into a manuscript that cites them.

Chrome is recessive: solid hairline gridlines, muted axis ink, thin marks, no top/right spines.
Dashes mean *threshold* and nothing else. Type stays at or above 7pt — the figure-text floor Nature
and Science set — and `savefig.dpi` is 300, so an exported panel is legible at print scale. SVG
export stays vector regardless.

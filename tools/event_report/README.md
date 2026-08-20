# diffBloch event report

A visualizer consumer, not part of the diffBloch core library. It reads the canonical JSONL reports
produced by CLI commands like `diffbloch refine`.

| module | role |
| --- | --- |
| `reader.py` | find, parse, and slice a report |
| `style.py` | the shared palette and chrome; colour assigned by job, applied per figure |
| `figures.py` | matplotlib figures over the parsed records, grouped into stage sections |
| `event_report.ipynb` | interactive viewer; a thin driver over `reader` + `figures` |

Plotting logic lives in `figures.py` rather than in notebook cells so it can be imported and tested
(`tests/unit/test_event_report_tool.py`). The notebook holds no rendering logic of its own, and its
outputs are not committed.

The notebook is the only rendering surface. The JSONL report itself is the machine-readable
artifact — anything wanting tables or a different presentation reads the report directly rather
than going through a renderer here.

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
newest timestamped report for the commands below. The same pattern works for
preprocess/infer/converge:

```bash
uv run diffbloch infer "$EXPERIMENT"
REPORT=$(ls -t "$EXPERIMENT"/reproducibility/reports/report-*.jsonl | head -n 1)
```

A command that *fails* still leaves its report, under `report-<stamp>-failed.jsonl` — the stage
events in it are how you see where the run stopped.

Open the notebook with that JSONL preselected:

```bash
DIFFBLOCH_EVENT_LOG="$REPORT" \
  uv run jupyter lab tools/event_report/event_report.ipynb
```

Inside the notebook you can edit the JSONL path field or, when `ipywidgets` is available, use the
upload control that accepts a picked or dragged `.jsonl` file.

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
| `per_dataset_summary` | `refine` | mean final scores per dataset (pooled runs only) |
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

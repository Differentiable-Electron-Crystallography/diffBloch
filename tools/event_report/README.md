# diffBloch event report

This top-level tool is a visualizer consumer, not part of the diffBloch core library. It reads the
canonical JSONL reports produced by CLI commands like `diffbloch refine` and renders a lightweight
HTML report.

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
newest timestamped report for the commands below.

The same pattern works for preprocess/infer/converge:

```bash
uv run diffbloch infer "$EXPERIMENT"
REPORT=$(ls -t "$EXPERIMENT"/reproducibility/reports/report-*.jsonl | head -n 1)
```

Render a static HTML view:

```bash
uv run python tools/event_report/event_report.py \
  "$REPORT" \
  --output "$EXPERIMENT/reproducibility/event_report.html"
```

Open the interactive notebook with that same JSONL preselected:

```bash
DIFFBLOCH_EVENT_LOG="$REPORT" \
  uv run jupyter lab tools/event_report/event_report.ipynb
```

Inside the notebook you can also edit the JSONL path text field. When `ipywidgets` is available, the
notebook shows an upload control that accepts a picked or dragged `.jsonl` file; otherwise the path
field / `DIFFBLOCH_EVENT_LOG` workflow is the supported route.

The notebook imports matplotlib, renders figures from the event payloads, and includes an explicit
`export_figures(...)` helper guarded by `EXPORT_FIGURES = False`. Figures are rendered when the
JSONL contains the matching events: epoch curves, orientation before/after, orientation search
traces, final per-rotation scores, per-dataset summaries, thickness grids, thickness profiles,
coupling geometry, and coupling segment heatmaps.

The JSONL report is the durable contract. Runtime diffBloch code emits structured events and app
loggers persist only their declared output; report tools decide how to render them. Visualization
PNG/SVG export belongs here in `tools/event_report`, never in the core library or runtime loggers.

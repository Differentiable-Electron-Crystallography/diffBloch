# Observability

Domain observations are typed events with pluggable logger sinks. The pure core emits event values;
a `Logger` attached at the app boundary interprets them (the null default discards them). Solver
diagnostics ride stdlib `logging` instead — two channels, two mechanisms.

`diffbloch refine` writes the canonical versioned JSONL report by default after the command
completes successfully:

```bash
uv run diffbloch refine <experiment_dir>
```

The file is `<experiment_dir>/reproducibility/reports/report-YYYYMMDDTHHMMSSZ.jsonl`. CLI runs stream to a
temporary report while work is in flight and promote it to that path only after the requested stage
completes; failed CLI runs rely on the console observation stream and do not leave a canonical
`report-*.jsonl` artifact.
Each line is an `EventRecord`: one maximal schema shared by every event, with run identity,
sequence, timestamp, channel, step, optional dataset/rotation placement, scalar measurements,
numeric series arrays, artifact paths, and the full structured payload. Visualizers consume that
record stream without importing runtime logger backends or relying on live WebSocket listeners.
The `ExperimentDeclared` payload includes the experiment directory plus the structure CIF and
`.cif_pets` input refs.

App workflow sections are declared explicitly. `RunStageStarted` / `RunStageStopped` bracket
`converge`, `preprocess`, `infer`, and `refine` stages, including elapsed time and stop status.
Report tools should key sections from those lifecycle events rather than inferring stage boundaries
from result-event names.

The report stream includes enough structured data for the top-level `tools/event_report/` consumer
to render the old human-facing views without runtime plot writers: preprocess completion,
per-epoch train/validation wR2 and R_obs, orientation before/after scores with fitted angle deltas,
per-dataset final rotation metrics, and thickness score grids. Higher-cardinality visualization
data is batched into single events: `OrientationSearchTrace` carries the scored search path for one
rotation, and `RotationCouplingSegments` carries the per-segment tilt/beam geometry for one
rotation. Matplotlib rendering and optional figure export live in
`tools/event_report/event_report.ipynb`, not in `src/diffBloch`.

```{eval-rst}
.. automodule:: diffBloch.observability
```

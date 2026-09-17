# Observability

Domain observations are typed events with pluggable logger sinks. The pure core emits event values;
a `Logger` attached at the app boundary interprets them (the null default discards them). Solver
diagnostics ride stdlib `logging` instead — two channels, two mechanisms.

`diffbloch refine` writes the canonical versioned JSONL report by default:

```bash
uv run diffbloch refine <experiment_dir>
```

The file is `<experiment_dir>/reproducibility/reports/report-YYYYMMDDTHHMMSSZ.jsonl`. CLI runs stream
to a temporary report while work is in flight and promote it when the command stops. A run that
*failed* is promoted under `report-YYYYMMDDTHHMMSSZ-failed.jsonl` instead, so a completed report is
never confused with a partial one — but the partial one is kept, because its `RunStageStopped`
event is the structured record of where the run stopped. The promotion happens on the way out of a
context manager, so an interrupt or an uncaught error leaves neither a missing report nor a
lingering temporary file.

Each line is an `EventRecord`: one maximal schema shared by every event, with run identity,
sequence, timestamp, channel, step, optional dataset/rotation placement, scalar measurements,
numeric series arrays, artifact paths, and the remaining structured payload. Numeric arrays live in
`series` only and are *not* repeated in `payload`; `series | payload` reconstructs the emitting
dataclass exactly. Visualizers consume that record stream without importing runtime logger backends
or relying on live WebSocket listeners. The `ExperimentDeclared` payload includes the experiment
directory plus the structure CIF and `.cif_pets` input refs.

App workflow sections are declared explicitly. `RunStageStarted` / `RunStageStopped` bracket
`converge`, `preprocess`, `infer`, and `refine` stages, including elapsed time and stop status.
Report tools should key sections from those lifecycle events rather than inferring stage boundaries
from result-event names.

The report stream carries enough structured data to rebuild the old human-facing views without any
runtime plot writer: preprocess completion, per-epoch train/validation wR2 and R_obs, orientation
before/after scores with fitted angle deltas, per-dataset final rotation metrics, and thickness
score grids. `tools/event_report/` is the shipped consumer. Higher-cardinality visualization
data is batched into single events: `OrientationSearchTrace` carries the scored search path for one
rotation, and `RotationCouplingSegments` carries the per-segment tilt/beam geometry for one
rotation. Both store their columns in parallel evaluation order, so row position is the trial or
segment index and no `range(n)` column is written. Matplotlib rendering and optional figure export
live in `tools/event_report/figures.py`, not in `src/diffBloch`.

The consumer half of the contract lives here too. `EVENT_TYPES` is the registry of every event
class keyed by the name `event_type` records, and `event_from_record` rebuilds a record into the
event it was written from — tuples as tuples, JSON's `"NaN"` as the float, nested value dataclasses
re-materialized — so a reader works with real events rather than a payload dict whose keys it has
to know. A record that no longer fits (a required field renamed or removed, an unknown event type)
raises `ReportSchemaError` naming the record and field; a field *added* by a newer writer is
ignored, so an older reader survives additive changes. `RefinementOutputsWritten` carries its
paths relative to the `experiment_directory` it also names, so a report copied elsewhere still says
what was produced where.

Every event names a rotation the same way: `(dataset, rotation_index)`, where the index is the
rotation's position within its own dataset file. A multi-dataset run pools its datasets onto one
global index space for the engine, the train/validation split and the thickness networks, but the
pattern records the offset pooling added (`PatternBatch.pool_offset`) and every emitter reports
`pattern.dataset_rotation_index`, so the pooled index never reaches the report and consumers pair
preprocess and refinement events by key with no mapping between stages.

```{eval-rst}
.. automodule:: diffBloch.observability
```

"""Render a standalone HTML report from a diffBloch JSONL event log.

This script deliberately lives outside ``src/diffBloch``. It is a consumer of the event contract,
not part of the refinement library or its core numerical path. Figures belong in
``tools/event_report/event_report.ipynb``; this HTML renderer stays dependency-light and writes only
the requested report file.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import fmean

from diffBloch.observability import EventRecord


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event_log", type=Path, help="JSONL report written by ReportLogger")
    parser.add_argument("--output", type=Path, default=Path("event_report.html"))
    args = parser.parse_args()

    records = read_records(args.event_log)
    args.output.write_text(render_html(records))
    print(args.output)
    return 0


def read_records(path: Path) -> list[EventRecord]:
    return [
        EventRecord.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def render_html(records: list[EventRecord]) -> str:
    channels = Counter(record.channel for record in records)
    channel_rows = "\n".join(
        f"<tr><td>{_esc(channel)}</td><td>{count}</td></tr>"
        for channel, count in sorted(channels.items())
    )
    event_rows = "\n".join(_event_row(record) for record in records)
    embedded = json.dumps([record.model_dump(mode="json") for record in records])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>diffBloch Event Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1f2933; }}
    h1, h2, h3 {{ line-height: 1.2; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(12rem, 1fr)); gap: 0.75rem; }}
    .metric {{ border: 1px solid #d9e2ec; border-radius: 6px; padding: 0.65rem 0.75rem; }}
    .metric strong {{ display: block; color: #52606d; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; }}
    .metric span {{ display: block; font-size: 1.35rem; margin-top: 0.15rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; font-size: 0.92rem; }}
    th, td {{ border-bottom: 1px solid #d9e2ec; padding: 0.4rem; text-align: left; vertical-align: top; }}
    th {{ background: #f0f4f8; position: sticky; top: 0; }}
    code, pre {{ background: #f0f4f8; border-radius: 4px; padding: 0.15rem 0.25rem; }}
    pre {{ overflow-x: auto; padding: 0.75rem; }}
    input {{ width: 24rem; max-width: 100%; padding: 0.4rem; }}
    .empty {{ color: #7b8794; }}
  </style>
</head>
<body>
  <h1>diffBloch Event Report</h1>
  <p>{len(records)} event records from the canonical JSONL artifact.</p>

  <h2>Experiment</h2>
  {_experiment_summary(records)}

  <h2>Run Stages</h2>
  {_stage_summary(records)}

  <h2>Convergence</h2>
  {_stage_section(records, "converge", _convergence_section(records))}

  <h2>Preprocess</h2>
  {_stage_section(records, "preprocess", _preprocess_section(records))}

  <h2>Inference</h2>
  {_stage_section(records, "infer", _inference_section(records))}

  <h2>Refinement</h2>
  {_stage_section(records, "refine", _refinement_section(records))}

  <h2>Channels</h2>
  <table>
    <thead><tr><th>Channel</th><th>Events</th></tr></thead>
    <tbody>{channel_rows}</tbody>
  </table>

  <h2>Events</h2>
  <input id="filter" placeholder="Filter channel, type, dataset, metric...">
  <table id="report">
    <thead>
      <tr><th>#</th><th>Type</th><th>Channel</th><th>Step</th><th>Dataset</th><th>Rotation</th><th>Measurements</th><th>Series</th></tr>
    </thead>
    <tbody>{event_rows}</tbody>
  </table>

  <h2>Raw Event JSON</h2>
  <details><summary>Show embedded records</summary><pre id="raw"></pre></details>

  <script>
    const records = {embedded};
    document.getElementById("raw").textContent = JSON.stringify(records, null, 2);
    const filter = document.getElementById("filter");
    filter.addEventListener("input", () => {{
      const needle = filter.value.toLowerCase();
      for (const row of document.querySelectorAll("#report tbody tr")) {{
        row.style.display = row.textContent.toLowerCase().includes(needle) ? "" : "none";
      }}
    }});
  </script>
</body>
</html>
"""


def _experiment_summary(records: list[EventRecord]) -> str:
    declared = _last(records, "ExperimentDeclared")
    if declared is None:
        return '<p class="empty">No ExperimentDeclared event found.</p>'
    payload = declared.payload
    items = (
        ("Name", payload.get("name")),
        ("Directory", payload.get("experiment_directory")),
        ("Structure CIF", payload.get("structure")),
        ("Experimental Data", payload.get("experimental_data")),
    )
    return (
        '<div class="cards">'
        + "".join(_metric_card(label, value) for label, value in items)
        + "</div>"
    )


def _stage_summary(records: list[EventRecord]) -> str:
    starts = _of_type(records, "RunStageStarted")
    stops = _of_type(records, "RunStageStopped")
    if not starts and not stops:
        return '<p class="empty">No declared run-stage lifecycle events found.</p>'
    latest_stop = {
        str(record.payload.get("stage")): record for record in stops if record.payload.get("stage")
    }
    rows = []
    for record in starts:
        stage = str(record.payload.get("stage", ""))
        stop = latest_stop.get(stage)
        rows.append(
            _row(
                [
                    stage,
                    record.payload.get("experiment_directory"),
                    "" if stop is None else stop.payload.get("status"),
                    "" if stop is None else _fmt(stop.payload.get("elapsed_seconds")),
                    "" if stop is None else stop.payload.get("error_type"),
                ]
            )
        )
    return _table(("Stage", "Experiment Directory", "Status", "Elapsed Seconds", "Error"), rows)


def _stage_section(records: list[EventRecord], stage: str, body: str) -> str:
    if not _stage_started(records, stage):
        return f'<p class="empty">No declared {stage} stage found.</p>'
    return _stage_lifecycle_cards(records, stage) + body


def _stage_lifecycle_cards(records: list[EventRecord], stage: str) -> str:
    stop = _stage_stop(records, stage)
    items = (
        ("Stage", stage),
        ("Status", None if stop is None else stop.payload.get("status")),
        ("Elapsed Seconds", None if stop is None else _fmt(stop.payload.get("elapsed_seconds"))),
    )
    return (
        '<div class="cards">'
        + "".join(_metric_card(label, value) for label, value in items)
        + "</div>"
    )


def _convergence_section(records: list[EventRecord]) -> str:
    trials = _of_type(records, "ConvergenceTrial")
    if not trials:
        return '<p class="empty">No convergence trials found.</p>'
    rows = []
    for record in trials:
        payload = record.payload
        rows.append(
            _row(
                [
                    payload.get("pass_index"),
                    payload.get("control"),
                    payload.get("trial_index"),
                    _fmt(payload.get("previous")),
                    _fmt(payload.get("candidate")),
                    _fmt(payload.get("r_factor")),
                    payload.get("n_compared_hkl"),
                ]
            )
        )
    return _table(
        ("Pass", "Control", "Trial", "Previous", "Candidate", "R-factor", "Compared HKL"),
        rows,
    )


def _preprocess_section(records: list[EventRecord]) -> str:
    return (
        "<h3>Preprocess Summary</h3>"
        + _preprocess_summary(records)
        + "<h3>Orientation Optimization</h3>"
        + _orientation_table(records)
        + "<h3>Thickness Grids</h3>"
        + _thickness_summary(records)
    )


def _preprocess_summary(records: list[EventRecord]) -> str:
    completed = _last(records, "PreprocessCompleted")
    if completed is None:
        return '<p class="empty">No PreprocessCompleted event found.</p>'
    payload = completed.payload
    items = (
        ("Rotations", payload.get("n_rotations")),
        ("Matched HKL", payload.get("matched_hkl")),
        ("Total HKL", payload.get("total_hkl")),
    )
    return (
        '<div class="cards">'
        + "".join(_metric_card(label, value) for label, value in items)
        + "</div>"
    )


def _inference_section(records: list[EventRecord]) -> str:
    completed = _last(records, "InferenceCompleted")
    scored = _of_type(records, "RotationScored")
    if completed is None and not scored:
        return '<p class="empty">No inference events found.</p>'
    parts: list[str] = []
    if completed is not None:
        payload = completed.payload
        parts.append(
            '<div class="cards">'
            + "".join(
                _metric_card(label, value)
                for label, value in (
                    ("Rotations", payload.get("n_rotations")),
                    ("Evaluated", payload.get("n_evaluated")),
                    ("Mean R_obs", _fmt(payload.get("mean_r_obs"))),
                )
            )
            + "</div>"
        )
    if scored:
        rows = [
            _row(
                [
                    record.rotation_index,
                    _fmt(record.payload.get("r_obs")),
                    record.payload.get("n_observed"),
                    record.payload.get("n_beams"),
                ]
            )
            for record in scored
        ]
        parts.append(_table(("Rotation", "R_obs", "Observed HKL", "Solve Beams"), rows))
    return "".join(parts)


def _refinement_section(records: list[EventRecord]) -> str:
    completed = _last(records, "RefinementCompleted")
    parts: list[str] = []
    if completed is not None:
        payload = completed.payload
        best_loss = payload.get("best_loss")
        parts.append(
            '<div class="cards">'
            + "".join(
                _metric_card(label, value)
                for label, value in (
                    ("Steps", payload.get("n_steps")),
                    ("Best Epoch", _one_based(payload.get("best_step"))),
                    ("Best Loss", _fmt(best_loss)),
                    ("Selection", payload.get("selection")),
                )
            )
            + "</div>"
        )
    parts.append("<h3>Epoch Curve</h3>" + _epoch_table(records))
    parts.append("<h3>Per-Dataset Summary</h3>" + _dataset_summary(records))
    outputs = _last(records, "RefinementOutputsWritten")
    if outputs is not None and outputs.artifacts:
        rows = [_row((name, path)) for name, path in sorted(outputs.artifacts.items())]
        parts.append("<h3>Outputs</h3>" + _table(("Artifact", "Path"), rows))
    return "".join(parts)


def _epoch_table(records: list[EventRecord]) -> str:
    steps = _of_type(records, "RefinementStep")
    if not steps:
        return '<p class="empty">No refinement epochs found.</p>'
    rows = []
    for record in steps:
        payload = record.payload
        rows.append(
            _row(
                [
                    _one_based(payload.get("iteration")),
                    _fmt(payload.get("loss")),
                    _mean(
                        payload.get("wr2"),
                        payload.get("n_wr2_evaluated"),
                        payload.get("n_rotations"),
                    ),
                    _mean(
                        payload.get("r_obs"),
                        payload.get("n_r_obs_evaluated"),
                        payload.get("n_rotations"),
                    ),
                    _mean(
                        payload.get("val_wr2"),
                        payload.get("val_n_wr2_evaluated"),
                        payload.get("val_n_rotations"),
                    ),
                    _mean(
                        payload.get("val_r_obs"),
                        payload.get("val_n_r_obs_evaluated"),
                        payload.get("val_n_rotations"),
                    ),
                ]
            )
        )
    return _table(
        ("Epoch", "Loss", "Train wR2", "Train R_obs", "Validation wR2", "Validation R_obs"),
        rows,
    )


def _orientation_table(records: list[EventRecord]) -> str:
    fits = _of_type(records, "OrientationOptimized")
    if not fits:
        return '<p class="empty">No orientation optimization events found.</p>'
    rows = []
    for record in sorted(fits, key=lambda item: (item.dataset or "", item.rotation_index or -1)):
        payload = record.payload
        residual = str(payload.get("residual", "score"))
        score = _number(payload.get("score"))
        seed_score = _number(payload.get("seed_score"))
        improvement = None if score is None or seed_score is None else seed_score - score
        rows.append(
            _row(
                [
                    record.dataset or "",
                    record.rotation_index,
                    residual,
                    _fmt(seed_score),
                    _fmt(score),
                    _fmt(improvement),
                    _fmt(payload.get("alpha")),
                    _fmt(payload.get("beta")),
                    _fmt(payload.get("omega")),
                    payload.get("n_matched_hkl"),
                    payload.get("n_trials"),
                    payload.get("n_passes"),
                ]
            )
        )
    return _table(
        (
            "Dataset",
            "Rotation",
            "Residual",
            "Before",
            "After",
            "Improvement",
            "Delta alpha deg",
            "Delta beta deg",
            "Delta omega deg",
            "Matched HKL",
            "Trials",
            "Passes",
        ),
        rows,
    )


def _dataset_summary(records: list[EventRecord]) -> str:
    metrics = [record for record in _of_type(records, "RefinedRotationMetrics") if record.dataset]
    datasets = sorted({str(record.dataset) for record in metrics})
    if len(datasets) <= 1:
        return '<p class="empty">Single-dataset run; per-dataset summary omitted.</p>'
    grouped: dict[str, list[EventRecord]] = defaultdict(list)
    for record in metrics:
        grouped[str(record.dataset)].append(record)
    rows = []
    for dataset in datasets:
        group = grouped[dataset]
        wr2 = [_number(record.payload.get("wr2")) for record in group]
        r_obs = [_number(record.payload.get("r_obs")) for record in group]
        rows.append(
            _row(
                [
                    dataset,
                    len(group),
                    _fmt(_finite_mean(wr2)),
                    _fmt(_finite_mean(r_obs)),
                    sum(int(record.payload.get("n_matched", 0)) for record in group),
                    sum(1 for record in group if record.payload.get("is_validation")),
                ]
            )
        )
    return _table(
        ("Dataset", "Rotations", "Mean wR2", "Mean R_obs", "Matched HKL", "Validation"),
        rows,
    )


def _thickness_summary(records: list[EventRecord]) -> str:
    fits = _of_type(records, "ThicknessOptimized")
    if not fits:
        return '<p class="empty">No thickness grid events found.</p>'
    grouped: dict[str, list[EventRecord]] = defaultdict(list)
    for record in fits:
        grouped[record.dataset or ""].append(record)
    rows = []
    for dataset, group in sorted(grouped.items()):
        thicknesses = [_number(record.payload.get("thickness")) for record in group]
        scores = [_number(record.payload.get("score")) for record in group]
        rows.append(
            _row(
                [
                    dataset,
                    len(group),
                    _fmt(_finite_mean(thicknesses)),
                    _fmt(_finite_mean(scores)),
                    max(len(record.series.get("candidate_score", ())) for record in group),
                ]
            )
        )
    return _table(("Dataset", "Rotations", "Mean Thickness", "Mean Score", "Grid Points"), rows)


def _event_row(record: EventRecord) -> str:
    measurements = ", ".join(f"{name}={value:g}" for name, value in record.measurements.items())
    series = ", ".join(f"{name}[{len(values)}]" for name, values in record.series.items())
    cells = [
        str(record.sequence),
        record.event_type,
        record.channel,
        "" if record.step is None else str(record.step),
        "" if record.dataset is None else record.dataset,
        "" if record.rotation_index is None else str(record.rotation_index),
        measurements,
        series,
    ]
    return _row(cells)


def _table(headers: Iterable[object], rows: Iterable[str]) -> str:
    header = "<thead>" + _row(headers, tag="th") + "</thead>"
    body = "<tbody>" + "\n".join(rows) + "</tbody>"
    return f"<table>{header}{body}</table>"


def _row(cells: Iterable[object], *, tag: str = "td") -> str:
    return "<tr>" + "".join(f"<{tag}>{_esc(_blank_none(cell))}</{tag}>" for cell in cells) + "</tr>"


def _metric_card(label: str, value: object) -> str:
    return f'<div class="metric"><strong>{_esc(label)}</strong><span>{_esc(_blank_none(value))}</span></div>'


def _of_type(records: list[EventRecord], event_type: str) -> list[EventRecord]:
    return [record for record in records if record.event_type == event_type]


def _last(records: list[EventRecord], event_type: str) -> EventRecord | None:
    for record in reversed(records):
        if record.event_type == event_type:
            return record
    return None


def _stage_started(records: list[EventRecord], stage: str) -> bool:
    return any(
        record.event_type == "RunStageStarted" and record.payload.get("stage") == stage
        for record in records
    )


def _stage_stop(records: list[EventRecord], stage: str) -> EventRecord | None:
    for record in reversed(records):
        if record.event_type == "RunStageStopped" and record.payload.get("stage") == stage:
            return record
    return None


def _mean(value: object, evaluated: object, total: object) -> str:
    rendered = _fmt(value)
    if evaluated is None or total is None:
        return rendered
    return f"{rendered} [{_blank_none(evaluated)}/{_blank_none(total)}]"


def _finite_mean(values: Iterable[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return fmean(finite) if finite else None


def _one_based(value: object) -> int | None:
    number = _number(value)
    return None if number is None else int(number) + 1


def _number(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _fmt(value: object) -> str:
    number = _number(value)
    if number is None:
        return ""
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.6g}"


def _blank_none(value: object) -> object:
    return "" if value is None else value


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    raise SystemExit(main())

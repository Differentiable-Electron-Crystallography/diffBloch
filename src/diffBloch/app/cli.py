"""Thin command-line entry point.

This is the orchestration / SLURM boundary: a workflow engine (e.g. Dagster or Prefect) shells out
to it, or a SLURM job runs it. Kept deliberately thin — it delegates to the library and holds no
science.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import ValidationError

from diffBloch import __version__
from diffBloch.app.loggers import ConsoleLogger, ReportLogger, print_summary_box
from diffBloch.app.loggers.summary import SummaryLogger
from diffBloch.app.program import (
    converge_experiment,
    preprocess_experiment,
    refine_experiment,
    run_experiment,
)
from diffBloch.config import load_config, write_experiment_lock
from diffBloch.observability import (
    EventRecord,
    Logger,
    MultiLogger,
)

# The input/config failures a command reports as a one-line `error:` instead of a traceback.
# Anything else is a bug or an interrupt and propagates (and `_reported_run` discards the partial
# report).
_COMMAND_ERRORS = (FileNotFoundError, ValueError, ValidationError, yaml.YAMLError)


def _add_stage_flags(parser: argparse.ArgumentParser) -> None:
    """Add the flags shared by ``infer``, ``preprocess``, and ``refine`` (same preprocess surface)."""
    parser.add_argument("experiment_directory", help="Path to the experiment directory")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="ignore any existing preprocess checkpoints and recompute (regenerates each "
        "dataset's plan.<stem>.npz/.lock)",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="neither read nor write the preprocess checkpoint (leave the experiment dir alone)",
    )
    parser.add_argument(
        "--device",
        metavar="DEVICE",
        default="cuda",
        help="run the forward solve on this torch device (default: cuda; use 'cpu' to override)",
    )
    parser.add_argument(
        "--workers",
        metavar="N",
        type=int,
        default=1,
        help="fan orientation-plan builds and per-rotation searches over N threads (default 1); "
        "cap host threads to 1 (OMP_NUM_THREADS/MKL_NUM_THREADS/TORCH_NUM_THREADS, or "
        "torch.set_num_threads(1)) or the node-sized BLAS/torch pools oversubscribe the cores",
    )
    parser.add_argument(
        "--max-batch",
        metavar="N",
        type=int,
        default=None,
        help="cap the matrix_exp propagator block to N (N,N) operators (memory only, matches the "
        "unbounded solve to machine precision); default derives a memory-safe block per beam "
        "count. Raise to fill a larger GPU, e.g. 1024 on a high-memory accelerator",
    )


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to a subcommand. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="diffbloch",
        description="Differentiable Bloch-wave electron-diffraction structure refinement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "a typical workflow:\n"
            "  diffbloch validate experiment/experiment.yaml\n"
            "  diffbloch converge experiment/\n"
            "  diffbloch preprocess experiment/\n"
            "  diffbloch refine experiment/\n"
            "\n"
            "Each command has its own flags; see e.g. 'diffbloch refine --help'. "
            "reproducibility/experiment.lock is created automatically on first run; "
            "use 'diffbloch lock-experiment --force experiment/' to refresh it after inputs "
            "intentionally change."
        ),
    )
    parser.add_argument("--version", action="version", version=f"diffbloch {__version__}")
    parser.add_argument("--debug", action="store_true", help="show full tracebacks on error")
    sub = parser.add_subparsers(dest="command")

    # Registered in typical workflow order so the --help listing reads as the pipeline.
    p_validate = sub.add_parser("validate", help="Validate an experiment.yaml and report")
    p_validate.add_argument("config", help="Path to experiment.yaml")

    p_lock_experiment = sub.add_parser(
        "lock-experiment",
        help="Create reproducibility/experiment.lock from the current input files (other commands "
        "create it automatically on first run; existing locks require --force)",
    )
    p_lock_experiment.add_argument(
        "--force",
        action="store_true",
        help="rewrite an existing experiment.lock after an intentional input change; this "
        "invalidates existing plan and refinement locks",
    )
    p_lock_experiment.add_argument("experiment_directory", help="Path to the experiment directory")

    p_converge = sub.add_parser(
        "converge", help="Test convergence of g_max, sg_max, and rocking-curve tilt steps"
    )
    p_converge.add_argument("experiment_directory", help="Path to the experiment directory")
    p_converge.add_argument(
        "--device",
        metavar="DEVICE",
        default="cuda",
        help="run the convergence simulations on this torch device (default: cuda)",
    )
    p_converge.add_argument(
        "--orientations",
        metavar="N",
        type=int,
        default=1,
        help="use the first N orientations for convergence testing (default: 1)",
    )
    p_preprocess = sub.add_parser(
        "preprocess", help="Settle the coupled preprocess Plan and write the checkpoint (no score)"
    )
    _add_stage_flags(p_preprocess)
    p_infer = sub.add_parser("infer", help="Score every rotation of an experiment")
    _add_stage_flags(p_infer)
    p_refine = sub.add_parser(
        "refine", help="Gradient-refine the structure against the data (reuses the checkpoint)"
    )
    _add_stage_flags(p_refine)
    p_refine.add_argument(
        "--verbose-refinement",
        action="store_true",
        help="also report per-rotation wR2/R_obs/diffraction-loss every step, not just the epoch "
        "mean (n_orientations x louder; a diagnosis tool, off by default)",
    )
    p_refine.add_argument(
        "--profile",
        action="store_true",
        help="log per-phase wall time (structure factors, each rotation's solve, backward, "
        "optimizer step) via stdlib diagnostics logging; forces a CUDA sync per measured block "
        "(real overhead) so use only to diagnose one run, not routinely",
    )
    p_refine.add_argument(
        "--no-checkpoint-activations",
        action="store_true",
        help="do not gradient-checkpoint each per-orientation/per-segment solve; trades a full "
        "forward recompute on backward for higher peak memory (gradients are unaffected either "
        "way) -- try this if backward is much slower than forward and you have memory headroom",
    )

    args = parser.parse_args(argv)

    if args.command == "validate":
        # Expected user errors get a concise stderr message + nonzero exit; tracebacks are reserved
        # for --debug. This is the CLI/orchestration boundary, not a place to leak Python internals.
        try:
            cfg = load_config(args.config)
        except (FileNotFoundError, yaml.YAMLError, ValidationError) as exc:
            if args.debug:
                raise
            print(f"error: {args.config}: {exc}", file=sys.stderr)
            return 1
        print(f"OK: experiment '{cfg.name}' validated.")
        return 0

    if args.command == "lock-experiment":
        lock_path = (
            Path(args.experiment_directory) / "reproducibility" / "experiment.lock"
        ).resolve()
        replacing = lock_path.exists()
        try:
            experiment_lock = write_experiment_lock(args.experiment_directory, force=args.force)
        except (FileExistsError, FileNotFoundError, ValidationError, yaml.YAMLError) as exc:
            if args.debug:
                raise
            print(f"error: {exc}", file=sys.stderr)
            return 1
        refs = (
            [entry.ref for entry in experiment_lock.experimental_data]
            if isinstance(experiment_lock.experimental_data, list)
            else [experiment_lock.experimental_data.ref]
        )
        print(f"{'rewrote' if replacing else 'wrote'} {lock_path}")
        if replacing:
            print(
                "warning: if this accepts changed input bytes, existing plan and refinement locks "
                "are invalid and must be regenerated"
            )
        print(f"  - {'Structure':<20} {experiment_lock.structure.ref}")
        for ref in refs:
            print(f"  - {'Experimental data':<20} {ref}")
        return 0

    if args.command == "infer":
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
        )
        try:
            with _reported_run(args.experiment_directory) as run:
                run_experiment(
                    args.experiment_directory,
                    logger=run.logger,
                    checkpoint=not args.no_checkpoint,
                    refresh=args.refresh,
                    device=args.device,
                    workers=args.workers,
                    max_batch=args.max_batch,
                )
        except _COMMAND_ERRORS as exc:
            if args.debug:
                raise
            return _error(exc)
        # ConsoleLogger printed "INFER COMPLETE" off the run's terminal event.
        _print_notebooks(_write_stage_notebooks(run.report_path, args.experiment_directory))
        return 0

    if args.command == "preprocess":
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
        )
        try:
            with _reported_run(args.experiment_directory) as run:
                plan = preprocess_experiment(
                    args.experiment_directory,
                    logger=run.logger,
                    checkpoint=not args.no_checkpoint,
                    refresh=args.refresh,
                    device=args.device,
                    workers=args.workers,
                    max_batch=args.max_batch,
                )
        except _COMMAND_ERRORS as exc:
            if args.debug:
                raise
            return _error(exc)
        # ConsoleLogger printed "PREPROCESS COMPLETE" the moment preprocessing settled -- the same
        # box a refine/infer run gets, from the same sink.
        print()
        print("Pipeline")
        for index, record in enumerate(plan.provenance, start=1):
            print(f"  {index:>2}. {record.name.replace('_', ' ').title()}")
        print()
        # List the checkpoint pairs actually on disk (one per dataset; none under --no-checkpoint).
        reproducibility_dir = Path(args.experiment_directory) / "reproducibility"
        checkpoints = (
            sorted(reproducibility_dir.glob("plan.*.npz")) if reproducibility_dir.is_dir() else []
        )
        if checkpoints:
            print("Output files")
            for npz in checkpoints:
                print(f"  • {'Plan':<20} {npz.resolve()}")
                lock = npz.with_suffix(".lock")
                if lock.exists():
                    print(f"  • {'Plan Lock':<20} {lock.resolve()}")
        else:
            print("Output files")
        _print_notebooks(_write_stage_notebooks(run.report_path, args.experiment_directory))
        return 0

    if args.command == "refine":
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
        )
        try:
            with _reported_run(args.experiment_directory, refinement_report=True) as run:
                refine_experiment(
                    args.experiment_directory,
                    logger=run.logger,
                    checkpoint=not args.no_checkpoint,
                    refresh=args.refresh,
                    device=args.device,
                    workers=args.workers,
                    max_batch=args.max_batch,
                    verbose=args.verbose_refinement,
                    profile=args.profile,
                    checkpoint_activations=not args.no_checkpoint_activations,
                )
        except _COMMAND_ERRORS as exc:
            if args.debug:
                raise
            return _error(exc)
        # ConsoleLogger printed "REFINEMENT COMPLETE" and the artifact list off the run's terminal
        # event; this file adds the outputs it chose the location of. The JSON report is written
        # but not printed: the visualization notebooks are how it is read.
        print(f"  • {'Refinement Report':<20} {run.refinement_report_path}")
        _print_notebooks(_write_stage_notebooks(run.report_path, args.experiment_directory))
        return 0

    if args.command == "converge":
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
        )
        try:
            with _reported_run(args.experiment_directory) as run:
                settled = converge_experiment(
                    args.experiment_directory,
                    logger=run.logger,
                    device=args.device,
                    n_orientations=args.orientations,
                )
        except _COMMAND_ERRORS as exc:
            if args.debug:
                raise
            return _error(exc)
        print()
        print_summary_box(
            "CONVERGENCE COMPLETE",
            (
                ("g_max", f"{settled.g_max:g}"),
                ("sg_max", f"{settled.sg_max:g}"),
                ("Tilt steps", str(settled.tilt_steps)),
            ),
        )
        _print_notebooks(_write_stage_notebooks(run.report_path, args.experiment_directory))
        return 0

    parser.print_help()
    return 0


@dataclass(frozen=True)
class _ReportedRun:
    """The sinks one command runs against, plus where its reports will land."""

    logger: Logger
    report_path: Path
    refinement_report_path: Path | None


@contextmanager
def _reported_run(
    experiment_directory: str | Path, *, refinement_report: bool = False
) -> Iterator[_ReportedRun]:
    """Attach the console and canonical-report sinks for one command.

    ``refinement_report=True`` (refine) also attaches the ``refinement_report.txt`` writer. The
    JSONL report is written only when the command completes; on any exception, *including* the
    ones :func:`main` does not catch, the partial report is deleted.
    """
    root = Path(experiment_directory)
    if not root.exists():
        raise FileNotFoundError(root)
    report = ReportLogger(
        ReportLogger.timestamped_path(root / "reproducibility" / "reports").resolve(),
        completed_only=True,
    )
    sinks: tuple[Logger, ...] = (ConsoleLogger(), report)
    refinement_report_path = None
    if refinement_report:
        refinement_report_path = (root / "refinement_report.txt").resolve()
        sinks = (*sinks, SummaryLogger(refinement_report_path))
    with report:
        yield _ReportedRun(
            logger=MultiLogger(sinks),
            report_path=report.path,
            refinement_report_path=refinement_report_path,
        )


# The report rendering code lives in the checkout's tools/, outside the installed package.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_REPORT_TOOLS = _REPO_ROOT / "tools" / "event_report"

_REPORT_NOTEBOOK_CODE = """\
import sys
sys.path.insert(0, {repo_root!r})

import matplotlib.pyplot as plt
from IPython.display import HTML, Markdown, display
from tools.event_report import figures, reader

records = reader.stage_records(reader.read_records({report_name!r}), {stage!r})
for title, panels in figures.build_sections(records):
    display(Markdown(f"## {{title}}"))
    for figure in panels.values():
        display(HTML(figures.figure_dropdown_html(figure)))
plt.close("all")
"""


# Event types the visualization notebook has plots for. A stage that emitted none of them (e.g. a
# preprocess that only reused a checkpoint) gets no notebook rather than an empty one.
_PLOTTED_EVENTS = frozenset(
    {
        "ConvergenceTrial",
        "OrientationOptimized",
        "ThicknessOptimized",
        "RefinementStep",
        "RefinementOrientationStep",
        "RefinedRotationMetrics",
        "ThicknessProfile",
    }
)


# How a stage is named in its notebook's file name and heading, where that differs from the stage.
_NOTEBOOK_NAMES = {"converge": "convergence-test", "refine": "refinement"}


def _write_stage_notebooks(report_path: Path, experiment_directory: str | Path) -> list[Path]:
    """Write one visualization notebook per stage of the run that has something to plot.

    A command's report can hold several stages (``refine`` runs ``preprocess`` first); each gets its
    own notebook beside the report, named by the stage and the run's local date and time (e.g.
    ``refinement_2026-10-06_15-30-33.ipynb``), plotting only the events emitted while that stage
    was the innermost one running. Nothing is written when the rendering code in
    ``tools/event_report`` is absent, e.g. an installed package rather than a checkout.
    """
    if not _REPORT_TOOLS.is_dir():
        return []
    # Stages in the order their first plottable event appeared (preprocess before refine).
    plotted_stages: list[str] = []
    open_stages: list[str] = []
    for line in report_path.read_text().splitlines():
        record = EventRecord.model_validate_json(line)
        if record.event_type == "RunStageStarted":
            open_stages.append(str(record.payload["stage"]))
        if (
            open_stages
            and record.event_type in _PLOTTED_EVENTS
            and open_stages[-1] not in plotted_stages
        ):
            plotted_stages.append(open_stages[-1])
        if record.event_type == "RunStageStopped" and open_stages:
            open_stages.pop()
    if not plotted_stages:
        return []
    experiment_name = load_config(Path(experiment_directory) / "experiment.yaml").name
    # The report's stamp is UTC; the notebook is named and headed in local time for people.
    started = (
        datetime.strptime(report_path.stem.removeprefix("report-"), "%Y%m%dT%H%M%SZ")
        .replace(tzinfo=UTC)
        .astimezone()
    )
    written = []
    for stage in plotted_stages:
        name = _NOTEBOOK_NAMES.get(stage, stage)
        written.append(
            _write_report_notebook(
                report_path,
                stage,
                report_path.with_name(f"{name}_{started:%Y-%m-%d_%H-%M-%S}.ipynb"),
                f"{name.capitalize()} visualization notebook, {experiment_name}, "
                f"{started.day} {started:%B %Y %H:%M}",
            )
        )
    return written


def _print_notebooks(notebooks: list[Path]) -> None:
    for notebook in notebooks:
        print(f"  • {'Visualization':<20} {notebook}")


def _write_report_notebook(report_path: Path, stage: str, notebook_path: Path, title: str) -> Path:
    """Write ``notebook_path``: a title and one code cell plotting ``stage`` of ``report_path``.

    Run All renders the plots for that stage's events. The notebook reads the report by file name
    relative to itself.
    """
    code = _REPORT_NOTEBOOK_CODE.format(
        repo_root=str(_REPO_ROOT), report_name=report_path.name, stage=stage
    )
    notebook = {
        "cells": [
            {"cell_type": "markdown", "id": "title", "metadata": {}, "source": [f"# {title}"]},
            {
                "cell_type": "code",
                "id": "plots",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code.splitlines(keepends=True),
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(notebook, indent=1) + "\n")
    return notebook_path


def _error(exc: Exception) -> int:
    print(f"error: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

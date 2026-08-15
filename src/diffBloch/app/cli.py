"""Thin command-line entry point.

This is the orchestration / SLURM boundary: a workflow engine (e.g. Dagster or Prefect) shells out
to it, or a SLURM job runs it. Kept deliberately thin — it delegates to the library and holds no
science.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import cast

import yaml
from pydantic import ValidationError

from diffBloch import __version__
from diffBloch.app.loggers import ConsoleLogger, CSVLogger, residual_label
from diffBloch.app.loggers.summary import SummaryLogger
from diffBloch.app.program import (
    converge_experiment,
    preprocess_experiment,
    refine_experiment,
    run_experiment,
)
from diffBloch.config import load_config, write_experiment_lock
from diffBloch.engine.plan import OrientationPlanLike
from diffBloch.observability import (
    NULL_LOGGER,
    Logger,
    MultiLogger,
    OrientationOptimized,
    RecordingLogger,
)


def _print_summary_box(title: str, rows: tuple[tuple[str, str], ...]) -> None:
    """Print a consistently aligned 62-column completion summary.

    ``label_width`` must exceed the longest label any caller passes: the format spec pads but does
    not truncate, so a longer label silently pushes its value past the box border and misaligns that
    row against every other.
    """
    width = 62
    label_width = 26
    value_width = width - label_width - 3
    heading = f" {title} "
    print(f"╭{heading:─^{width}}╮")
    for label, value in rows:
        print(f"│ {label:<{label_width}} {value:<{value_width}} │")
    print(f"╰{'─' * width}╯")


def _add_stage_flags(parser: argparse.ArgumentParser) -> None:
    """Add the flags shared by ``infer``, ``preprocess``, and ``refine`` (same preprocess surface)."""
    parser.add_argument("experiment_directory", help="Path to the experiment directory")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="silence the per-step / per-rotation observation stream (console logging is on by "
        "default; the run summary line still prints)",
    )
    parser.add_argument(
        "--csv", metavar="PATH", help="append per-rotation observations to a long-format CSV log"
    )
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
    parser.add_argument(
        "--plot-thickness",
        action="store_true",
        help="save one wR2-vs-thickness PNG per rotation from the thickness grid search; ORs with "
        "preprocess.thickness.plot in experiment.yaml, so "
        "either turns it on. Defaults to '<inputs.structure's directory>/thickness_optim', "
        "override with --plot-thickness-dir",
    )
    parser.add_argument(
        "--plot-thickness-dir",
        metavar="PATH",
        default=None,
        help="override the output directory for thickness plots; only takes effect when plotting "
        "is on (--plot-thickness or preprocess.thickness.plot)",
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
    p_converge.add_argument(
        "--quiet",
        action="store_true",
        help="silence the per-trial observation stream (the settled result still prints)",
    )
    p_converge.add_argument(
        "--csv", metavar="PATH", help="append per-trial observations to a long-format CSV log"
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
        if not args.quiet:
            logging.basicConfig(
                level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
            )
        try:
            result = run_experiment(
                args.experiment_directory,
                logger=_build_logger(console=not args.quiet, csv=args.csv),
                checkpoint=not args.no_checkpoint,
                refresh=args.refresh,
                device=args.device,
                workers=args.workers,
                max_batch=args.max_batch,
                plot_thickness=args.plot_thickness,
                plot_thickness_dir=args.plot_thickness_dir,
            )
        except (FileNotFoundError, ValueError, ValidationError, yaml.YAMLError) as exc:
            if args.debug:
                raise
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"evaluated {result.n_evaluated} rotations; mean R_obs = {result.mean_r_obs:.4f}")
        return 0

    if args.command == "preprocess":
        if not args.quiet:
            logging.basicConfig(
                level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
            )
        try:
            progress_logger = _build_logger(console=not args.quiet, csv=args.csv)
            summary_logger = RecordingLogger()
            logger: Logger = (
                summary_logger
                if progress_logger is NULL_LOGGER
                else MultiLogger((progress_logger, summary_logger))
            )
            plan = preprocess_experiment(
                args.experiment_directory,
                logger=logger,
                checkpoint=not args.no_checkpoint,
                refresh=args.refresh,
                device=args.device,
                workers=args.workers,
                max_batch=args.max_batch,
                plot_thickness=args.plot_thickness,
                plot_thickness_dir=args.plot_thickness_dir,
            )
        except (FileNotFoundError, ValueError, ValidationError, yaml.YAMLError) as exc:
            if args.debug:
                raise
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print()
        built = cast(tuple[OrientationPlanLike, ...], plan.orientations)
        total_hkl = sum(int(op.pattern.hkl.shape[0]) for op in built)
        matched_hkl = sum(int(op.alignment.hkl.shape[0]) for op in built)
        fitted = [
            event for event in summary_logger.events if isinstance(event, OrientationOptimized)
        ]
        mean_loss = (
            f"{sum(event.score for event in fitted) / len(fitted):.6g}"
            if fitted
            else "n/a (checkpoint reused)"
        )
        mean_label = f"Mean {residual_label(fitted[0].residual)}" if fitted else "Mean score"
        _print_summary_box(
            "PREPROCESS COMPLETE",
            (
                ("Rotations", str(len(plan.orientations))),
                ("Stages", str(len(plan.provenance))),
                ("Total HKLs", str(total_hkl)),
                ("Matched HKLs", str(matched_hkl)),
                ("Solve beams (max/rotation)", str(max(int(op.beam_hkl.shape[0]) for op in built))),
                (mean_label, mean_loss),
            ),
        )
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
        return 0

    if args.command == "refine":
        if not args.quiet:
            logging.basicConfig(
                level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
            )
        try:
            # The written summary is one more sink on the run's event stream, chosen here beside
            # the console/CSV ones rather than by refine_experiment: an API caller composes it (or
            # not) for themselves instead of having a file appear as a side effect of refining.
            report_path = (Path(args.experiment_directory) / "refinement_report.txt").resolve()
            refine_sinks: tuple[Logger, ...] = (
                _build_logger(console=not args.quiet, csv=args.csv),
                SummaryLogger(report_path),
            )
            refined = refine_experiment(
                args.experiment_directory,
                logger=MultiLogger(refine_sinks),
                checkpoint=not args.no_checkpoint,
                refresh=args.refresh,
                device=args.device,
                workers=args.workers,
                max_batch=args.max_batch,
                verbose=args.verbose_refinement,
                profile=args.profile,
                checkpoint_activations=not args.no_checkpoint_activations,
                plot_thickness=args.plot_thickness,
                plot_thickness_dir=args.plot_thickness_dir,
            )
        except (FileNotFoundError, ValueError, ValidationError, yaml.YAMLError) as exc:
            if args.debug:
                raise
            print(f"error: {exc}", file=sys.stderr)
            return 1
        best = refined.history[refined.best_step]
        wr2 = "n/a" if best.wr2 is None else f"{best.wr2:.6g}"
        r_obs = "n/a" if best.r_obs is None else f"{best.r_obs:.6g}"
        diff_loss = "n/a" if best.diff_loss is None else f"{best.diff_loss:.6g}"
        counts = refined.reflection_counts
        print()
        _print_summary_box(
            "REFINEMENT COMPLETE",
            (
                ("Best epoch", str(refined.best_step + 1)),
                ("Objective", f"{refined.best_loss:.6g}"),
                ("wR2", wr2),
                ("R_obs", r_obs),
                ("Diffraction loss", diff_loss),
                (
                    "HKLs (Observed/total)",
                    f"{counts['matched_i_gt_3sigma']} / {counts['matched']}",
                ),
            ),
        )
        print()
        print("Output files")
        for name, path in refined.artifacts.items():
            print(f"  • {name.replace('_', ' ').title():<20} {path}")
        print(f"  • {'Refinement Report':<20} {report_path}")
        return 0

    if args.command == "converge":
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
        )
        try:
            settled = converge_experiment(
                args.experiment_directory,
                logger=_build_logger(console=not args.quiet, csv=args.csv),
                device=args.device,
                n_orientations=args.orientations,
            )
        except (FileNotFoundError, ValueError, ValidationError, yaml.YAMLError) as exc:
            if args.debug:
                raise
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print("========================================")
        print("HYPERPARAMETER OPTIMIZATION RESULT")
        print(f"gmax: {settled.g_max:g}")
        print(f"sgmax: {settled.sg_max:g}")
        print(f"tilt_steps: {settled.tilt_steps}")
        print("========================================")
        print(
            f"optimized_hyperparams gmax={settled.g_max:g} "
            f"sgmax={settled.sg_max:g} tilt_steps={settled.tilt_steps}"
        )
        return 0

    parser.print_help()
    return 0


def _build_logger(*, console: bool, csv: str | None, per_rotation: bool = True) -> Logger:
    """Combine the requested observation sinks (none => the null logger that discards events).

    ``per_rotation`` opts the console into the settled per-rotation stream.
    """
    sinks: list[Logger] = []
    if console:
        sinks.append(ConsoleLogger(per_rotation=per_rotation))
    if csv is not None:
        sinks.append(CSVLogger(Path(csv)))
    if not sinks:
        return NULL_LOGGER
    if len(sinks) == 1:
        return sinks[0]
    return MultiLogger(tuple(sinks))


if __name__ == "__main__":
    sys.exit(main())

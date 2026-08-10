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
from torch import Tensor

from diffBloch import __version__
from diffBloch.app.loggers import ConsoleLogger, CSVLogger, residual_label
from diffBloch.app.loggers.summary import SummaryLogger
from diffBloch.app.program import (
    converge_experiment,
    preprocess_experiment,
    refine_experiment,
    run_experiment,
)
from diffBloch.config import load_config, pack_run
from diffBloch.engine import ModelRefinementResult
from diffBloch.engine.plan import OrientationPlanLike
from diffBloch.io import read_experimental_data
from diffBloch.observability import (
    NULL_LOGGER,
    Logger,
    MultiLogger,
    OrientationOptimized,
    RecordingLogger,
    RefinedRotationMetrics,
)
from diffBloch.preprocess.plan import Plan, unique_hkl_count


class _PromptFormatter(logging.Formatter):
    """Console formatter: plain timestamped lines, except WARNING+ gets a hard-to-miss banner.

    The plain format (``%(asctime)s %(message)s``) has no level name in it at all, so a warning --
    e.g. the negative-ADP auto-correction or structure/experimental-data cell mismatch checks --
    renders identically to routine progress output and scrolls past unnoticed in a wall of
    per-rotation lines. WARNING+ is rendered as a bold/colored banner instead; skipped when stderr
    isn't a real terminal (piped to a file/CI log), where ANSI codes would just be noise.
    """

    _TIMESTAMPED = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    _BOLD_YELLOW = "\033[1;33m"
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        base = self._TIMESTAMPED.format(record)
        if record.levelno < logging.WARNING or not sys.stderr.isatty():
            return base
        banner = f"{self._BOLD_YELLOW}##### WARNING #####{self._RESET}"
        return f"{banner}\n{base}\n{banner}"


def _configure_logging() -> None:
    """Install the console handler every ``run`` subcommand shares (see ``_PromptFormatter``)."""
    handler = logging.StreamHandler()
    handler.setFormatter(_PromptFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])


def _announce_multi_dataset(experiment_directory: str | Path) -> None:
    """Print a hard-to-miss banner at the top of the run when ``inputs.multi_dataset`` is set.

    Printed unconditionally (not through ``logging``, so it isn't affected by ``--quiet``) --
    running against several combined files instead of one is a fact about the whole run that
    should be obvious before any per-rotation output starts, not something to notice only by
    reading ``experiment.yaml``.
    """
    cfg = load_config(Path(experiment_directory) / "experiment.yaml")
    if not cfg.inputs.multi_dataset:
        return
    assert isinstance(cfg.inputs.exp_data, list)
    bold_cyan, reset = ("\033[1;36m", "\033[0m") if sys.stderr.isatty() else ("", "")
    print(f"{bold_cyan}{'#' * 62}", file=sys.stderr)
    print("MULTI-DATASET MODE", file=sys.stderr)
    print(f"Using {len(cfg.inputs.exp_data)} datasets:", file=sys.stderr)
    for path in cfg.inputs.exp_data:
        print(f"  - {path}", file=sys.stderr)
    print(f"{'#' * 62}{reset}", file=sys.stderr)


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


def _print_preprocess_summary(
    experiment_directory: str | Path,
    plan: Plan,
    fitted: list[OrientationOptimized],
) -> None:
    """Print the preprocess outcome: one box per dataset when combined, one box overall otherwise.

    A combined (``inputs.multi_dataset``) run gets **no pooled box** -- every number in it (HKLs,
    mean wR2) would blend datasets that can genuinely differ (different crystal, different damage
    state), which is exactly the kind of thing worth seeing broken out, not averaged away. Re-derives
    each file's rotation range the same way ``from_experiment`` assigns it (file order in
    ``inputs.exp_data``, a running offset, never restarting at 0) and slices ``plan.orientations`` /
    ``fitted`` by ``rotation_index`` into that range.
    """
    root = Path(experiment_directory)
    cfg = load_config(root / "experiment.yaml")
    built = cast(tuple[OrientationPlanLike, ...], plan.orientations)
    n_stages = len(plan.provenance)

    def box(
        title: str,
        group: tuple[OrientationPlanLike, ...],
        group_fitted: list[OrientationOptimized],
        *,
        include_stages: bool,
    ) -> None:
        mean_label = f"Mean {residual_label(group_fitted[0].residual)}" if group_fitted else "Mean score"
        mean_value = (
            f"{sum(event.score for event in group_fitted) / len(group_fitted):.6g}"
            if group_fitted
            else "n/a (checkpoint reused)"
        )
        rows = [("Rotations", str(len(group)))]
        if include_stages:
            rows.append(("Stages", str(n_stages)))
        rows += [
            ("Unique exp HKLs", str(unique_hkl_count(op.pattern.hkl for op in group))),
            ("Unique matched HKLs", str(unique_hkl_count(op.alignment.hkl for op in group))),
            (mean_label, mean_value),
        ]
        _print_summary_box(title, tuple(rows))

    if not cfg.inputs.multi_dataset:
        box("PREPROCESS COMPLETE", built, fitted, include_stages=True)
        return
    assert isinstance(cfg.inputs.exp_data, list)
    print(f"Stages: {n_stages}")
    print()
    offset = 0
    for path in cfg.inputs.exp_data:
        n_rotations = len(read_experimental_data(root / path).zone_axis_ids)
        end = offset + n_rotations
        group = tuple(op for op in built if offset <= op.pattern.rotation_index < end)
        group_fitted = [event for event in fitted if offset <= event.rotation_index < end]
        box(path, group, group_fitted, include_stages=False)
        print()
        offset = end


def _print_refinement_summary(
    experiment_directory: str | Path,
    refined: ModelRefinementResult,
    plan: Plan,
    rotations: list[RefinedRotationMetrics],
) -> None:
    """Print the refinement outcome: one box per dataset when combined, one box overall otherwise.

    Same reasoning as ``_print_preprocess_summary``: wR2/R_obs/matched counts can genuinely differ
    per combined dataset (different crystal state, different damage), so a single pooled box would
    hide exactly the kind of thing worth seeing broken out. ``Best epoch``/``Objective`` describe the
    one shared optimization (there's no separate training loop per dataset), so those print once,
    outside any box, rather than being repeated identically in every one or dropped.

    Matched/strong/unmatched are deduplicated ``(h, k, l)`` counts (:func:`unique_hkl_count`) off
    ``plan`` -- the *settled* geometry, unaffected by refinement -- rather than a sum of each
    rotation's own count, which double-counts any reflection seen in more than one rotation. wR2/
    R_obs still come from ``rotations`` (:class:`RefinedRotationMetrics`), since those genuinely do
    depend on the refined structure and can't be read off the geometry alone.
    """
    root = Path(experiment_directory)
    cfg = load_config(root / "experiment.yaml")
    built = cast(tuple[OrientationPlanLike, ...], plan.orientations)

    def strong_matched_hkl(op: OrientationPlanLike) -> Tensor:
        pattern_index = op.alignment.pattern_index
        strong = op.pattern.intensities[pattern_index] > 3.0 * op.pattern.sigmas[pattern_index]
        return op.alignment.hkl[strong]

    def box(
        title: str, group: list[RefinedRotationMetrics], op_group: tuple[OrientationPlanLike, ...]
    ) -> None:
        if not group:
            _print_summary_box(title, (("Rotations", "0 (n/a -- checkpoint reused)"),))
            return
        n_observed = unique_hkl_count(op.pattern.hkl for op in op_group)
        n_matched = unique_hkl_count(op.alignment.hkl for op in op_group)
        n_strong = unique_hkl_count(strong_matched_hkl(op) for op in op_group)
        _print_summary_box(
            title,
            (
                ("Rotations", str(len(group))),
                ("Mean wR2", f"{sum(r.wr2 for r in group) / len(group):.6g}"),
                ("Mean R_obs", f"{sum(r.r_obs for r in group) / len(group):.6g}"),
                ("Matched (strong/total)", f"{n_strong} / {n_matched}"),
                ("Unmatched HKLs", str(n_observed - n_matched)),
            ),
        )

    print(f"Best epoch: {refined.best_step + 1}    Objective: {refined.best_loss:.6g}")
    print()
    if not cfg.inputs.multi_dataset:
        box("REFINEMENT COMPLETE", rotations, built)
        return
    assert isinstance(cfg.inputs.exp_data, list)
    offset = 0
    for path in cfg.inputs.exp_data:
        n_rotations = len(read_experimental_data(root / path).zone_axis_ids)
        end = offset + n_rotations
        op_group = tuple(op for op in built if offset <= op.pattern.rotation_index < end)
        box(path, [r for r in rotations if r.dataset_label == path], op_group)
        print()
        offset = end


def _add_run_flags(parser: argparse.ArgumentParser) -> None:
    """Add the flags shared by ``run infer`` and ``run preprocess`` (same preprocess surface)."""
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
        "--tui",
        action="store_true",
        help="render the run as a live terminal dashboard instead of the scrolling console log "
        "(requires the 'diffBloch[tui]' extra); replaces --quiet's console stream rather than "
        "composing with it, since a live display owns the terminal",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="ignore any existing preprocess checkpoint and recompute (regenerates plan.npz/.lock)",
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
        "--orientations-csv",
        metavar="PATH",
        default=None,
        help="overwrite every rotation's orientation from a 'Rotation Index'/'Orientation Matrix' "
        "CSV before the recipe's fitting steps run (part of the recipe, so it restales any "
        "existing checkpoint); preprocess.optimize_orientation still controls whether the search "
        "then refines from that seed or is skipped so the imported orientations are used as-is",
    )
    parser.add_argument(
        "--plot-thickness",
        action="store_true",
        help="save one wR2-vs-thickness PNG per rotation from the thickness grid search (requires "
        "the 'diffBloch[plot]' extra); ORs with preprocess.thickness.plot in experiment.yaml, so "
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
    )
    parser.add_argument("--version", action="version", version=f"diffbloch {__version__}")
    parser.add_argument("--debug", action="store_true", help="show full tracebacks on error")
    sub = parser.add_subparsers(dest="command")

    p_validate = sub.add_parser("validate", help="Validate an experiment.yaml and report")
    p_validate.add_argument("config", help="Path to experiment.yaml")

    p_run = sub.add_parser("run", help="Run artifact commands")
    run_sub = p_run.add_subparsers(dest="run_command")
    p_infer = run_sub.add_parser("infer", help="Score every rotation of an experiment")
    _add_run_flags(p_infer)
    p_preprocess = run_sub.add_parser(
        "preprocess", help="Settle the coupled preprocess Plan and write the checkpoint (no score)"
    )
    _add_run_flags(p_preprocess)
    p_refine = run_sub.add_parser(
        "refine", help="Gradient-refine the structure against the data (reuses the checkpoint)"
    )
    _add_run_flags(p_refine)
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
    p_converge = run_sub.add_parser(
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
    p_pack = run_sub.add_parser("pack", help="Export a run directory for transfer/archive")
    p_pack.add_argument("run_directory", help="Path to canonical run artifact directory")
    p_pack.add_argument(
        "--format",
        choices=["zip", "tar", "bagit", "ro-crate"],
        default="zip",
        help="Export package format",
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

    if args.command == "run" and args.run_command == "infer":
        _announce_multi_dataset(args.experiment_directory)
        if not args.quiet:
            _configure_logging()
        try:
            result = run_experiment(
                args.experiment_directory,
                logger=_build_logger(console=not args.quiet, csv=args.csv, tui=args.tui),
                checkpoint=not args.no_checkpoint,
                refresh=args.refresh,
                device=args.device,
                workers=args.workers,
                max_batch=args.max_batch,
                orientations_csv=args.orientations_csv,
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

    if args.command == "run" and args.run_command == "preprocess":
        _announce_multi_dataset(args.experiment_directory)
        if not args.quiet:
            _configure_logging()
        try:
            progress_logger = _build_logger(console=not args.quiet, csv=args.csv, tui=args.tui)
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
                orientations_csv=args.orientations_csv,
                plot_thickness=args.plot_thickness,
                plot_thickness_dir=args.plot_thickness_dir,
            )
        except (FileNotFoundError, ValueError, ValidationError, yaml.YAMLError) as exc:
            if args.debug:
                raise
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print()
        fitted = [
            event for event in summary_logger.events if isinstance(event, OrientationOptimized)
        ]
        _print_preprocess_summary(args.experiment_directory, plan, fitted)
        print()
        print("Pipeline")
        for index, record in enumerate(plan.provenance, start=1):
            print(f"  {index:>2}. {record.name.replace('_', ' ').title()}")
        print()
        print("Output files")
        reproducibility_dir = Path(args.experiment_directory) / "reproducibility"
        print(f"  • {'Plan':<20} {(reproducibility_dir / 'plan.npz').resolve()}")
        print(f"  • {'Plan Lock':<20} {(reproducibility_dir / 'plan.lock').resolve()}")
        return 0

    if args.command == "run" and args.run_command == "refine":
        _announce_multi_dataset(args.experiment_directory)
        if not args.quiet:
            _configure_logging()
        try:
            # The written summary is one more sink on the run's event stream, chosen here beside
            # the console/CSV ones rather than by refine_experiment: an API caller composes it (or
            # not) for themselves instead of having a file appear as a side effect of refining.
            report_path = (Path(args.experiment_directory) / "refinement_report.txt").resolve()
            summary_logger = RecordingLogger()
            refine_sinks: tuple[Logger, ...] = (
                _build_logger(
                    console=not args.quiet, csv=args.csv, tui=args.tui, per_rotation=False
                ),
                SummaryLogger(report_path),
                summary_logger,
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
                orientations_csv=args.orientations_csv,
                plot_thickness=args.plot_thickness,
                plot_thickness_dir=args.plot_thickness_dir,
            )
        except (FileNotFoundError, ValueError, ValidationError, yaml.YAMLError) as exc:
            if args.debug:
                raise
            print(f"error: {exc}", file=sys.stderr)
            return 1
        rotations = [
            event for event in summary_logger.events if isinstance(event, RefinedRotationMetrics)
        ]
        # Cheap: refine_experiment already left a fresh, self-consistent checkpoint on disk, so this
        # is a checkpoint-reuse load (no recompute), just to get the settled geometry for the
        # deduplicated HKL counts below -- matched/strong/unmatched are unaffected by refinement.
        settled_plan = preprocess_experiment(
            args.experiment_directory,
            logger=NULL_LOGGER,
            checkpoint=True,
            refresh=False,
            device=args.device,
            workers=args.workers,
            max_batch=args.max_batch,
        )
        print()
        _print_refinement_summary(args.experiment_directory, refined, settled_plan, rotations)
        print("Output files")
        for name, path in refined.artifacts.items():
            print(f"  • {name.replace('_', ' ').title():<20} {path}")
        print(f"  • {'Refinement Report':<20} {report_path}")
        return 0

    if args.command == "run" and args.run_command == "converge":
        _announce_multi_dataset(args.experiment_directory)
        _configure_logging()
        try:
            settled = converge_experiment(
                args.experiment_directory,
                logger=ConsoleLogger(),
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

    if args.command == "run" and args.run_command == "pack":
        try:
            output = pack_run(args.run_directory, package_format=args.format)
        except (FileNotFoundError, ValueError) as exc:
            if args.debug:
                raise
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(output)
        return 0

    parser.print_help()
    return 0


def _build_logger(
    *, console: bool, csv: str | None, per_rotation: bool = True, tui: bool = False
) -> Logger:
    """Combine the requested observation sinks (none => the null logger that discards events).

    ``tui`` swaps the scrolling :class:`ConsoleLogger` for the live
    :class:`~diffBloch.app.loggers.tui.TuiLogger`; the two are alternatives, never both, because a
    live display owns the terminal while the console bridges events onto stdlib ``logging``.
    ``per_rotation`` opts the console into the settled per-rotation stream.
    """
    sinks: list[Logger] = []
    if console and tui:
        from diffBloch.app.loggers.tui import TuiLogger

        sinks.append(TuiLogger())
    elif console:
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

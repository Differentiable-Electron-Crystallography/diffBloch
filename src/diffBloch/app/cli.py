"""Thin command-line entry point.

This is the orchestration / SLURM boundary (synthesis §18): a Dagster/Prefect op shells out to it,
or a SLURM job runs it. Kept deliberately thin — it delegates to the library and holds no science.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from diffBloch import __version__
from diffBloch.app.loggers import ConsoleLogger, CSVLogger
from diffBloch.app.program import preprocess_experiment, run_experiment
from diffBloch.config import load_config, pack_run
from diffBloch.observability import NULL_LOGGER, Logger, MultiLogger


def _add_run_flags(parser: argparse.ArgumentParser) -> None:
    """Add the flags shared by ``run infer`` and ``run preprocess`` (same preprocess surface)."""
    parser.add_argument("experiment_directory", help="Path to the experiment directory")
    parser.add_argument(
        "--console", action="store_true", help="stream per-rotation observations to stderr"
    )
    parser.add_argument(
        "--csv", metavar="PATH", help="append per-rotation observations to a long-format CSV log"
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
        default=None,
        help="run the forward solve on this torch device (e.g. 'cuda'); default CPU",
    )
    parser.add_argument(
        "--workers",
        metavar="N",
        type=int,
        default=1,
        help="fan the per-rotation orientation search over N threads (default 1); cap host threads "
        "to 1 (OMP_NUM_THREADS/MKL_NUM_THREADS/TORCH_NUM_THREADS, or torch.set_num_threads(1)) or "
        "the node-sized BLAS/torch pools oversubscribe the cores",
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
        if args.console:
            logging.basicConfig(level=logging.INFO, format="%(message)s")
        try:
            result = run_experiment(
                args.experiment_directory,
                logger=_build_logger(console=args.console, csv=args.csv),
                checkpoint=not args.no_checkpoint,
                refresh=args.refresh,
                device=args.device,
                workers=args.workers,
            )
        except (FileNotFoundError, ValueError, ValidationError, yaml.YAMLError) as exc:
            if args.debug:
                raise
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"evaluated {result.n_evaluated} rotations; mean R_obs = {result.mean_r_obs:.4f}")
        return 0

    if args.command == "run" and args.run_command == "preprocess":
        if args.console:
            logging.basicConfig(level=logging.INFO, format="%(message)s")
        try:
            plan = preprocess_experiment(
                args.experiment_directory,
                logger=_build_logger(console=args.console, csv=args.csv),
                checkpoint=not args.no_checkpoint,
                refresh=args.refresh,
                device=args.device,
                workers=args.workers,
            )
        except (FileNotFoundError, ValueError, ValidationError, yaml.YAMLError) as exc:
            if args.debug:
                raise
            print(f"error: {exc}", file=sys.stderr)
            return 1
        recipe = ", ".join(record.name for record in plan.provenance)
        print(f"preprocessed {len(plan.orientations)} rotations; recipe: {recipe}")
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


def _build_logger(*, console: bool, csv: str | None) -> Logger:
    """Combine the requested observation sinks (none => the null logger that discards events)."""
    sinks: list[Logger] = []
    if console:
        sinks.append(ConsoleLogger())
    if csv is not None:
        sinks.append(CSVLogger(Path(csv)))
    if not sinks:
        return NULL_LOGGER
    if len(sinks) == 1:
        return sinks[0]
    return MultiLogger(tuple(sinks))


if __name__ == "__main__":
    sys.exit(main())

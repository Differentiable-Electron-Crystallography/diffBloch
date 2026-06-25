"""Thin command-line entry point.

This is the orchestration / SLURM boundary (synthesis §18): a Dagster/Prefect op shells out to it,
or a SLURM job runs it. Kept deliberately thin — it delegates to the library and holds no science.
"""

from __future__ import annotations

import argparse
import sys

import yaml
from pydantic import ValidationError

from diffBloch import __version__
from diffBloch.config import load_config, pack_run


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


if __name__ == "__main__":
    sys.exit(main())

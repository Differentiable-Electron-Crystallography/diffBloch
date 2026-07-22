"""Profile a few refinement steps to attribute the per-step cost (op-level breakdown).

The `--console` refinement stream reports per-step *totals* (~40 s/step on abiraterone/A100) but not
what *inside* a step dominates. This runs a handful of gradient steps under ``torch.profiler`` and
prints the CUDA op table, so the cost lands on the operations that map to the reference's forward
phases (``diffbloch_private`` times these as ``t_union`` / ``t_struct`` / ``t_off`` / ``t_diag`` /
``t_pinv``): ``aten::linalg_eigh`` = the eigensolve, ``aten::matrix_exp`` = the propagator,
``aten::bmm`` / ``aten::mm`` = the structure-matrix / gather, plus the autograd backward ops.

Usage: uv run python scripts/profile_refine.py [experiment_dir] [--device cuda] [--steps 5]
Run it on its own GPU (an uncontended profile) -- not alongside a live refine on the same device.
"""

from __future__ import annotations

import argparse
import sys

import torch
from torch.profiler import ProfilerActivity, profile

from diffBloch.app.program import preprocess_experiment
from diffBloch.config import load_experiment
from diffBloch.engine import (
    build_refinement_model,
    build_refinement_problem,
    run_refinement_model,
)
from diffBloch.io import read_observations, read_structure
from diffBloch.observability import NULL_LOGGER
from diffBloch.preprocess import build_engine, from_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment_dir", nargs="?", default="examples/experiments/abiraterone-checkpoint"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=5, help="profiled steps (after 1 warmup)")
    args = parser.parse_args(argv)

    root = args.experiment_dir
    cfg, _ = load_experiment(root)

    # Settle the Plan (reuse a checkpoint if present, else build one) and rejoin the structure side.
    prepared = preprocess_experiment(root, device=args.device, logger=NULL_LOGGER)
    structure = read_structure(
        f"{root}/{cfg.inputs.structure}", load_hydrogens=cfg.inputs.load_hydrogens
    )
    observations = read_observations(f"{root}/{cfg.inputs.observations}")
    refinement = from_experiment(structure, observations, cfg).refinement

    engine = build_engine(
        prepared, refinement, loss=cfg.refinement.objective.to_loss(), method=cfg.solver.refine
    )
    initial = refinement.params.to(args.device)
    model = build_refinement_model(initial=initial)
    problem = build_refinement_problem()
    trainable = cfg.refinement.trainable.to_spec()

    run = lambda steps: run_refinement_model(  # noqa: E731
        engine,
        model,
        problem,
        trainable=trainable,
        steps=steps,
        optimizer=cfg.refinement.optimizer.name,
        lr=cfg.refinement.optimizer.lr,
        logger=NULL_LOGGER,
    )

    print(f"warmup 1 step on {args.device} ...", flush=True)
    run(1)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()

    print(f"profiling {args.steps} steps ...", flush=True)
    activities = [ProfilerActivity.CPU]
    if args.device.startswith("cuda"):
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, record_shapes=True) as prof:
        run(args.steps)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

    sort_key = "cuda_time_total" if args.device.startswith("cuda") else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=25), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

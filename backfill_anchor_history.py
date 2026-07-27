"""Backfill anchor-history.csv from historical main commits.

One-off, resumable. For every first-parent main commit since the frozen-checkpoint
anchor landed (3e18991, 2026-07-08), run `test_quartz_coupled_anchor` in a detached
worktree with a conftest shim that records the measured mean_r_obs, and append one
CSV row. Commits where the measurement fails get a gap row (empty value) so the
plot can break the line rather than interpolate.

Usage (from a checkout of the `badges` branch, with the main repo as a sibling):

    python backfill_anchor_history.py --repo /path/to/diffBloch [--limit N]

Requires: git-lfs (run `git lfs fetch origin --all` in the repo first), uv.
The e2e anchor reuses each commit's own committed plan.npz/plan.lock checkpoint,
so per-commit cost is dominated by `uv sync` — a fresh venv is only built when
uv.lock/pyproject.toml changed since the previous commit.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

WINDOW_START = "3e18991"  # first commit with the frozen-checkpoint coupled anchor
CSV_PATH = Path(__file__).parent / "anchor-history.csv"
CSV_HEADER = ["date", "sha", "mean_r_obs"]
PYTEST_TIMEOUT_S = 300

# Written into the worktree root; pytest imports root conftest before test modules,
# so patching the module attribute intercepts the test's `from ... import run_experiment`.
CONFTEST_SHIM = """\
import json
import os

import diffBloch.app.program as _prog

_orig_run_experiment = _prog.run_experiment


def _recording_run_experiment(*args, **kwargs):
    result = _orig_run_experiment(*args, **kwargs)
    out = os.environ.get("ANCHOR_BACKFILL_OUT")
    if out:
        try:
            with open(out, "a") as fh:
                fh.write(json.dumps({"mean_r_obs": float(result.mean_r_obs)}) + "\\n")
        except Exception:
            pass
    return result


_prog.run_experiment = _recording_run_experiment
"""


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None,
        timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=cwd, env=env, timeout=timeout,
        capture_output=True, text=True, check=False,
    )


def first_parent_shas(repo: Path) -> list[str]:
    proc = run(["git", "rev-list", "--first-parent", "--reverse",
                f"{WINDOW_START}^..origin/main"], cwd=repo)
    proc.check_returncode()
    return proc.stdout.split()


def commit_date(repo: Path, sha: str) -> str:
    proc = run(["git", "show", "-s", "--format=%cI", sha], cwd=repo)
    proc.check_returncode()
    return proc.stdout.strip()


def existing_shas() -> set[str]:
    if not CSV_PATH.exists():
        return set()
    with CSV_PATH.open() as fh:
        return {row["sha"] for row in csv.DictReader(fh)}


def append_row(date: str, sha: str, value: str) -> None:
    new_file = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as fh:
        # LF, not the csv default CRLF: CI appends rows with plain `echo`, and the file
        # must not end up with mixed line endings.
        writer = csv.writer(fh, lineterminator="\n")
        if new_file:
            writer.writerow(CSV_HEADER)
        writer.writerow([date, sha, value])


def lockfiles_changed(repo: Path, prev: str | None, sha: str) -> bool:
    if prev is None:
        return True
    proc = run(["git", "diff", "--quiet", prev, sha, "--",
                "uv.lock", "pyproject.toml"], cwd=repo)
    return proc.returncode != 0


def measure(worktree: Path, venv: Path) -> str | None:
    """Run the coupled anchor with the shim; return mean_r_obs as str, or None."""
    (worktree / "conftest.py").write_text(CONFTEST_SHIM)
    out_path = worktree / "anchor-backfill-out.jsonl"
    out_path.unlink(missing_ok=True)
    env = os.environ | {
        "UV_PROJECT_ENVIRONMENT": str(venv),
        "ANCHOR_BACKFILL_OUT": str(out_path),
    }

    sync = run(["uv", "sync", "--dev", "--frozen", "-q"], cwd=worktree, env=env)
    if sync.returncode != 0:  # frozen sync can fail on odd historical states
        sync = run(["uv", "sync", "--dev", "-q"], cwd=worktree, env=env)
        if sync.returncode != 0:
            print(f"    uv sync failed: {sync.stderr.strip().splitlines()[-1:]}")
            return None

    try:
        test = run(
            ["uv", "run", "pytest", "tests/e2e/test_anchor.py",
             "-m", "e2e", "-k", "coupled and not full", "-q", "--no-header"],
            cwd=worktree, env=env, timeout=PYTEST_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        print("    pytest timed out")
        return None

    if not out_path.exists():
        tail = (test.stdout + test.stderr).strip().splitlines()[-3:]
        print(f"    no measurement written; pytest rc={test.returncode}; tail={tail}")
        return None
    # Last line wins (the coupled test calls run_experiment once, but be safe).
    value = json.loads(out_path.read_text().strip().splitlines()[-1])["mean_r_obs"]
    if test.returncode != 0:
        print(f"    NOTE: measured {value} but pytest rc={test.returncode} (drifted value?)")
    return repr(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True,
                        help="path to the diffBloch main checkout")
    parser.add_argument("--limit", type=int, default=None,
                        help="process at most N unprocessed commits (smoke runs)")
    args = parser.parse_args()
    repo = args.repo.resolve()

    shas = first_parent_shas(repo)
    done = existing_shas()
    todo = [s for s in shas if s not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(shas)} commits in window, {len(done)} already recorded, "
          f"processing {len(todo)}")

    tmp = Path(tempfile.mkdtemp(prefix="anchor-backfill-"))
    worktree = tmp / "wt"
    venv = tmp / "venv"
    run(["git", "worktree", "add", "--detach", str(worktree), "origin/main"],
        cwd=repo).check_returncode()
    try:
        prev: str | None = None
        for i, sha in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {sha[:9]}")
            run(["git", "checkout", "--force", "--detach", sha],
                cwd=worktree).check_returncode()
            run(["git", "clean", "-fdx", "--exclude=anchor-backfill-out.jsonl"],
                cwd=worktree)
            if lockfiles_changed(repo, prev, sha):
                print("    lockfile changed -> fresh venv")
                shutil.rmtree(venv, ignore_errors=True)
            value = None
            try:
                value = measure(worktree, venv)
            except Exception as exc:  # noqa: BLE001 - one bad commit must not stop the sweep
                print(f"    unexpected failure: {exc}")
            append_row(commit_date(repo, sha), sha, value if value is not None else "")
            print(f"    -> {value if value is not None else 'GAP'}")
            prev = sha
    finally:
        run(["git", "worktree", "remove", "--force", str(worktree)], cwd=repo)
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

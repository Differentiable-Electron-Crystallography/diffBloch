"""Refuse a version bump whose committed locks were not refreshed alongside it.

A release here is an ordinary PR: the only signal is the ``__version__`` line in
``src/diffBloch/__init__.py``, and merging that PR is what publishes to PyPI. That makes this check
the thing standing between a bump and a broken release, because the committed locks record the code
version their results were produced with, and ``preprocess_lock_status`` keys checkpoint reuse on
it. Ship a new version without re-running them and the published release advertises results it
cannot reproduce.

Covers the e2e fixture as well as the examples. Its plan lock and ``.npz`` are committed on purpose
(see .gitignore) so the anchor job can skip the expensive fit; left stale after a bump, that job
recomputes the plan on every run from then on, not just once. Note that the fixture is refreshed by
``diffbloch preprocess --refresh``, NOT by running the e2e suite: the only e2e test loads the
fixture and compares inference against ``reference_results.json``, so it never writes a plan lock
and can never clear this check.

Runs on every PR and no-ops when ``__version__`` is untouched -- it is a required status check, so
it must always report rather than be skipped by a ``paths:`` filter.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import yaml
from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parents[2]
INIT = "src/diffBloch/__init__.py"

# Where committed locks live, and the command that regenerates each family's.
LOCK_ROOTS = {
    "examples": "uv run diffbloch refine <experiment_dir> --refresh",
    # NOT `pytest -m e2e` -- that suite only reads the fixture (see module docstring). Only the
    # preprocess stage writes plan.<stem>.npz/.lock, and only --refresh rebuilds them; `refine`
    # would also work but would leave refinement artifacts the fixture is not supposed to carry.
    "tests/fixtures": "uv run diffbloch preprocess tests/fixtures/quartz_anchor --refresh",
}
VERSION_RE = re.compile(r'^__version__ = "([^"]+)"', re.MULTILINE)

# Mirrors `_release` in diffBloch/config/manifest.py: code_version is "<version>+g<sha>[.dirty]" and
# the reuse gate keys on the release part alone. Duplicated rather than imported because that module
# pulls pydantic and the config schema, which would turn a seconds-long guard into a full `uv sync`.
RELEASE_SUFFIX = "+g"


def git(*args: str) -> str:
    """Run a git command in the repo root and return its stdout, or exit with its stderr."""
    done = subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        sys.exit(f"::error::git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout


def parse_version(source: str, origin: str) -> str:
    """Pull the `__version__` string out of the text of an `__init__.py`."""
    found = VERSION_RE.search(source)
    if found is None:
        sys.exit(f"::error::no __version__ assignment found in {origin}")
    return found.group(1)


def pep440(raw: str, origin: str) -> Version:
    """Parse `raw` as a PEP 440 version, naming `origin` if it is not one."""
    try:
        return Version(raw)
    except InvalidVersion:
        sys.exit(f"::error::{origin} is not a PEP 440 version: {raw}")


def stale_locks(release: str, root: str) -> list[tuple[Path, str]]:
    """Every lock under `root` whose recorded code version is not `release`.

    Keys on the presence of the `code_version` field rather than on filenames: the `experiment.lock`
    files identify input bytes only and carry no code axis, so a version bump legitimately leaves
    them alone.

    Parsed as YAML because the two lock families are written in different formats -- plan and
    refinement locks are JSON (`model_dump_json`), experiment locks are YAML and read back with
    `yaml.safe_load` (manifest.py:172). JSON is valid YAML, so one loader covers both.
    """
    stale = []
    for lock in sorted((ROOT / root).rglob("*.lock")):
        try:
            parsed = yaml.safe_load(lock.read_text())
        except (OSError, yaml.YAMLError) as exc:
            sys.exit(f"::error::{lock.relative_to(ROOT)} is not readable: {exc}")
        if not isinstance(parsed, dict):
            sys.exit(f"::error::{lock.relative_to(ROOT)} is not a lock mapping")
        recorded = parsed.get("code_version")
        if recorded is None:
            continue
        if recorded.split(RELEASE_SUFFIX, 1)[0] != release:
            stale.append((lock.relative_to(ROOT), recorded))
    return stale


def highest_released() -> Version | None:
    """The greatest version carried by a `v*` tag, or None when nothing has been released.

    Tags that are not PEP 440 versions are ignored rather than fatal: the release path only ever
    creates `v<version>`, but a hand-made tag like `v1-paper-submission` should not break the guard.
    """
    released = []
    for line in git("tag", "--list", "v*").splitlines():
        raw = line.strip().removeprefix("v")
        if not raw:
            continue
        try:
            released.append(Version(raw))
        except InvalidVersion:
            continue
    return max(released, default=None)


def summary(text: str) -> None:
    """Append to the job summary when running under Actions; harmless locally."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a") as fh:
            fh.write(text)


def main() -> None:
    base_ref = os.environ.get("BASE") or os.environ.get("GITHUB_BASE_REF")
    if not base_ref:
        sys.exit("::error::BASE (or GITHUB_BASE_REF) must name the branch this PR targets")

    head = parse_version((ROOT / INIT).read_text(), INIT)
    base = parse_version(git("show", f"origin/{base_ref}:{INIT}"), f"origin/{base_ref}:{INIT}")

    if head == base:
        print(f"__version__ is unchanged at {head}; not a release PR")
        return

    head_v = pep440(head, "__version__")

    tag = f"v{head}"
    if git("tag", "--list", tag).strip():
        sys.exit(f"::error::{tag} already exists -- {head} has been released; choose a new version")

    # Regression is measured against what has actually been RELEASED, not against the base branch's
    # `__version__`. That line is only a draft until a release publishes it, and comparing to it
    # forbids a legitimate move that PEP 440 orders downwards: an unreleased 0.2.0 becoming its own
    # release candidate 0.2.0rc1, since rc sorts below the final. Tags are the record of what
    # shipped (release.yml writes one only after PyPI accepts the upload), so with nothing released
    # there is nothing to regress below.
    if (highest := highest_released()) is not None and head_v <= highest:
        sys.exit(f"::error::__version__ {head} is not above the last release {highest}")

    failed = False
    for root, remedy in LOCK_ROOTS.items():
        stale = stale_locks(head, root)
        if not stale:
            continue
        failed = True
        print(f"::error::__version__ is {head} but {len(stale)} lock(s) under {root}/ record older")
        for path, recorded in stale:
            print(f"::error file={path}::{path} records code_version {recorded}, expected {head}")
        print(f"::error::refresh them with: {remedy}")
    if failed:
        sys.exit(1)

    print(f"release PR: {base} -> {head}, all committed locks refreshed")
    summary(f"Release PR: **{base} -> {head}** — committed locks refreshed, `{tag}` free\n")


if __name__ == "__main__":
    main()

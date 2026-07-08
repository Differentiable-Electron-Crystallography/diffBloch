"""End-to-end checkpoint/resume through ``run_experiment`` on the quartz anchor.

Proves the full path: a first ``run_experiment`` computes the settled ``Plan`` and writes
``plan.npz`` + ``plan.lock`` into the (copied) experiment dir; a second identical run is a full
**reuse** -- it loads the checkpoint and skips the expensive preprocess entirely, yielding a
byte-identical aggregate. Reuse is asserted from the emitted diagnostic log (deterministic), not
timing. Opt-in ``e2e`` (the first run does the full fit); the driver's reuse/resume/stale logic is
pinned fast in ``tests/unit/test_program_checkpoint.py``.
"""

import logging
import shutil
from pathlib import Path

import pytest

from diffBloch.app.program import run_experiment

pytestmark = pytest.mark.e2e

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "quartz_anchor"
_INPUTS = (
    "experiment.yaml",
    "experiment.lock",
    "enantiomer_1.cif",
    "exp_data.cif_pets",
)


def test_run_experiment_checkpoints_then_reuses(tmp_path: Path, caplog) -> None:
    exp = tmp_path / "quartz"
    exp.mkdir()
    for name in _INPUTS:
        shutil.copy(FIXTURE_ROOT / name, exp / name)

    first = run_experiment(exp)
    assert (exp / "plan.npz").exists() and (exp / "plan.lock").exists()

    with caplog.at_level(logging.INFO, logger="diffBloch.app.program"):
        second = run_experiment(exp)

    assert "full reuse" in caplog.text  # the second run loaded the checkpoint, did not re-fit
    assert second.n_evaluated == first.n_evaluated
    assert second.mean_r_obs == pytest.approx(first.mean_r_obs, abs=1e-12)

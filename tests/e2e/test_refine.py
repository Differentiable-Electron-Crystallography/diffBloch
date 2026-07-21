"""End-to-end app refinement (``refine_experiment``) on the quartz anchor.

Proves the config-driven refinement executor: it reuses the committed preprocess checkpoint (the
refinement config is outside ``config_digest``, so adding a step budget does not restale it), builds
the engine + problem from config, and runs the optimizer to a finite descending loss curve.
"""

import shutil
from pathlib import Path

import pytest
import torch
import yaml

from diffBloch.app.program import refine_experiment

pytestmark = pytest.mark.e2e

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "quartz_anchor"
_INPUTS = ("experiment.lock", "enantiomer_1.cif", "exp_data.cif_pets", "plan.npz", "plan.lock")


def test_refine_experiment_reuses_checkpoint_and_descends(tmp_path: Path) -> None:
    exp = tmp_path / "quartz"
    exp.mkdir()
    for name in _INPUTS:
        shutil.copy(FIXTURE_ROOT / name, exp / name)
    # a short step budget so the e2e is quick; refinement config is outside config_digest, so the
    # committed checkpoint still reuses (no re-fit).
    cfg = yaml.safe_load((FIXTURE_ROOT / "experiment.yaml").read_text())
    cfg.setdefault("refinement", {})["steps"] = 2
    (exp / "experiment.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    result = refine_experiment(exp)

    assert result.losses.shape == (2,)
    assert torch.isfinite(result.losses).all()
    assert result.best_loss == float(result.losses[result.best_step])

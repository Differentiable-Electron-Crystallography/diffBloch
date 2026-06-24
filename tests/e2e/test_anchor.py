"""North-star characterization anchor: one Bloch simulation of one quartz rotation.

Migration stage 2 (see ``ROADMAP.md``) makes the fixture self-contained: the real quartz input
files and private reference metrics are present in this package and hash-verified. Later stages make
the final physics assertions executable. When ported it will pin, on a fixed seed / CPU / float64:

  * ``R_obs`` for the single rotation, and
  * the intermediates ``Fgb``, the structure matrix ``A``, the exit wave ``psi``, and ``I_sim``.

Until the ``io/`` + ``core/`` kernels are ported, only the physics execution is skipped. Fixture
discovery, lock verification, and reference metadata checks run now.
"""

import json
from pathlib import Path

import pytest

from diffBloch.config import load_experiment, sha256_file

pytestmark = pytest.mark.e2e

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "quartz_anchor"


@pytest.mark.parametrize("material", ["quartz"])
def test_single_rotation_anchor(material: str) -> None:
    assert material == "quartz"

    cfg, lock = load_experiment(FIXTURE_ROOT)
    assert cfg.name == "quartz-anchor"
    assert cfg.solver.inference == "bloch_eigen"
    assert cfg.numerics.sg_max == 0.01
    assert cfg.inputs.structure == lock.structure.ref
    assert cfg.inputs.observations == lock.observations.ref
    assert cfg.inputs.orientations == lock.orientations.ref

    manifest = json.loads((FIXTURE_ROOT / "anchor_manifest.json").read_text())
    reference = json.loads((FIXTURE_ROOT / manifest["reference_results"]["path"]).read_text())
    assert (
        sha256_file(FIXTURE_ROOT / "reference_results.json")
        == manifest["reference_results"]["sha256"]
    )
    assert reference["seed"] == manifest["execution"]["seed"]
    assert reference["n_rotations"] == manifest["reference_results"]["n_rotations"]
    assert reference["N_int_all"] == manifest["reference_results"]["N_int_all"]
    assert reference["N_int_obs"] == manifest["reference_results"]["N_int_obs"]
    assert reference["summary"]["R_obs"] == pytest.approx(
        manifest["reference_results"]["summary"]["R_obs"]
    )

    for tensor in ("Fgb", "A", "psi", "I_sim"):
        assert manifest["intermediate_tensors"][tensor]["status"] == "pending"

    pytest.skip("pending physics execution — IO/core kernels land in migration stages 3-8")

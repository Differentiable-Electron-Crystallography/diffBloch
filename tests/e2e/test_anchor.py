"""North-star characterization anchor: one Bloch simulation of one quartz rotation.

Migration stage 0 (see ``ROADMAP.md``). This test pins the forward-model physics so every subsequent
extraction can assert "the core physics model has not changed". When ported it will pin, on a fixed
seed / CPU / float64:

  * ``R_obs`` for the single rotation, and
  * the intermediates ``Fgb``, the structure matrix ``A``, the exit wave ``psi``, and ``I_sim``.

Until the ``io/`` + ``core/`` kernels are ported, this is a skipped placeholder so the e2e
harness is wired and green from the first commit.
"""

import pytest

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize("material", ["quartz"])
def test_single_rotation_anchor(material: str) -> None:
    pytest.skip("pending port — migration stage 0 (see ROADMAP.md)")

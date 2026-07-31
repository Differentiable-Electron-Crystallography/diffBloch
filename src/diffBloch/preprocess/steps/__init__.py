"""The composable ``Plan -> Plan`` steps -- the swappable units the pipeline and driver compose.

Each module here produces a ``Plan -> Plan`` transform (or the small drivers/measures its adapters
use): :mod:`~diffBloch.preprocess.steps.beams` (``select_beams``),
:mod:`~diffBloch.preprocess.steps.convergence` / :mod:`~diffBloch.preprocess.steps.coverage` (the
two
convergence operations), :mod:`~diffBloch.preprocess.steps.optimize_orientation` /
:mod:`~diffBloch.preprocess.steps.optimize_thickness` (the accuracy fits), and
:mod:`~diffBloch.preprocess.steps.rocking_curve` (``integrate_rocking_curve``).

Steps depend only on the parent *spine* (``plan``, ``pipeline``, ``experiment``, ``scoring``,
``orientation``); the parent *orchestrators* (``pipeline`` composition, ``driver``) compose steps.
The public names are re-exported flat at :mod:`diffBloch.preprocess`.
"""

from __future__ import annotations

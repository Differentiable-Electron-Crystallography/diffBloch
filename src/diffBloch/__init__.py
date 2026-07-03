"""diffBloch — differentiable Bloch-wave electron-diffraction structure refinement."""

import logging as _logging

# Diagnostics channel (design/decisions/effects-and-observability.md): the library installs no
# reporter -- just a NullHandler at the package root so importing diffBloch never emits "No handlers
# could be found" and no records surface unless the app (or a probe) configures a handler.
_logging.getLogger("diffBloch").addHandler(_logging.NullHandler())

__version__ = "0.2.0"

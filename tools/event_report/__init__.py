"""Consumers of the diffBloch JSONL event report: a shared reader, figures, and an HTML renderer.

Not part of the refinement library -- this package reads the event contract that
``src/diffBloch`` emits and decides how to present it. Rendering, plotting, and image export live
here and nowhere else.
"""

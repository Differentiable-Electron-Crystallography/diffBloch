"""Sphinx configuration for the diffBloch documentation.

Markdown sources are rendered by MyST; the API reference is generated from the package's
Google-style docstrings via autodoc + napoleon. Theme is furo.
"""

from __future__ import annotations

import importlib.metadata

# -- Project information -----------------------------------------------------
project = "diffBloch"
author = "diffBloch contributors"
copyright = "2026, diffBloch contributors"  # noqa: A001 (Sphinx expects this name)
release = importlib.metadata.version("diffBloch")
version = release

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",  # Markdown sources
    "sphinx.ext.autodoc",  # pull docstrings from the package
    "sphinx.ext.napoleon",  # parse Google-style docstrings
    "sphinx.ext.autosummary",  # summary tables / stub support
    "sphinx.ext.intersphinx",  # link external types (replaces mkdocstrings inventories)
    "sphinx.ext.viewcode",  # "source" links, mirroring the old show_source
    "sphinx_copybutton",  # copy button on code blocks
    "sphinxext.opengraph",  # Open Graph meta tags
    "sphinx_inline_tabs",  # tabbed content
]

# The docs tree holds only the authored pages; conf.py itself and build output are not sources.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Public records/classes are re-exported from package ``__init__``s (e.g. ``diffBloch.io`` and
# ``diffBloch.io.record`` both surface ``ObservationRecord``), so a bare cross-reference has more
# than one valid target. That ambiguity is benign; silence it so ``-W`` still fails on real
# structural problems (missing toctree entries, malformed directives). Unresolved refs stay silent
# anyway under the default ``nitpicky = False``.
suppress_warnings = ["ref.python"]

# -- MyST (Markdown) ---------------------------------------------------------
# ``colon_fence`` lets admonitions/directives use ::: fences; ``deflist`` for definition lists.
myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]
myst_heading_anchors = 3  # auto-slug headings so intra-page links resolve

# -- Autodoc / napoleon ------------------------------------------------------
# Centralised so the ``.. automodule::`` blocks in api/*.md stay bare.
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
    # Internal NDArray type aliases repeated per-module; no doc value, and documenting each creates
    # ambiguous ``FloatArray``/``IntArray`` cross-reference targets.
    "exclude-members": "FloatArray,IntArray",
}
autodoc_typehints = "signature"  # keep annotations in the signature (separate_signature analog)
autodoc_member_order = "bysource"
autodoc_preserve_defaults = True
napoleon_google_docstring = True
napoleon_numpy_docstring = False
autosummary_generate = True

# -- Intersphinx (external type cross-links) ---------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}

# -- HTML output (furo) ------------------------------------------------------
html_theme = "furo"
html_title = "diffBloch"
html_baseurl = "https://differentiable-electron-crystallography.github.io/diffBloch/"
html_static_path: list[str] = []
ogp_site_url = html_baseurl

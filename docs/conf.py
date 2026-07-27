"""Sphinx configuration for the diffBloch documentation.

Markdown sources are rendered by MyST; the API reference is generated from the package's
Google-style docstrings via autodoc + napoleon. Theme is furo.
"""

from __future__ import annotations

import importlib.metadata
import inspect
import subprocess
from pathlib import Path

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
    "sphinx.ext.viewcode",  # local rendered source pages, mirroring the old show_source
    "sphinx.ext.linkcode",  # API source links to the exact GitHub commit
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

# -- Source links -------------------------------------------------------------
_REPO_URL = "https://github.com/Differentiable-Electron-Crystallography/diffBloch"
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _source_ref() -> str:
    """Return the exact git commit for source links, falling back to ``main`` outside git."""
    try:
        result = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return "main"
    return result.stdout.strip() or "main"


_SOURCE_REF = _source_ref()


def linkcode_resolve(domain: str, info: dict[str, str]) -> str | None:
    """Link API objects to their source lines on GitHub at the current commit."""
    if domain != "py" or not info.get("module"):
        return None

    try:
        module = __import__(info["module"], fromlist=["_"])
        obj = module
        for part in info.get("fullname", "").split("."):
            obj = getattr(obj, part)
        source_file = inspect.getsourcefile(obj)
        if source_file is None:
            return None
        source_lines, start = inspect.getsourcelines(obj)
        rel = Path(source_file).resolve().relative_to(_REPO_ROOT)
    except (AttributeError, OSError, TypeError, ValueError):
        return None

    end = start + len(source_lines) - 1
    return f"{_REPO_URL}/blob/{_SOURCE_REF}/{rel.as_posix()}#L{start}-L{end}"


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

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
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/logo.png"
ogp_site_url = html_baseurl

html_theme_options = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": _REPO_URL,
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

templates_path = ["_templates"]

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
#
# Make the ``zitterwalk`` package importable by autodoc. ``conf.py`` lives in
# ``docs/source/``; the package sits two levels up, at the repository root.

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../.."))

# Headless backend so the plot_directive and any doctest that touches
# matplotlib never try to open a window on the build machine.
import matplotlib  # noqa: E402

matplotlib.use("Agg")

import zitterwalk  # noqa: E402

# -- Project information -----------------------------------------------------

project = "ZitterWalk"
author = "Germán Godoy Gutiérrez"
copyright = f"{datetime.now():%Y}, {author}"
version = zitterwalk.__version__
release = zitterwalk.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",  # writes .nojekyll so GitHub Pages serves _static/_images
    "myst_parser",
    "matplotlib.sphinxext.plot_directive",
]

templates_path = ["_templates"]
exclude_patterns = []

# Accept both reStructuredText and Markdown sources (MyST).
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Enable a few handy MyST features (dollar-math for LaTeX in Markdown, etc.).
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence"]

# -- autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# -- napoleon ----------------------------------------------------------------
#
# The source docstrings are written in Google style ("Args:", "Returns:",
# "Raises:"); napoleon renders them as clean parameter tables.

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
# Render "Attributes:" sections as inline :ivar: fields instead of standalone
# attribute directives. Without this, slotted classes (Node, Edge) get each
# attribute documented twice -- once by the docstring section and once by
# autodoc's member scan -- which Sphinx reports as a duplicate object.
napoleon_use_ivar = True

# -- intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- doctest -----------------------------------------------------------------
#
# Every ``>>>`` block in the narrative pages is run by ``make doctest`` with the
# names below already imported, so the examples read like a real session.

doctest_global_setup = """
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from zitterwalk import Graph, Walker, DiscreteTimeWalk, viz
from zitterwalk.coin import (
    hadamard, grover, fourier, rotation, su2,
    build_coin_operator, random_coins, marked_coins,
)
"""

doctest_global_cleanup = """
import matplotlib.pyplot as plt
plt.close("all")
"""

# Be lenient about insignificant whitespace and long float tails so the
# examples stay readable without brittle output matching.
import doctest as _doctest  # noqa: E402

doctest_default_flags = (
    _doctest.NORMALIZE_WHITESPACE
    | _doctest.ELLIPSIS
    | _doctest.IGNORE_EXCEPTION_DETAIL
)

# -- matplotlib plot_directive ----------------------------------------------

plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [("png", 110)]
plot_rcparams = {"figure.autolayout": True}

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = f"ZitterWalk {release}"
html_logo = None

html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
}

# The custom footer template (``_templates/footer.html``) appends the review
# note to every page via the theme's ``extrafooter`` block.
html_show_sphinx = True
html_show_sourcelink = True

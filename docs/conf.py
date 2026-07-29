"""Sphinx configuration for Fair-Seldonian documentation."""

import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath("../src"))

import fair_seldonian

# -- Vendor example notebooks -----------------------------------------------
# The example notebooks live in ``examples/`` (outside the docs source tree).
# Copy them into ``docs/examples/`` at build time so Sphinx can render them,
# rather than committing duplicate copies. They ship with saved outputs and are
# rendered without re-execution (see ``nb_execution_mode`` below).

_docs_dir = Path(__file__).parent
_examples_dst = _docs_dir / "examples"
_examples_dst.mkdir(exist_ok=True)
for _nb in sorted((_docs_dir.parent / "examples").glob("*.ipynb")):
    shutil.copy2(_nb, _examples_dst / _nb.name)

# -- Project information -----------------------------------------------------

project = "Fair-Seldonian"
author = "Parul Gupta"
copyright = f"{datetime.now().year}, {author}"
version = ".".join(fair_seldonian.__version__.split(".")[:2])
release = fair_seldonian.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "myst_nb",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}
root_doc = "index"
language = "en"
pygments_style = "sphinx"

# -- Autodoc configuration --------------------------------------------------

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# -- Napoleon configuration -------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- Intersphinx configuration ----------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# -- HTML output configuration ----------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "prev_next_buttons_location": "both",
}
html_static_path = ["_static"]
html_show_sourcelink = True
html_show_copyright = True

# -- MyST / MyST-NB configuration -------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",  # render $...$ and $$...$$ math in notebook/markdown cells
]

# Render notebooks from their saved outputs; never execute at build time. The
# example notebooks require network access and heavy computation to run, and are
# committed with outputs already saved.
nb_execution_mode = "off"

# The notebooks link to sibling files in the repo's ``examples/`` dir (e.g.
# ``custom_constraint.py``) with relative paths that resolve on GitHub/nbviewer
# but not inside the rendered docs. Don't fail the build over those.
suppress_warnings = ["myst.xref_missing"]

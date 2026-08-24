"""Sphinx configuration for Fair-Seldonian documentation."""

import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath("../src"))
sys.path.insert(0, os.path.abspath("_ext"))

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
    "sphinx_design",
    "sphinx_copybutton",
    "fair_seldonian_docs",
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

# Fail fast rather than hanging on a slow mirror. Note that under ``-W`` an
# unreachable inventory still fails the build; Sphinx logs that warning with no
# subtype, so ``suppress_warnings`` cannot single it out.
intersphinx_timeout = 15


# -- Name the sidebar after the section ---------------------------------------


def _section_of_page(env, root_doc: str) -> dict[str, str]:
    """Map every document to the title of the top-level section containing it.

    The theme labels the left sidebar "Section Navigation" on every page, which
    says nothing: the reader already knows they are looking at navigation. The
    useful label is which section they are in, and the toctree already knows.

    An explicit toctree caption wins over the target page's own heading, so
    ``API reference <api/fair_seldonian>`` labels the sidebar "API reference"
    rather than "fair_seldonian package" -- matching what the navbar shows.
    """
    from sphinx import addnodes

    explicit: dict[str, str] = {}
    for node in env.get_doctree(root_doc).findall(addnodes.toctree):
        for title, docname in node["entries"]:
            if title:
                explicit[docname] = title

    labels: dict[str, str] = {}
    includes = env.toctree_includes
    for top in includes.get(root_doc, []):
        title = env.titles.get(top)
        label = explicit.get(top) or (title.astext() if title is not None else top)
        stack, seen = [top], set()
        while stack:
            doc = stack.pop()
            if doc in seen:
                continue
            seen.add(doc)
            labels[doc] = label
            stack.extend(includes.get(doc, []))
    return labels


def _set_section_title(app, pagename, templatename, context, doctree):
    # Reading the root doctree touches the filesystem, so resolve the whole map
    # once per build rather than once per page.
    labels = getattr(app.builder, "_fs_section_labels", None)
    if labels is None:
        labels = _section_of_page(app.builder.env, app.config.root_doc)
        app.builder._fs_section_labels = labels
    context["fs_section_title"] = labels.get(pagename)


def setup(app):
    app.connect("html-page-context", _set_section_title)
    return {"parallel_read_safe": True, "parallel_write_safe": True}


# -- HTML output configuration ----------------------------------------------

html_theme = "pydata_sphinx_theme"
_repo = "https://github.com/parulgupta1004/fair-seldonian"
html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_align": "left",
    "icon_links": [
        {"name": "GitHub", "url": _repo, "icon": "fa-brands fa-github"},
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/fair-seldonian/",
            "icon": "fa-brands fa-python",
        },
    ],
    "show_prev_next": True,
    "show_toc_level": 2,
    "use_edit_page_button": True,
    # Results appear as you type, in a modal, instead of requiring Enter and a
    # full page load of search.html. The index is already in memory by then --
    # see _templates/layout.html -- so the work per keystroke is a lookup.
    "search_as_you_type": True,
    # Expand the current section's pages rather than leaving them collapsed
    # behind a caret; there are only four or five per section.
    "show_nav_level": 2,
    "navigation_depth": 3,
    # Breadcrumbs matter more than usual here because the top navbar shows the
    # section, not the page, so a deep-linked reader has no other cue to where
    # they are.
    "article_header_start": ["breadcrumbs"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
}

# Left sidebar. The default is ``sidebar-collapse`` + ``sidebar-nav-bs``, which
# on a page with no child pages renders an empty "Section Navigation" heading --
# which is what every concept page did before they were grouped under
# ``concepts``. There is deliberately no search box here: the navbar already
# carries one at the top right, and two search affordances on the same page is
# one too many. ``indices`` is added because the rebuilt landing page no longer
# carries the genindex/modindex links it used to.
html_sidebars = {
    "**": [
        "sidebar-nav-bs",
        "indices",
    ],
    # The landing page carries its own card navigation.
    "index": [],
}

# Right sidebar: the page outline, plus the two links that act on the page.
html_theme_options["secondary_sidebar_items"] = [
    "page-toc",
    "edit-this-page",
    "sourcelink",
]
html_context = {
    "github_user": "parulgupta1004",
    "github_repo": "fair-seldonian",
    "github_version": "master",
    "doc_path": "docs",
    # Respect the reader's OS setting rather than forcing one mode.
    "default_mode": "auto",
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
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

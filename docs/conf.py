# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import datetime
import sys
sys.path.insert(0, os.path.abspath('../src/gensec'))
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('./extensions'))

# -- Project information -----------------------------------------------------

project = 'gensec'
copyright = '2026, Andrea Albero'
author = 'Andrea Albero'
show_authors = True

# Use the package version if available
try:
    from gensec import __version__ as version
except Exception:  # pragma: no cover - docs build fallback
    version = "0.0.0"
release = version

# -- Graph configuration -----------------------------------------------------

#
# inheritance_graph_attrs = dict(rankdir="TB", size='"6.0, 8.0"',
#                                fontsize=14, ratio='compress',
#                                orientation='L')
#
# inheritance_node_attrs = dict(shape='ellipse', fontsize=14, height=0.75,
#                               color='dodgerblue1', style='filled',
#                               orientation='L')#

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.mathjax',
    #'sphinx.ext.imgmath',
    #'sphinx.ext.pngmath',
    'sphinx.ext.napoleon',
    #'sphinx.ext.graphviz',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.githubpages',
    'sphinxcontrib.mermaid',
    #'sphinx.ext.viewcode',
    #'sphinxcontrib.mermaid',
    #'sphinx_search.extension',
    #'extensions.mycustomlang',
    #'sphinx.ext.intersphinx',
    "sphinx_multiversion",
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'shapely': ('https://shapely.readthedocs.io/en/latest/', None),
}

add_module_names = False
python_display_short_literal_types = True
autodoc_inheritdocstrings = False
autodoc_typehints = 'signature'
autodoc_typehints_description_target = 'documented'
autodoc_typehints_format = 'short'
autodoc_preserve_defaults = True

autodoc_default_options = {
    #'members': True,
    #'undoc-members': True,
    #'private-members': False,
    #'special-members': False,
    #'show-inheritance': True,
    #'inherited-members': True,
    #'exclude-members': '__init__',
    'no-source': True  # Include or exclude code from documentation
}

maximum_signature_line_length = 5
toc_object_entries_show_parents = 'hide'

# Napoleon settings
napoleon_googledocstring = False
napoleon_numpydocstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_custom_sections = None
napoleon_attr_annotations = False
numpydoc_attributes_as_param_list = False

# modindex_common_prefix = [
#     '',
# ]

# Local copy of js for mermaid
mermaid_js_url = "static/mermaid.min.js"
mermaid_init_js = "mermaid.initialize({startOnLoad:true, securityLevel:'loose'});"

# Nitpick error: see -->
# https://stackoverflow.com/questions/11417221/sphinx-autodoc-gives-warning-
#   pyclass-reference-target-not-found-type-warning
nitpick_ignore = [('py:class', 'type')]

# -- LaTeX -------------------------------------------------------------------

# See https://tex.stackexchange.com/questions/122407/writing-conditional-
#     equations-with-braces-in-sphinx
#imgmath_latex_preamble = r'\usepackage{array}'
#imgmath_image_format = 'svg'
#imgmath_font_size = 14
latex_engine = 'pdflatex'
# latex_table_style = ['booktabs', 'borderless']

### To work with TeXmaker
# imgmath_latex='C:\Program Files\PascalBrachet\\5.0.4\\texmaker.exe'
# latex_engine = 'pandoc'
# imgmath_latex='C:\Program Files\Pandoc\Pandoc\\2.6\pandoc.exe'
# latex_engine = 'xelatex'

# -- HTML---------------------------------------------------------------------

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects htmlstatic_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#html_theme = 'sphinx_rtd_theme'
html_theme = "furo"

# _docs_assets may not be present in git-archive checkouts used by
# sphinx-multiversion. Guard the import so conf.py always loads cleanly.
try:
    from importlib.resources import files as _ires_files
    import gensec._docs_assets as _docs_assets_pkg
    html_static_path = [
        "static",
        str(_ires_files(_docs_assets_pkg) / "static"),
    ]
except (ImportError, ModuleNotFoundError):
    html_static_path = ["static"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

#html_extra_path = [
#    #'../../requirements.txt',
#    #'../requirements.txt',
#    #'../src/config.py',
#    '../src/gensec/config.py',
#    #'requirements.txt',
#    "../qualification",  # relative to conf.py
#    "../data/qualiffiles",  # relative to conf.py
#    ]

html_favicon = '../src/gensec/_docs_assets/static/logo/logo_alone.png'
html_logo = '../src/gensec/_docs_assets/static/logo/logolight.png'
html_title = 'GenSec Documentation'
html_short_title = 'GenSec Doc'

html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/versioning.html",
        "sidebar/navigation.html",
        "sidebar/scroll-end.html",
    ]
}
# For RTD Template
#html_theme_options = {
#    'logo_only': True,
#    #'display_version': False,
#    #'search_field': True,
#}

# ---------------------------------------------------------------------------
# sphinx-multiversion configuration
# ---------------------------------------------------------------------------

# NOTE: documentation is versioned by minor (X.Y), not patch (X.Y.Z)
# Tags are intentionally NOT used for documentation builds
smv_branch_whitelist = r'^(doc/\d+\.\d+)$'
smv_tag_whitelist = r"$^"
smv_remote_whitelist = None
smv_prefer_remote_refs = False
# Auto-detect latest: highest doc/X.Y local branch. No manual update needed.
import re as _re
try:
    import subprocess as _sp
    _raw = _sp.check_output(
        ["git", "branch", "--list", "doc/*"],
        text=True, stderr=_sp.DEVNULL,
    ).splitlines()
    _doc_branches = sorted(
        (b.strip().lstrip("* ") for b in _raw
         if _re.match(r"^doc/\d+\.\d+$", b.strip().lstrip("* "))),
        key=lambda v: tuple(int(x) for x in v[4:].split(".")),
    )
    smv_latest_version = _doc_branches[-1] if _doc_branches else ""
except Exception:
    smv_latest_version = ""

# For furo Template
html_theme_options = {
    "sidebar_hide_name": True,  # or True if logo already contains the name
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#7C4DFF",
        "color-brand-content": "#7C4DFF",
    },
    "dark_css_variables": {
        "color-brand-primary": "#7C4DFF",
        "color-brand-content": "#7C4DFF",
    },
}

html_context = {
    "author": "Andrea ALBERO",
    "date": datetime.date.today().strftime("%d/%m/%y"),
}

# =============================================================================
# Non-destructive hardening block (append-only)
# Keeps your original conf.py intact; sets/adjusts only when missing
# =============================================================================

from pathlib import Path as _Path

try:
    _CONF_DIR = _Path(__file__).parent.resolve()
except Exception:
    import os as _os
    _CONF_DIR = _Path(_os.path.dirname(__file__)).resolve()

_REPO_ROOT = _CONF_DIR.parent

# -- Path hygiene: add local sources and extensions only if not already present
for _p in [
    _REPO_ROOT / "src",
    _REPO_ROOT / "src" / "gensec",
    _CONF_DIR,                      # so "extensions.mycustomlang" resolves if package
    _CONF_DIR / "extensions",       # if extensions/ is a plain folder
]:
    if _p.exists():
        _sp = str(_p)
        if _sp not in sys.path:
            sys.path.insert(0, _sp)

# -- Language default (kept minimal; set explicitly if you want translations)
if "language" not in globals():
    language = "en"

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import datetime

project = 'Metaseq'
copyright = f'{datetime.datetime.now().year} , Meta'
author = 'Meta Research'

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from glob import glob
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath('../..'))

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    'sphinx_copybutton',
    "sphinx_design",
    "sphinxcontrib.mermaid",
]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

suppress_warnings = ["myst.strikethrough"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_title = "Metaseq Documentation"
html_theme_options = {
    "source_repository": "https://github.com/facebookresearch/metaseq",
    "source_branch": "main",
    "source_directory": "docs/source",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ["custom.css"]

# html_logo = "_static/images/logo.svg"
# html_favicon = '_static/images/logo.svg'

autodoc_typehints = 'none'
autodoc_member_order = 'groupwise'
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'exclude-members': '__weakref__'
}

# allows us to automatically link to documentation from other libraries
intersphinx_mapping = {
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
    "python": ("https://docs.python.org/3.8", None),
    "torch": ("https://pytorch.org/docs/master/", None),
}

# -- LaTeX output -------------------------------------------------

latex_engine = "xelatex"

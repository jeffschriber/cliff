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
import sys
sys.path.insert(0, os.path.abspath('../'))
import cliff


# -- Project information -----------------------------------------------------

project = 'CLIFF'
copyright = '2020, Jeffrey B. Schriber'
author = 'Jeffrey B. Schriber'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # from Sphinx
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.extlinks',
    'sphinx.ext.graphviz',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.graphviz',
    "sphinx_autodoc_typehints",
    "sphinx.ext.githubpages",
    # from Astropy
    'sphinx_automodapi.automodapi',
    'sphinx_automodapi.automodsumm',
    'sphinx_automodapi.smart_resolver',
    # from Cloud
#    'cloud_sptheme.ext.index_styling',
#    'cloud_sptheme.ext.escaped_samp_literals',
#    # from Psi4
#    'sphinx_psi_theme.ext.psidomain',
#    'sphinx_psi_theme.ext.relbar_toc',
]

autosummary_generate = True
automodapi_toctreedirnm = 'api'
#numpydoc_show_class_members = False
#automodsumm_inherited_members = True
autodoc_typehints = "description"
napoleon_use_param = True
napoleon_use_rtype = True
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
import sphinx_rtd_theme
#html_theme = 'pyramid'
#autodoc_default_flags = ['members',
#                         'undoc-members',
#                         'inherited-members',  # disabled because there's a bug in sphinx
#                         'show-inheritance',
#                        ]


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# for code blocks
pygments_style = 'sphinx'


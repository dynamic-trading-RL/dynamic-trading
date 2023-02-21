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
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'dynamic-trading'
copyright = '2023, Federico Giorgi'
author = 'Federico Giorgi'
release = '00.00.01'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'classic'
html_static_path = ['_static']


# -- Options for Latex output ------------------------------------------------
latex_engine = 'pdflatex'
latex_elements = {
    'sphinxsetup': """%
pre_background-TeXcolor={RGB}{242,242,242},% alias of VerbatimColor
pre_border-TeXcolor={RGB}{32,32,32},%
pre_box-decoration-break=slice,
% pre_border-top-width=5pt,% will be ignored due to non-zero radii
% pre_border-right-width=10pt,
% pre_border-bottom-width=15pt,
% pre_border-left-width=20pt,
pre_border-width=3pt,% sets equally the four border-widths,
%                      needed for rounded corners
pre_border-top-left-radius=20pt,
pre_border-top-right-radius=0pt,
pre_border-bottom-right-radius=20pt,
pre_border-bottom-left-radius=0pt,
pre_box-shadow=10pt 10pt,
pre_box-shadow-TeXcolor={RGB}{192,192,192},
%
div.topic_border-TeXcolor={RGB}{102,102,102},%
div.topic_box-shadow-TeXcolor={RGB}{187,187,187},%
div.topic_background-TeXcolor={RGB}{238,238,255},%
div.topic_border-bottom-right-radius=10pt,%
div.topic_border-top-right-radius=10pt,%
div.topic_border-width=2pt,%
div.topic_box-shadow=10pt 10pt,%
%
div.danger_border-width=10pt,%
div.danger_padding=6pt,% (see Important notice above)
div.danger_background-TeXcolor={rgb}{0.6,.8,0.8},%
div.danger_border-TeXcolor={RGB}{64,64,64},%
div.danger_box-shadow=-7pt 7pt,%
div.danger_box-shadow-TeXcolor={RGB}{192,192,192},%
div.danger_border-bottom-left-radius=15pt%
""",
}

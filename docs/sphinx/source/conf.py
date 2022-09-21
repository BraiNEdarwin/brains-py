# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import autoapi

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'BRAINS-Py'
copyright = '2022, Unai Alegre-Ibarra et al.'
author = 'Unai Alegre-Ibarra et al.'
release = '1.0.1'

import os
import sys
#print(os.path.abspath('../../../brainspy'))
sys.path.insert(0, '/home/unai/Documents/3-Programming/bspy/brains-py')

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
'sphinx.ext.autodoc',
'sphinx.ext.napoleon',
#'sphinx.ext.autosummary',
#'autoapi.sphinx',
'sphinx.ext.viewcode'
]

#autoapi_modules = {
#   'brainspy': {
#      'prune': True,
#      'override': True,
#      'output': 'auto'
#   }
#}


templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

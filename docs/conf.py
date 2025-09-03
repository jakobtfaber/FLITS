import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'FLITS'
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

autodoc_mock_imports = ['astropy', 'baseband_analysis']
exclude_patterns = []
html_theme = 'alabaster'

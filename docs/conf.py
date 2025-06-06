"""API Documentation Generation Configuration"""

# Add path to the DAFCOM module so Sphinx can import it
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

# Add _static directory for mock modules
sys.path.insert(0, os.path.abspath('_static'))

# Mock imports for external dependencies that may cause issues
autodoc_mock_imports = [
    'numpy',
    'pandas',
    'xarray',
    'matplotlib',
    'scipy',
    'sklearn',
    'torch',
    'xgboost'
]

# Mock modules for documentation
import sys
from unittest.mock import MagicMock

class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

# Create mock for dafcom modules to handle import errors
MOCK_MODULES = [
    'dafcom.forecast',
    'dafcom.forecast.processor',
    'dafcom.forecast.processor.unified_processor',
    'dafcom.forecast.models',
    'dafcom.forecast.models.model_factory',
    'dafcom.forecast.models.time_series_data',
    'dafcom.forecast.models.training',
    'dafcom.forecast.models.two_stage_pipeline',
    'dafcom.forecast.models.xgboost_model',
    'dafcom.forecast.cli'
]

# Apply mocks to system modules
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = Mock()

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'pydata_sphinx_theme',
    'sphinx.ext.autosummary',
    'numpydoc',
]

# Add any paths that contain templates here
templates_path = ['_templates']

# Source suffix
source_suffix = ['.rst', '.md']

# The master toctree document
master_doc = 'index'

# General information about the project
project = 'DAFCOM'
copyright = '2025, NWS OAR'
author = 'NWS OAR'

# The full version, including alpha/beta/rc tags
release = '0.1.0'

# List of patterns to exclude from source files
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing
todo_include_todos = False

# -- Options for HTML output -------------------------------------------------
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_logo = None  # Add your logo path here if you have one

# PyData theme settings
html_theme_options = {
    "github_url": "https://github.com/your-username/DAFCOM",
    "show_prev_next": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/your-username/DAFCOM",
            "icon": "fab fa-github",
        },
    ],
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "navbar_align": "content",
    "navbar_center": ["navbar-nav", "version-switcher"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    "footer_end": ["theme-version"],
}

html_context = {
    "github_user": "your-username",
    "github_repo": "DAFCOM",
    "github_version": "main",
    "doc_path": "docs",
}

# -- Extension configurations -------------------------------------------------
autodoc_member_order = 'bysource'
autoclass_content = 'both'

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

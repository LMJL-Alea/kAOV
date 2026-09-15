import os
import sys

# -- Path setup --------------------------------------------------------
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information ------------------------------------------------
project = "kAOV"
copyright = "2026, kAOV authors"
author = "Polina Arsenteva, Anthony Ozier-Lafontaine, Ghislain Durif"

try:
    from importlib.metadata import version as _version

    release = _version("kAOV")
except Exception:
    release = "dev"
version = release

# -- General configuration -----------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "myst_parser",
    "nbsphinx",
]

# --- API reference ---
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
numpydoc_show_class_members = False 

# --- notebooks ---
nbsphinx_execute = "never"  
nbsphinx_allow_errors = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- Options for HTML output -----------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = []
html_theme_options = {
    "github_url": "https://github.com/LMJL-Alea/kAOV",
    "show_prev_next": False,
}

# Configuration file for the Sphinx documentation builder.

import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath("_ext"))

project = "onnx-light-cpu"
author = "xadupre"

extensions = [
    "breathe",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
    "onnx_kernels",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]
html_last_updated_fmt = "%b %d, %Y"

sphinx_gallery_conf = {
    "examples_dirs": "examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"/plot_",
}

# Breathe pulls the C++ API reference from the Doxygen XML generated below, so
# the documentation is composed from the ``///`` comments in the public headers
# and always reflects the current state of the project.
_docs_dir = os.path.dirname(os.path.abspath(__file__))
_doxygen_xml = os.path.join(_docs_dir, "_doxygen", "xml")
breathe_projects = {"onnx_light_cpu": _doxygen_xml}
breathe_default_project = "onnx_light_cpu"
breathe_domain_by_extension = {"h": "cpp"}


def _run_doxygen(app):
    """Generate the Doxygen XML consumed by Breathe before Sphinx reads it."""
    subprocess.run(["doxygen", "Doxyfile"], cwd=_docs_dir, check=True)


def setup(app):
    app.connect("builder-inited", _run_doxygen)


html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"
html_theme_options = {
    "footer_start": ["last-updated"],
    "footer_end": [],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/xadupre/onnx-light-cpu",
            "icon": "fa-brands fa-github",
        },
    ],
}

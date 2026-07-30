# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath("_ext"))

project = "onnx-light-cpu"
copyright = "2025, xadupre"
author = "xadupre"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
    "onnx_kernels",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

sphinx_gallery_conf = {
    "examples_dirs": "examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"/plot_",
}

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/xadupre/onnx-light-cpu",
            "icon": "fa-brands fa-github",
        },
    ],
}

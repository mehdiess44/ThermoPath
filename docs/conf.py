"""Configuration Sphinx pour la documentation ThermoPath."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

project = "ThermoPath"
author = "Alaae Najib"
copyright = "2026, Alaae Najib"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "ThermoPath Documentation"

autodoc_typehints = "description"
autodoc_mock_imports = [
    "numpy",
    "pandas",
    "streamlit",
    "paho",
    "paho.mqtt",
    "paho.mqtt.client",
    "paho.mqtt.publish",
    "joblib",
    "sklearn",
    "sklearn.ensemble",
    "sklearn.metrics",
    "sklearn.model_selection",
    "sklearn.preprocessing",
    "matplotlib",
    "seaborn",
    "plotly",
    "openpyxl",
    "xlsxwriter",
]

os.environ.setdefault("BROKER_HOST", "localhost")
os.environ.setdefault("BROKER_PORT", "1883")

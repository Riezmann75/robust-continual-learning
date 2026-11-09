#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Set PYTHON to a Python interpreter with matplotlib available.
"${PYTHON:-python3}" generate_assets.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

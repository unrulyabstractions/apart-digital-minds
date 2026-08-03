#!/usr/bin/env bash
# Build the AAAI-27 submission -> build/main.pdf
#   bash build.sh        full build with bibliography
#   bash build.sh fast   single pass
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
BUILD="$SRC/build"
mkdir -p "$BUILD"
cd "$SRC"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD" main.tex
if [ "${1:-}" != "fast" ]; then
  ( cd "$BUILD" && BIBINPUTS="$SRC:" BSTINPUTS="$SRC:" bibtex main ) || true
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD" main.tex
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD" main.tex
fi
# The supplementary material is a second document on its own deadline. It has
# no bibliography, so one pass settles it and a second fixes its references.
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD" supplement.tex
if [ "${1:-}" != "fast" ]; then
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD" supplement.tex
fi
echo "=== Done: $BUILD/main.pdf and $BUILD/supplement.pdf ==="

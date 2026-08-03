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
echo "=== Done: $BUILD/main.pdf ==="

#!/usr/bin/env bash
# run_cov.sh ── one‑liner coverage helper for pytest
# Usage: ./run_cov.sh <svm|rfc|knn|cnn> [additional pytest args]

set -euo pipefail

MODEL=${1:-}

if [[ -z "$MODEL" ]]; then
  echo "Usage: $0 <svm|rfc|knn|cnn|all> [extra pytest args]" >&2
  exit 1
fi

case "$MODEL" in
  svm|rfc|knn|cnn|all) ;;                          # allowed options
  *)  echo "Error: '$MODEL' is not one of svm, rfc, knn, cnn,all" >&2
      exit 1 ;;
esac

shift                                     # drop the model arg, pass the rest to pytest



if [[ "$MODEL" == "all" ]]; then
HTML_DIR="coverage"
XML_FILE="$HTML_DIR/coverage.xml"

mkdir -p "$HTML_DIR"
  pytest \
    --cov=src \
    --cov-report=term-missing \
    --cov-report="html:${HTML_DIR}" \
    --cov-report="xml:${XML_FILE}" \
    "$@"
else
HTML_DIR="coverage/$MODEL"
XML_FILE="$HTML_DIR/coverage.xml"

mkdir -p "$HTML_DIR"
  pytest \
    --cov="src/$MODEL" \
    --cov-report=term-missing \
    --cov-report="html:${HTML_DIR}" \
    --cov-report="xml:${XML_FILE}" \
    "$@"
fi

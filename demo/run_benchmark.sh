#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 -m jupyter notebook "$SCRIPT_DIR/benchmark.ipynb"

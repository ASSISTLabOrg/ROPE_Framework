#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing dependencies..."
pip3 install --quiet -e "$SCRIPT_DIR/../python"
pip3 install --quiet -e "$SCRIPT_DIR"
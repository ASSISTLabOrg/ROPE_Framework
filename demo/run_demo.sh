#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing dependencies..."
pip3 install --quiet -e "$SCRIPT_DIR/../python"
pip3 install --quiet -e "$SCRIPT_DIR"

echo "Starting SWW26 Demo..."
echo "----------------------"

echo "Step 1: Running Orbit Integration..."
python3 "$SCRIPT_DIR/integrate_orbit.py" --days 1

echo ""
echo "Step 2: Running Sensitivity Study (Historical Solar Phases)..."
python3 "$SCRIPT_DIR/sensitivity.py" --days 1 --dt 60

echo ""
echo "Step 3: Running Reentry Simulation (900km to Splashdown)..."
python3 "$SCRIPT_DIR/reentry_test.py"

echo ""
echo "----------------------"
echo "Demo complete. All outputs are saved in the 'output' directory."
ls -l "$SCRIPT_DIR/output"

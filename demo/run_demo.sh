#!/bin/bash

# run_demo.sh
# 
# Date: 2026-04-13
#
# Runs the full orbit integration and sensitivity study.

set -e

echo "Starting SWW26 Demo..."
echo "----------------------"

# 1. Run Orbit Integration
echo "Step 1: Running Orbit Integration..."
python3 integrate_orbit.py --days 1

# 2. Run sensitivity Study
echo ""
echo "Step 2: Running sensitivity Study (Historical Solar Phases)..."
python3 src/sensitivity.py --days 1 --dt 60

# 3. Run Reentry Simulation
echo ""
echo "Step 3: Running Reentry Simulation (900km to Splashdown)..."
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python3 src/reentry_test.py

echo ""
echo "----------------------"
echo "Demo complete. All outputs are saved in the 'output' directory."
ls -l output

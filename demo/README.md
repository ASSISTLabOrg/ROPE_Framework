# sww26-demo: Orbital Propagation & Perturbation Analysis

`sww26-demo` is a Python-based project for high-fidelity orbital propagation and atmospheric drag perturbation analysis. It provides tools to integrate satellite orbits comparing two atmospheric models: the standard **USSA76** and the state-of-the-art **NRLMSIS-2.1**.

The project handles automated space weather data fetching (F10.7, Ap indices) from CelesTrak to ensure accurate simulations during different solar cycle phases.

## Features

- **Orbital Integration:** Compare satellite trajectories under USSA76 and MSIS-2.1 drag models.
- **sensitivity Studies:** Analyze atmospheric density and orbital decay across eight distinct solar cycle phases (from SC23 to SC25).
- **Reentry Simulation:** High-fidelity 900 km to splashdown analysis with precision landing comparison.
- **Automated Space Weather:** Fetches and caches F10.7 and Ap indices from CelesTrak.
- **Visualizations:** Generates comprehensive plots for altitude decay, density comparison, and performance metrics.

## Prerequisites

- **Python:** >= 3.8
- **Dependencies:** `numpy`, `scipy`, `matplotlib`, `pandas`, `pymsis`

## Installation

To install the package and its dependencies, run:

```bash
pip install -e .
```

To install development dependencies (like `pytest`):

```bash
pip install -e ".[dev]"
```

## Project Structure

- `src/demo_lib/`: Core library containing physics constants, perturbations, and space weather data handling.
- `src/integrate_orbit.py`: Main script for single-orbit propagation and model comparison.
- `src/sensitivity.py`: Script for multi-phase solar cycle sensitivity studies.
- `src/reentry_test.py`: Specialized script for sub-orbital reentry and splashdown precision tests.
- `tests/`: Project unit tests.

## Usage

### 1. Run the Full Demo
The easiest way to see the project in action is to run the provided bash script:

```bash
./run_demo.sh
```
This script performs a 1-day orbit integration, a historical solar phase sensitivity study, and a sub-orbital reentry simulation.

### 2. Manual Orbit Integration
Run the main integration script with custom parameters:

```bash
python src/integrate_orbit.py --days 2 --epoch 2023-10-05T00:00:00
```
**Key Arguments:**
- `--epoch <ISO-DATE>`: Starting time (e.g., `2023-10-05T00:00:00`).
- `--days <NUMBER>`: Duration of simulation in days.
- `--ic <R_X R_Y R_Z V_X V_Y V_Z>`: Initial conditions (km, km/s).
- `--bstar <VALUE>`: Drag coefficient (default: `1e-4`).
- `--no-sw`: Run with fixed space weather indices (F10.7=150, Ap=4).
- `--refresh-sw`: Force re-download of space weather data.

### 3. Historical sensitivity Study
Compare atmospheric models across different solar phases:

```bash
python src/sensitivity.py --alt 450 --days 2
```

### 4. Reentry Precision Test
Run a sub-orbital reentry from 900 km to splashdown:

```bash
python src/reentry_test.py
```
This simulation calculates the separation distance and timing differences between the USSA76 and MSIS-2.1 models during high-drag events.

## Testing

To run the unit tests, use:

```bash
pytest
```

## Outputs

Results are saved in the `output/` directory:
- `orbit_comparison.png`: Comparison of altitude, eccentricity, and inclination.
- `orbit_density.png`: Atmospheric density profiles and geographic tracking.
- `sensitivity_static.png`: Density vs Altitude across solar phases.
- `sensitivity_performance.png`: Timing and decay performance.
- `sensitivity_trajectory.png`: Time-series data along the orbit.
- `reentry_test_zoom.png`: 4-panel reentry analysis with precision splashdown zoom.

## License

MIT

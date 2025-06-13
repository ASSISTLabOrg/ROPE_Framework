# ROPE Framework

**R**educed **O**rder **P**robabilistic **E**mulator Framework.  
This repo contains python scripts for predicing the neutral density in the thermosphere using surrogate ML models trained on physics simulations (TIE-GCM, WAM-IPE).

## Running the Demo
1) Install the python environment in rope.yml
2) Launch the server from CLI
    >>> ./launch.sh
3) Load and run demo.ipynb

## Basic client-side code

>>> from ROPE.forecaster import DensityForecaster
>>> df = DensityForecaster()
>>> trajectory_1 = df.make_trajectory(iso_timestamp, time, altitude, latitude, longitude)
>>> trajectory_2 = df.make_trajectory(iso_timestamp_2, time_2, altitude_2, latitude_2, longitude_2) # and so on
>>> density = df.forecast(trajectory_1, trajectory_2)

## Contents

### forecast/
Contains the methods needed to use trained forecasting models to predict thermosphere neutral densities.
### data/
Contains the data needed to run forecaster models.
### _notebooks/
Contains notebooks for testing and developing new models. (DEV)
### _scripts/
Contains scripts for testing and developing new models. (DEV)
### _tie_gcm_*
Contains current testing models. (DEV)
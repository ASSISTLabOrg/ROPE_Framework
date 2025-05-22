# ROPE Framework

**R**educed **O**rder **P**robabilistic **E**mulator Framework.  
This repo contains python scripts for predicing the neutral density in the thermosphere using surrogate ML models trained on physics simulations (TIE-GCM, WAM-IPE).

## Contents

### forecast/  
Contains the methods needed to use trained forecasting models to predict thermosphere neutral densities.
### train/  
Contains the methods used in training and re-training the various forecaster methods.
### _data/
Contains the data needed to run forecaster models. Temporary - these data will eventually be held at a remote location.
### _notebooks/  
Contains notebooks for testing and developing new models.
### _scripts/
Contains scripts for testing and developing new models.
### _tie_gcm_*
Contains testing models.

import numpy as np
from demo_lib.perturbations import _MU, _RE

def test_constants():
    """Verify core physics constants."""
    assert np.isclose(_MU, 398600.4418)
    assert np.isclose(_RE, 6378.137)

def test_import():
    """Ensure the library can be imported."""
    import demo_lib
    from demo_lib.perturbations import ussa76
    from demo_lib.space_weather import SpaceWeather
    assert ussa76 is not None
    assert SpaceWeather is not None

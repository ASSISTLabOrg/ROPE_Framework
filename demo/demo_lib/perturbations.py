"""
perturbations.py
----------------

Date: 2026-04-13

Force models for perturbed orbit propagation.

Atmosphere models
-----------------
  USSA76 : classic 28-layer exponential model (no space weather)
  MSIS   : NRLMSIS-2.1 via pymsis, sampled at the satellite's true
           geographic position (lon/lat from ECI + GMST).
           Accepts either a scalar Ap or the full 7-element ap vector
           required for the time-history correction in MSIS-2.1.

Gravity
-------
  Zonal harmonics J2 - J6

All units: km, km/s, km/s^2
"""

import numpy as np
from datetime import datetime, timedelta
from .rope import Rope

# ---------------------------------------------------------------------------
# USSA76
# ---------------------------------------------------------------------------

_LAYERS = np.array([
    [0,      1.225,       7.310],
    [25,     4.008e-2,    6.427],
    [30,     1.841e-2,    6.546],
    [40,     3.996e-3,    7.360],
    [50,     1.027e-3,    8.342],
    [60,     3.097e-4,    7.583],
    [70,     8.283e-5,    6.661],
    [80,     1.846e-5,    5.927],
    [90,     3.416e-6,    5.533],
    [100,    5.606e-7,    5.703],
    [110,    9.708e-8,    6.782],
    [120,    2.222e-8,    9.973],
    [130,    8.152e-9,   13.243],
    [140,    3.831e-9,   16.322],
    [150,    2.076e-9,   21.652],
    [180,    5.194e-10,  27.974],
    [200,    2.541e-10,  34.934],
    [250,    6.073e-11,  43.342],
    [300,    1.916e-11,  49.755],
    [350,    7.014e-12,  54.513],
    [400,    2.803e-12,  58.019],
    [450,    1.184e-12,  60.980],
    [500,    5.215e-13,  65.654],
    [600,    1.137e-13,  76.377],
    [700,    3.070e-14, 100.587],
    [800,    1.136e-14, 147.203],
    [900,    5.759e-15, 208.020],
    [1000,   3.561e-15, 250.000],
], dtype=float)

_H_ALT    = _LAYERS[:, 0]
_RHO_BASE = _LAYERS[:, 1]
_SCALE_H  = _LAYERS[:, 2]


def ussa76(z: float) -> float:
    """Atmospheric density [kg/m^3] at geometric altitude z [km]."""
    z = float(np.clip(z, 0.0, 1000.0))
    i = int(np.searchsorted(_H_ALT, z, side="right")) - 1
    i = int(np.clip(i, 0, len(_LAYERS) - 1))
    return _RHO_BASE[i] * np.exp(-(z - _H_ALT[i]) / _SCALE_H[i])


# ---------------------------------------------------------------------------
# ECI -> geographic lon/lat
# ---------------------------------------------------------------------------

_J2000_JD = 2451545.0


def _gmst_rad(date: datetime) -> float:
    """Greenwich Mean Sidereal Time [rad] from UTC datetime (IAU formula)."""
    y, mo, d = date.year, date.month, date.day
    h = date.hour + date.minute / 60.0 + date.second / 3600.0
    if mo <= 2:
        y -= 1
        mo += 12
    A  = int(y / 100)
    B  = 2 - A + int(A / 4)
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (mo + 1)) + d + h / 24.0 + B - 1524.5
    theta = (280.46061837 + 360.98564736629 * (jd - _J2000_JD)
             + 0.000387933 * ((jd - _J2000_JD) / 36525.0)**2)
    return np.radians(theta % 360.0)


def eci_to_geodetic(r_eci: np.ndarray, date: datetime):
    """
    ECI position → (longitude [deg], geocentric latitude [deg], altitude [km]).
    Uses a spherical Earth — adequate for atmospheric density lookup.
    """
    gmst    = _gmst_rad(date)
    x, y, z = r_eci
    r       = np.linalg.norm(r_eci)

    lon_rad = np.arctan2(y, x) - gmst
    lon_deg = np.degrees(lon_rad) % 360.0
    if lon_deg > 180.0:
        lon_deg -= 360.0

    lat_deg = np.degrees(np.arcsin(np.clip(z / r, -1.0, 1.0)))
    alt_km  = r - 6378.14
    return lon_deg, lat_deg, alt_km

def lon_to_lst(lon_deg: float, date: datetime):
    return lon_deg / 15.0 + (date.hour + date.minute/60 + date.second/3600)


# ---------------------------------------------------------------------------
# MSIS density wrapper
# ---------------------------------------------------------------------------

def msis_density(z_km: float,
                 lon_deg: float,
                 lat_deg: float,
                 date: datetime,
                 f107: float,
                 f107a: float,
                 ap7) -> float:
    """
    Total mass density [kg/m^3] from NRLMSIS-2.1.

    Parameters
    ----------
    ap7 : float or list[float]
        Scalar daily Ap  OR  7-element list matching pymsis aps convention:
          [daily_Ap, ap_now, ap_3hr, ap_6hr, ap_9hr,
           mean_ap_12to33hr, mean_ap_36to57hr]
    """
    import pymsis

    if np.isscalar(ap7):
        ap_arr = [[float(ap7)] * 7]
    else:
        ap_arr = [list(ap7)]

    out = pymsis.calculate(
        [date], [lon_deg], [lat_deg], [z_km],
        f107s=[f107], f107as=[f107a], aps=ap_arr,
        version=2.1,
    )
    rho = float(out[0, pymsis.Variable.MASS_DENSITY])
    return max(rho, 1e-30)

# ---------------------------------------------------------------------------
# ROPE density wrapper
# ---------------------------------------------------------------------------

def rope_density(rp: Rope,
                 date: datetime, # UTC
                 lon_deg: float,
                 lat_deg: float, 
                 alt_km: float):
    """
    Total mass density [kg/m^3] from rope-0.2
    """
    return rp.get(date, lon_to_lst(lon_deg, date), lat_deg, alt_km)["density"]
    # discards uncertainty

# ---------------------------------------------------------------------------
# Drag
# ---------------------------------------------------------------------------

_RE   = 6378.14
_WE   = np.array([0.0, 0.0, 7.2921159e-5])
_RHO0 = 0.15696615


def drag_acceleration(r_eci: np.ndarray, v_eci: np.ndarray,
                      bstar: float, rho: float) -> np.ndarray:
    """ECI drag acceleration [km/s^2]."""
    v_rel   = v_eci - np.cross(_WE, r_eci)
    v_rel_m = v_rel * 1000.0
    speed_m = np.linalg.norm(v_rel_m)
    uv      = v_rel_m / speed_m
    return -(rho / _RHO0) * bstar * speed_m**2 * uv / 1000.0


# ---------------------------------------------------------------------------
# J2-J6 gravity
# ---------------------------------------------------------------------------

_MU      = 3.986004418e5
_RE_GRAV = 6378.0000014

_J2 =  0.001082
_J3 = -2.33936e-3 * _J2
_J4 = -1.49601e-3 * _J2
_J5 = -0.20995e-3 * _J2
_J6 =  0.49941e-3 * _J2


def gravity_perturbations(r_eci: np.ndarray) -> np.ndarray:
    """Zonal harmonic acceleration J2-J6 [km/s^2]."""
    x, y, z = r_eci
    r  = np.linalg.norm(r_eci)
    r2 = r * r
    z2 = z * z
    base = _MU / r2
    rr   = _RE_GRAV / r

    f2 = -1.5 * _J2 * base * rr**2
    c2 = 1.0 - 5.0 * z2 / r2
    aJ2 = np.array([f2 * c2 * x / r,
                    f2 * c2 * y / r,
                    f2 * (3.0 - 5.0 * z2 / r2) * z / r])

    f3 = 0.5 * _J3 * base * rr**3
    c3 = 5.0 * (7.0 * z2 * z / r**3 - 3.0 * z / r)
    aJ3 = np.array([f3 * c3 * x / r,
                    f3 * c3 * y / r,
                    f3 * 3.0 * (1.0 - 10.0 * z2 / r2 + 35.0 * z2**2 / (3.0 * r2**2))])

    f4 = (5.0 / 8.0) * _J4 * base * rr**4
    c4 = 3.0 - 42.0 * z2 / r2 + 63.0 * z2**2 / r2**2
    aJ4 = np.array([f4 * c4 * x / r,
                    f4 * c4 * y / r,
                    f4 * (15.0 - 70.0 * z2 / r2 + 63.0 * z2**2 / r2**2) * z / r])

    f5 = (1.0 / 8.0) * _J5 * base * rr**5
    c5 = 3.0 * (35.0 * z / r - 210.0 * z2 * z / r**3 + 231.0 * z2**2 * z / r**5)
    aJ5 = np.array([f5 * c5 * x / r,
                    f5 * c5 * y / r,
                    f5 * (693.0 * z2**3 / r**6 - 945.0 * z2**2 / r2**2
                          + 315.0 * z2 / r2 - 15.0)])

    f6 = -(1.0 / 16.0) * _J6 * base * rr**6
    c6 = 35.0 - 945.0 * z2 / r2 + 3465.0 * z2**2 / r2**2 - 3003.0 * z2**3 / r2**3
    aJ6 = np.array([f6 * c6 * x / r,
                    f6 * c6 * y / r,
                    f6 * (245.0 - 2205.0 * z2 / r2 + 4851.0 * z2**2 / r2**2
                          - 3003.0 * z2**3 / r2**3) * z / r])

    return aJ2 + aJ3 + aJ4 + aJ5 + aJ6


# ---------------------------------------------------------------------------
# Public APIs — return (acceleration [km/s^2], rho [kg/m^3])
# ---------------------------------------------------------------------------

def perturbations_ussa76(r_eci: np.ndarray, v_eci: np.ndarray,
                         bstar: float):
    """Total perturbation using USSA76. Returns (a, rho)."""
    alt = np.linalg.norm(r_eci) - _RE
    rho = ussa76(alt)
    a   = drag_acceleration(r_eci, v_eci, bstar, rho) + gravity_perturbations(r_eci)
    return a, rho


def perturbations_msis(r_eci: np.ndarray, v_eci: np.ndarray,
                       bstar: float,
                       date: datetime,
                       f107: float,
                       f107a: float,
                       ap7):
    """
    Total perturbation using NRLMSIS-2.1 at actual satellite position.
    Returns (a, rho).

    ap7 : scalar Ap  or  7-element list (see msis_density docstring)
    """
    lon, lat, alt = eci_to_geodetic(r_eci, date)
    rho = msis_density(alt, lon_deg=lon, lat_deg=lat, date=date,
                       f107=f107, f107a=f107a, ap7=ap7)
    a   = drag_acceleration(r_eci, v_eci, bstar, rho) + gravity_perturbations(r_eci)
    return a, rho

def perturbations_rope(r_eci: np.ndarray, v_eci: np.ndarray,
                       bstar: float,
                       date: datetime,
                       rp: Rope):
    """
    Total perturbation using ROPE-0.2.0 at actual satellite position.
    Returns (a, rho).
    """
    lon, lat, alt = eci_to_geodetic(r_eci, date)
    rho = rope_density(rp, date, lon, lat, alt)
    a   = drag_acceleration(r_eci, v_eci, bstar, rho) + gravity_perturbations(r_eci)
    return a, rho
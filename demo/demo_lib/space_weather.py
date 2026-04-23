"""
space_weather.py
----------------

Date: 2026-04-13

Fetch, parse, and query CelesTrak Space Weather data (SW-Last5Years.csv
or SW-All.csv) for accurate F10.7 and Ap indices.

The CSV is downloaded once and cached locally as a pickle so subsequent
runs are instant.  Pass force_refresh=True to re-download.

Public API
----------
    sw = SpaceWeather()               # loads / downloads data
    f107, f107a, ap7 = sw.get(date)   # query for a specific UTC datetime

    # get() returns:
    #   f107   : daily F10.7 observed (previous day, as MSIS expects)
    #   f107a  : centered 81-day average F10.7 observed
    #   ap7    : 7-element ap vector for MSIS:
    #               [0] daily Ap
    #               [1] 3-hr ap for current 3-hr window
    #               [2] ap 3 hrs prior
    #               [3] ap 6 hrs prior
    #               [4] ap 9 hrs prior
    #               [5] mean of eight 3-hr ap from 12–33 hrs prior
    #               [6] mean of eight 3-hr ap from 36–57 hrs prior
"""

import io
import os
import pickle
import ssl
import urllib.error
import urllib.request
from datetime import datetime, timedelta, date as date_type
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_URL_RECENT = "https://celestrak.org/SpaceData/SW-Last5Years.csv"
_URL_ALL    = "https://celestrak.org/SpaceData/SW-All.csv"

_CACHE_PATH_RECENT = Path(__file__).parent / ".sw_cache_recent.pkl"
_CACHE_PATH_ALL    = Path(__file__).parent / ".sw_cache_all.pkl"

_AP_COLS = ["AP1", "AP2", "AP3", "AP4", "AP5", "AP6", "AP7", "AP8"]


# ---------------------------------------------------------------------------
# Download + parse
# ---------------------------------------------------------------------------

def _download_csv(url: str) -> pd.DataFrame:
    """Download CelesTrak SW CSV and return a DataFrame indexed by date."""
    print(f"  Downloading space weather data from:\n    {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "orbit-propagator/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.URLError as e:
        if "SSL" not in str(e) and "certificate" not in str(e).lower():
            raise
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            raw = resp.read().decode("utf-8")

    # The file has a header section terminated by BEGIN OBSERVED / similar
    # keywords in the legacy format; the CSV format is clean — just skip
    # any comment/keyword lines that don't start with a digit or "DATE".
    lines = []
    for line in raw.splitlines():
        s = line.strip()
        if s.startswith("DATE") or (s and s[0].isdigit()):
            lines.append(s)

    df = pd.read_csv(
        io.StringIO("\n".join(lines)),
        parse_dates=["DATE"],
    )
    df = df.set_index("DATE").sort_index()
    return df


def _load_or_download(force_refresh: bool = False,
                      use_all: bool = False) -> pd.DataFrame:
    cache_path = _CACHE_PATH_ALL if use_all else _CACHE_PATH_RECENT
    cached_df = None
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cached_df = pickle.load(f)

    # Return cache immediately if fresh enough and no forced refresh
    if cached_df is not None and not force_refresh:
        last = cached_df.index[-1].date()
        # If use_all, we don't necessarily need it to be "fresh" if we already have it,
        # but for recent data we want to keep it updated.
        if use_all or (datetime.utcnow().date() - last).days <= 2:
            return cached_df
        print(f"  Cache {cache_path.name} stale, attempting refresh ...")

    # Try to download fresh data
    url = _URL_ALL if use_all else _URL_RECENT
    try:
        df = _download_csv(url)
        with open(cache_path, "wb") as f:
            pickle.dump(df, f)
        print(f"  Cached {len(df)} days of space weather data → {cache_path}")
        return df
    except Exception as e:
        if cached_df is not None:
            print(f"  Download failed ({e}); using existing cache "
                  f"({len(cached_df)} rows through {cached_df.index[-1].date()})")
            return cached_df
        raise RuntimeError(
            f"No SW cache and download failed: {e}\n"
            "Run once with network access to populate the cache."
        ) from e


# ---------------------------------------------------------------------------
# SpaceWeather class
# ---------------------------------------------------------------------------

class SpaceWeather:
    """
    Provides F10.7 and Ap indices for any UTC date covered by CelesTrak's
    space weather file.

    Parameters
    ----------
    force_refresh : bool
        Re-download even if a fresh cache exists.
    use_all : bool
        Use SW-All.csv (back to 1957) instead of SW-Last5Years.csv.
    """

    def __init__(self, force_refresh: bool = False, use_all: bool = False):
        self._df = _load_or_download(force_refresh=force_refresh, use_all=use_all)
        self._dates = self._df.index  # DatetimeIndex

    # ------------------------------------------------------------------
    def _row(self, d: date_type) -> pd.Series:
        """Return the DataFrame row for date d, with nearest-day fallback."""
        key = pd.Timestamp(d)
        if key in self._df.index:
            return self._df.loc[key]
        # Fall back to nearest available date (handles future predictions)
        idx = self._df.index.get_indexer([key], method="nearest")[0]
        return self._df.iloc[idx]

    # ------------------------------------------------------------------
    def get(self, dt: datetime):
        """
        Return (f107, f107a, ap7) for the given UTC datetime.

        f107   : F10.7 observed on the *previous* day  (MSIS convention)
        f107a  : centered 81-day average F10.7 observed
        ap7    : 7-element list for pymsis `aps` parameter:
                    [daily_Ap,
                     ap_current_3hr, ap_3hr_ago, ap_6hr_ago, ap_9hr_ago,
                     mean_ap_12to33hr_ago, mean_ap_36to57hr_ago]

        The 3-hourly ap values are selected from AP1..AP8 based on the
        UTC hour of `dt`.  The historical ap windows use the preceding days.
        """
        d      = dt.date()
        d_prev = d - timedelta(days=1)

        row_today = self._row(d)
        row_prev  = self._row(d_prev)

        # F10.7 daily (previous day) and 81-day average
        f107  = self._safe(row_prev,  "F10.7_OBS",          fallback=150.0)
        f107a = self._safe(row_today, "F10.7_OBS_CENTER81", fallback=150.0)

        # Daily Ap
        daily_ap = self._safe(row_today, "AP_AVG", fallback=4.0)

        # 3-hourly ap for current and prior windows
        ap_vec = self._build_ap_vector(dt)

        ap7 = [daily_ap] + ap_vec
        return float(f107), float(f107a), ap7

    # ------------------------------------------------------------------
    @staticmethod
    def _safe(row, col, fallback):
        try:
            v = row[col]
            if pd.isna(v):
                return fallback
            return float(v)
        except (KeyError, TypeError):
            return fallback

    # ------------------------------------------------------------------
    def _get_3hr_ap(self, d: date_type, hour: int) -> float:
        """
        Return the 3-hourly ap value for a given date and UTC hour.
        hour is mapped to AP1..AP8 (each covers 3 hours starting at 0,3,6,...).
        """
        row = self._row(d)
        slot = int(hour // 3)           # 0-based slot index [0..7]
        col  = _AP_COLS[slot]
        return self._safe(row, col, fallback=4.0)

    def _build_ap_vector(self, dt: datetime) -> list:
        """
        Build the 6-element 3-hourly ap history vector expected by pymsis:
          [ap_now, ap_3hr_ago, ap_6hr_ago, ap_9hr_ago,
           mean(ap_12..33hr_ago),  mean(ap_36..57hr_ago)]
        """
        ap = []
        for lag_hr in [0, 3, 6, 9]:
            t_lag  = dt - timedelta(hours=lag_hr)
            ap.append(self._get_3hr_ap(t_lag.date(), t_lag.hour))

        # Mean of eight 3-hr slots from 12 to 33 hours prior
        vals = []
        for lag_hr in range(12, 36, 3):
            t_lag = dt - timedelta(hours=lag_hr)
            vals.append(self._get_3hr_ap(t_lag.date(), t_lag.hour))
        ap.append(float(np.mean(vals)))

        # Mean of eight 3-hr slots from 36 to 57 hours prior
        vals = []
        for lag_hr in range(36, 60, 3):
            t_lag = dt - timedelta(hours=lag_hr)
            vals.append(self._get_3hr_ap(t_lag.date(), t_lag.hour))
        ap.append(float(np.mean(vals)))

        return ap

    # ------------------------------------------------------------------
    def summary(self, dt: datetime) -> str:
        f107, f107a, ap7 = self.get(dt)
        return (f"  date={dt.date()}  F10.7={f107:.1f}  F10.7a={f107a:.1f}  "
                f"Ap(daily)={ap7[0]:.0f}  ap(now)={ap7[1]:.0f}")


# ---------------------------------------------------------------------------
# SpaceWeatherEngine
# ---------------------------------------------------------------------------

class SpaceWeatherEngine:
    """CPU-optimized engine for ultra-fast space weather lookups."""

    def __init__(self, sw_loader, epoch, duration_days):
        # 1. Define a regular time grid (seconds from epoch)
        # Sampling every 1 hour (3600s) is sufficient for MSIS drivers.
        t_max = (duration_days + 1) * 86400
        self.t_grid = np.arange(-86400, t_max + 86400, 3600.0)

        # 2. Pre-allocate NumPy buffers
        self.f107  = np.zeros_like(self.t_grid)
        self.f107a = np.zeros_like(self.t_grid)
        self.ap7   = np.zeros((len(self.t_grid), 7))

        # 3. Pre-fill buffers using the slow loader (once)
        for i, t in enumerate(self.t_grid):
            dt = epoch + timedelta(seconds=float(t))
            f, fa, a7 = sw_loader.get(dt)
            self.f107[i]  = f
            self.f107a[i] = fa
            self.ap7[i, :] = a7

    def query(self, t_sec):
        """Ultra-fast O(1) lookup using float seconds."""
        # Linear interpolation for F10.7
        f107_val  = np.interp(t_sec, self.t_grid, self.f107)
        f107a_val = np.interp(t_sec, self.t_grid, self.f107a)

        # Piecewise constant lookup for Ap indices
        # Binary search (O(log N)) is extremely fast on CPU
        idx = np.searchsorted(self.t_grid, t_sec, side='right') - 1
        return f107_val, f107a_val, self.ap7[idx]

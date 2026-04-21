"""
rope.py — Python binding for the ROPE framework.

Wraps librope.so via ctypes for fast in-process interpolation queries, and
the rope CLI subprocess for forecast and server lifecycle management.

Typical usage
-------------
    from rope import Rope

    r = Rope() #config_path="config/rope.conf"
    r.forecast("2024-02-09 00:00:00", horizon=24)

    with r:   # opens handle, closes on exit
        result = r.get(time="2024-02-09T06:00:00Z", lst=7.5, lat=45.0, alt_km=400.0)
        print(result)  # {"density": 4.72e-12, "uncertainty": 3.5e-13}
"""

import ctypes
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROPE_HOLD   = 0
ROPE_INTERP = 1

_ERR_NAMES = {
    0: "ok",
    1: "no server",
    2: "no forecast cached",
    3: "time out of range",
    4: "spatial point out of range",
    5: "bad argument",
    6: "internal error",
}


class RopeError(RuntimeError):
    def __init__(self, code: int, message: str):
        super().__init__(f"[{_ERR_NAMES.get(code, str(code))}] {message}")
        self.code = code


class Rope:
    """
    Client for the ROPE atmospheric density service.

    Parameters
    ----------
    lib_path    : Path to librope.so / librope.dylib / rope.dll.
                  Defaults to lib/librope.so relative to this file.
    exe_path    : Path to the rope CLI executable.
                  Defaults to bin/rope relative to this file.
    socket_path : Unix domain socket path. None → platform default (/tmp/rope.sock).
    config_path : Path to rope.conf passed to `rope forecast`.
    """

    def __init__(
        self,
        lib_path: "str | Path | None" = None,
        exe_path: "str | Path | None" = None,
        socket_path: "str | None" = None,
        config_path: "str | Path | None" = None,
    ):
        here = Path(__file__).parent

        if lib_path is None:
            candidates = [
                here / "lib" / "librope.so",
                here / "lib" / "librope.dylib",
                here / "bin"  / "rope.dll",
                here / "build" / "librope.so",
            ]
            for c in candidates:
                if c.exists():
                    lib_path = c
                    break
            else:
                raise FileNotFoundError(
                    "librope not found; pass lib_path= explicitly or check your package layout"
                )

        if exe_path is None:
            for c in [here / "bin" / "rope", here / "bin" / "rope.exe", here / "build" / "rope"]:
                if c.exists():
                    exe_path = c
                    break

        self._lib_path    = Path(lib_path)
        self._exe_path    = Path(exe_path) if exe_path else None
        self._socket_path = socket_path
        self._config_path = Path(config_path) if config_path else None
        self._handle: "int | None" = None
        self._lib         = self._load_lib()

    # ------------------------------------------------------------------
    # Context manager — opens/closes the interpolation handle
    # ------------------------------------------------------------------

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Handle lifecycle
    # ------------------------------------------------------------------

    def open(self):
        """Fetch the cached grid from the server and open an interpolation handle."""
        if self._handle is not None:
            return
        err  = ctypes.create_string_buffer(256)
        sock = self._socket_path.encode() if self._socket_path else None
        handle = self._lib.rope_open(sock, err, len(err))
        if not handle:
            raise RopeError(1, err.value.decode())
        self._handle = handle

    def close(self):
        """Release the interpolation handle. Does not affect the server."""
        if self._handle is not None:
            self._lib.rope_close(self._handle)
            self._handle = None

    def refresh(self):
        """Re-fetch the grid from the server (picks up a new forecast)."""
        self.close()
        self.open()

    # ------------------------------------------------------------------
    # Server commands (via CLI subprocess)
    # ------------------------------------------------------------------

    def forecast(self, start: "str | datetime", horizon: int) -> dict:
        """
        Ask the server to run a forecast and cache the resulting grid.

        Parameters
        ----------
        start   : Forecast start time — ISO 8601 string or datetime (UTC).
        horizon : Forecast duration in hours.

        Returns the server response, e.g.:
            {"status": "ok", "window_start": "...", "window_end": "..."}
        """
        if self._exe_path is None:
            raise RuntimeError("rope executable not found; cannot run forecast")

        if isinstance(start, datetime):
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            start = start.strftime("%Y-%m-%d %H:%M:%S")

        cmd = [str(self._exe_path), "forecast", "--start", start, "--horizon", str(horizon)]
        if self._config_path:
            cmd += ["--config", str(self._config_path)]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RopeError(6, (proc.stderr or proc.stdout).strip())

        # Take the last non-empty line — guards against any preamble lines
        # (e.g. duplicate response if a stale server was replaced mid-flight).
        lines = [l for l in proc.stdout.splitlines() if l.strip()]
        return json.loads(lines[-1])

    # ------------------------------------------------------------------
    # Interpolation queries (via C API)
    # ------------------------------------------------------------------

    def get(
        self,
        time: "float | str | datetime",
        lst: float,
        lat: float,
        alt_km: float,
        mode: int = ROPE_INTERP,
    ) -> dict:
        """
        Query density and uncertainty at a single point.

        Parameters
        ----------
        time   : Query time — Unix timestamp, ISO 8601 string, or datetime (UTC).
        lst    : Local Solar Time, hours [0, 24).
        lat    : Geodetic latitude, degrees [-87.5, 87.5].
        alt_km : Geometric altitude, km [100, 980].
        mode   : ROPE_INTERP (default) or ROPE_HOLD.

        Returns {"density": float, "uncertainty": float} in kg/m³.
        """
        if self._handle is None:
            self.open()

        density     = ctypes.c_double()
        uncertainty = ctypes.c_double()
        err         = ctypes.create_string_buffer(256)

        rc = self._lib.rope_query(
            self._handle, mode, _to_unix(time), lst, lat, alt_km,
            ctypes.byref(density), ctypes.byref(uncertainty),
            err, len(err),
        )
        if rc != 0:
            raise RopeError(rc, err.value.decode())

        return {"density": density.value, "uncertainty": uncertainty.value}

    def get_batch(
        self,
        times: list,
        lsts: list,
        lats: list,
        alts_km: list,
        mode: int = ROPE_INTERP,
    ) -> "list[dict]":
        """
        Query density and uncertainty at N points in one call.

        Each parameter is a list of length N.
        Returns a list of N dicts: [{"density": float, "uncertainty": float}, ...].
        """
        if self._handle is None:
            self.open()

        n = len(times)
        if not (len(lsts) == len(lats) == len(alts_km) == n):
            raise ValueError("all input lists must have the same length")

        DA      = ctypes.c_double * n
        t_arr   = DA(*(_to_unix(t) for t in times))
        lst_arr = DA(*lsts)
        lat_arr = DA(*lats)
        alt_arr = DA(*alts_km)
        den_arr = DA()
        unc_arr = DA()
        err     = ctypes.create_string_buffer(256)

        rc = self._lib.rope_query_batch(
            self._handle, mode, n,
            t_arr, lst_arr, lat_arr, alt_arr,
            den_arr, unc_arr,
            err, len(err),
        )
        if rc != 0:
            raise RopeError(rc, err.value.decode())

        return [{"density": den_arr[i], "uncertainty": unc_arr[i]} for i in range(n)]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_lib(self) -> ctypes.CDLL:
        lib = ctypes.CDLL(str(self._lib_path))

        lib.rope_open.restype  = ctypes.c_void_p
        lib.rope_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]

        lib.rope_query.restype  = ctypes.c_int
        lib.rope_query.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.c_char_p, ctypes.c_int,
        ]

        lib.rope_query_batch.restype  = ctypes.c_int
        lib.rope_query_batch.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.c_char_p, ctypes.c_int,
        ]

        lib.rope_close.restype  = None
        lib.rope_close.argtypes = [ctypes.c_void_p]

        return lib


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _to_unix(t: "float | str | datetime") -> float:
    if isinstance(t, datetime):
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return t.timestamp()
    if isinstance(t, str):
        # Accept ISO 8601 with or without Z/offset
        t = t.rstrip("Z")
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M"):
            try:
                return datetime.strptime(t, fmt).replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                pass
        raise ValueError(f"cannot parse time string: {t!r}")
    return float(t)

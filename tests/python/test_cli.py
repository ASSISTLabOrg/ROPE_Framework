"""
Category C — end-to-end CLI integration tests.

Exercises the full rope pipeline: server spawn, forecast, single-point query,
batch query, idle timeout, and clean shutdown — all via the rope CLI binary.

Required env vars (injected by CTest; fall back to sensible defaults for
manual runs from the build directory):
  ROPE_EXE          path to the rope binary
  ROPE_FIXTURE_DIR  path to the tests/fixtures directory
"""

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

ROPE_EXE    = os.environ.get("ROPE_EXE", "rope")
FIXTURE_DIR = Path(os.environ.get("ROPE_FIXTURE_DIR",
                                   Path(__file__).parent.parent / "fixtures"))

# Must match the sw_test.csv coverage window used by the pipeline tests.
# sw_test.csv covers 2023-12-31T22:00:00 through 2024-01-01T02:00:00 (5 rows);
# horizon=3 with seq_len=3 requires exactly (seq_len-1)+horizon = 5 rows.
FORECAST_START   = "2024-01-01 00:00:00"
FORECAST_HORIZON = 3
QUERY_TIME       = "2024-01-01T01:00:00"


def _rope(*args, timeout=60):
    return subprocess.run(
        [ROPE_EXE, *args],
        capture_output=True, text=True, timeout=timeout,
    )


def _write_conf(path: Path, idle_timeout_seconds: int = 30) -> None:
    path.write_text(
        f"[paths]\n"
        f"exported_dir = {FIXTURE_DIR / 'test_models'}\n"
        f"driver_csv   = {FIXTURE_DIR / 'sw_test.csv'}\n"
        f"ic_csv       = {FIXTURE_DIR / 'ic_test.csv'}\n"
        f"[server]\n"
        f"idle_timeout_seconds = {idle_timeout_seconds}\n"
    )


# ---------------------------------------------------------------------------
# Module-scoped server fixture — one server shared across most tests.
# ---------------------------------------------------------------------------

class _Server:
    def __init__(self, sock: str, conf: str):
        self.sock = sock
        self.conf = conf

    def run(self, *args, timeout=60):
        return _rope("--socket", self.sock, *args, timeout=timeout)


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    tmp  = tmp_path_factory.mktemp("rope_cli")
    sock = str(tmp / "rope.sock")
    conf = tmp / "rope.conf"
    _write_conf(conf)

    result = _rope(
        "--socket", sock, "forecast",
        "--start", FORECAST_START, "--horizon", str(FORECAST_HORIZON),
        "--config", str(conf),
    )
    assert result.returncode == 0, f"server startup failed:\n{result.stderr}"

    srv = _Server(sock, str(conf))
    yield srv

    _rope("--socket", sock, "exit", timeout=10)


# ---------------------------------------------------------------------------
# forecast
# ---------------------------------------------------------------------------

def test_forecast_returns_ok(server):
    result = server.run(
        "forecast",
        "--start", FORECAST_START, "--horizon", str(FORECAST_HORIZON),
        "--config", server.conf,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout.strip().splitlines()[-1])
    assert data["status"] == "ok"
    assert "window_start" in data
    assert "window_end"   in data


# ---------------------------------------------------------------------------
# get — single point
# ---------------------------------------------------------------------------

def test_get_interp_returns_valid_point(server):
    result = server.run(
        "get", "--mode", "interp",
        "--time", QUERY_TIME, "--lst", "12.0", "--lat", "45.0", "--alt", "400.0",
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["density"]     > 0
    assert data["uncertainty"] >= 0


def test_get_hold_returns_valid_point(server):
    result = server.run(
        "get", "--mode", "hold",
        "--time", QUERY_TIME, "--lst", "12.0", "--lat", "45.0", "--alt", "400.0",
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["density"]     > 0
    assert data["uncertainty"] >= 0


def test_get_time_out_of_range_fails(server):
    result = server.run(
        "get", "--mode", "interp",
        "--time", "2030-01-01T00:00:00",
        "--lst", "12.0", "--lat", "45.0", "--alt", "400.0",
        timeout=10,
    )
    assert result.returncode != 0


# ---------------------------------------------------------------------------
# get — batch
# ---------------------------------------------------------------------------

def test_batch_get_from_csv(server, tmp_path):
    csv = tmp_path / "queries.csv"
    csv.write_text(
        "YYYY,MM,DD,HH,MIN,SS,lst,lat,alt_km\n"
        "2024,01,01,01,00,00,12.0,45.0,400.0\n"
        "2024,01,01,01,30,00,6.0,-30.0,300.0\n"
    )
    out = tmp_path / "results.json"
    result = server.run(
        "get", "--mode", "interp",
        "--file", str(csv), "--output", str(out),
        timeout=15,
    )
    assert result.returncode == 0
    rows = json.loads(out.read_text())
    assert len(rows) == 2
    for row in rows:
        assert row["density"]     > 0
        assert row["uncertainty"] >= 0


# ---------------------------------------------------------------------------
# Idle timeout — separate short-lived server
# ---------------------------------------------------------------------------

def test_idle_timeout_shuts_down_server(tmp_path):
    sock = str(tmp_path / "rope_idle.sock")
    conf = tmp_path / "rope_idle.conf"
    _write_conf(conf, idle_timeout_seconds=5)

    start = _rope(
        "--socket", sock, "forecast",
        "--start", FORECAST_START, "--horizon", str(FORECAST_HORIZON),
        "--config", str(conf),
    )
    assert start.returncode == 0, f"idle-timeout server startup failed:\n{start.stderr}"

    # Wait long enough for the watcher to fire (timeout=5s, watcher checks every 1s,
    # so 8 seconds gives two full check cycles of margin).
    time.sleep(8)

    probe = _rope(
        "--socket", sock, "get", "--mode", "interp",
        "--time", QUERY_TIME, "--lst", "12.0", "--lat", "45.0", "--alt", "400.0",
        timeout=5,
    )
    assert probe.returncode != 0, "server should have shut down due to idle timeout"

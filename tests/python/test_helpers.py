"""
Unit tests for pure helper functions in python/rope.py.
None of these require the compiled librope or a running server.
"""

import configparser
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Make rope.py importable from the project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))
from rope import _to_unix, _conf_get, _write_temp_conf, RopeError, _ERR_NAMES


# ---------------------------------------------------------------------------
# _to_unix
# ---------------------------------------------------------------------------

class TestToUnix:
    def test_float_passthrough(self):
        assert _to_unix(1704067200.0) == 1704067200.0

    def test_int_cast(self):
        assert _to_unix(0) == 0.0

    def test_datetime_utc(self):
        dt = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert _to_unix(dt) == pytest.approx(1704067200.0)

    def test_datetime_naive_treated_as_utc(self):
        # Naive datetime → UTC assumed
        dt = datetime(2024, 1, 1, 0, 0, 0)
        assert _to_unix(dt) == pytest.approx(1704067200.0)

    def test_iso_string_T_separator(self):
        assert _to_unix("2024-01-01T00:00:00") == pytest.approx(1704067200.0)

    def test_iso_string_Z_suffix(self):
        assert _to_unix("2024-01-01T00:00:00Z") == pytest.approx(1704067200.0)

    def test_iso_string_space_separator(self):
        assert _to_unix("2024-01-01 00:00:00") == pytest.approx(1704067200.0)

    def test_iso_string_no_seconds(self):
        assert _to_unix("2024-01-01T06:30") == pytest.approx(1704090600.0)

    def test_bad_string_raises(self):
        with pytest.raises(ValueError):
            _to_unix("not-a-date")


# ---------------------------------------------------------------------------
# _conf_get
# ---------------------------------------------------------------------------

class TestConfGet:
    def _make_conf(self, content: str) -> Path:
        fd, path = tempfile.mkstemp(suffix=".conf")
        with os.fdopen(fd, "w") as f:
            f.write(content)
        return Path(path)

    def test_reads_existing_key(self, tmp_path):
        p = tmp_path / "test.conf"
        p.write_text("[decoder]\ndevice = cuda\n")
        assert _conf_get(p, "decoder", "device", "cpu") == "cuda"

    def test_returns_fallback_for_missing_key(self, tmp_path):
        p = tmp_path / "test.conf"
        p.write_text("[decoder]\n")
        assert _conf_get(p, "decoder", "device", "cpu") == "cpu"

    def test_returns_fallback_for_missing_section(self, tmp_path):
        p = tmp_path / "test.conf"
        p.write_text("")
        assert _conf_get(p, "missing_section", "key", "default") == "default"

    def test_returns_fallback_for_nonexistent_file(self, tmp_path):
        p = tmp_path / "no_such_file.conf"
        assert _conf_get(p, "any", "key", "fallback") == "fallback"


# ---------------------------------------------------------------------------
# _write_temp_conf
# ---------------------------------------------------------------------------

class TestWriteTempConf:
    def test_creates_readable_file(self, tmp_path):
        base = tmp_path / "base.conf"
        base.write_text("[decoder]\ndevice = cpu\n")
        path = _write_temp_conf(base, "decoder", "device", "cuda")
        try:
            assert os.path.exists(path)
            cp = configparser.ConfigParser()
            cp.read(path)
            assert cp.get("decoder", "device") == "cuda"
        finally:
            os.unlink(path)

    def test_preserves_other_sections(self, tmp_path):
        base = tmp_path / "base.conf"
        base.write_text("[models]\ndir = /tmp/models\n[decoder]\ndevice = cpu\n")
        path = _write_temp_conf(base, "decoder", "device", "cuda:0")
        try:
            cp = configparser.ConfigParser()
            cp.read(path)
            assert cp.get("models", "dir") == "/tmp/models"
            assert cp.get("decoder", "device") == "cuda:0"
        finally:
            os.unlink(path)

    def test_nonexistent_base_creates_minimal_conf(self, tmp_path):
        base = tmp_path / "nonexistent.conf"
        path = _write_temp_conf(base, "new_section", "key", "value")
        try:
            cp = configparser.ConfigParser()
            cp.read(path)
            assert cp.get("new_section", "key") == "value"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# RopeError
# ---------------------------------------------------------------------------

class TestRopeError:
    def test_is_runtime_error(self):
        err = RopeError(1, "server not found")
        assert isinstance(err, RuntimeError)

    def test_message_includes_error_name(self):
        err = RopeError(3, "query is out of range")
        assert "time out of range" in str(err)

    def test_message_includes_detail(self):
        err = RopeError(3, "query is out of range")
        assert "query is out of range" in str(err)

    def test_code_attribute(self):
        err = RopeError(4, "lat out of range")
        assert err.code == 4

    def test_unknown_code_uses_numeric(self):
        err = RopeError(99, "unexpected")
        assert "99" in str(err)

    @pytest.mark.parametrize("code,name", list(_ERR_NAMES.items()))
    def test_all_known_error_codes_have_names(self, code, name):
        err = RopeError(code, "detail")
        assert name in str(err)

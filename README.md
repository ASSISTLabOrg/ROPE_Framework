# ROPE

ROPE is a tool for forecasting upper-atmosphere density. Given a start time and a forecast window, it produces a 3-D grid of density and uncertainty estimates that can be queried at any point within the grid.

## Notice

This is a beta release of the ROPE tool. It is not recommended for use in production until the full version is released. Breaking changes to the public API may be introduced during the beta testing phase.

## Contact Info

Contact [Violet Player](mailto:violet.player@noaa.gov) or [Piyush Mehta](mailto:piyush.mehta@mail.wvu.edu) with any questions. Technical support questions should be directed to [Violet Player](mailto:violet.player@noaa.gov).

## Requirements

- Linux (x86_64, glibc 2.31+), macOS 12+, or Windows 10+
- Python 3.8+ (python wrapper only)

## Installation

Download the release archive for your platform and hardware from [Google Drive](https://drive.google.com/drive/folders/1gVQ0gqzwfKDaZIuT8tWoCyWQOL_hT4Ol?usp=drive_link)

| Platform | File |
|----------|------|
| Linux, CPU | `rope_framework-<version>-linux-x86_64-cpu.tar.gz` |
| Linux, CUDA 12 | `rope_framework-<version>-linux-x86_64-cuda12.tar.gz` |
| macOS (Apple Silicon) | `rope_framework-<version>-macos-arm64-cpu.tar.gz` |
| Windows, CPU | `rope_framework-<version>-windows-x64-cpu.zip` |
| Windows, CUDA 12 | `rope_framework-<version>-windows-x64-cuda12.zip` |

**Currently Unsupported:** macOS (Intel), Linux (ARM64), ROCm, CUDA 13+

Extract the archive. The layout inside is:

```
rope_framework-<version>-<platform>/
    bin/        rope executable
    lib/        runtime libraries
    config/     rope.conf
    models/     (download separately)
    data/       (download separately)
    include/    C API header (rope.h)
```

**macOS only:** After extracting, macOS Gatekeeper will block the bundled `.dylib` files with `library load disallowed by system policy`. Clear the quarantine attribute before use:

```
xattr -r -d com.apple.quarantine rope_framework-<version>-macos-arm64-cpu/
```

**Download models and data separately.** The `models/` and `data/` directories are not included in the release archive. Download them from [Google Drive](https://drive.google.com/drive/folders/1gVQ0gqzwfKDaZIuT8tWoCyWQOL_hT4Ol?usp=drive_link) and place them inside the extracted folder alongside `bin/` and `lib/`.

Once in place the directory structure should look like the layout above.

## Configuration

The default config file is `config/rope.conf`. It is pre-configured to match the standard layout, so in most cases nothing needs to be changed.

If you need to change paths or thread counts, open the file in a text editor. The settings are documented with comments inside it.

The one setting you may want to change is `decoder.device`:

```
[decoder]
device = cpu    # or: cuda, cuda:0, cuda:1
```

This only has an effect on builds that include the LibTorch backend (the CPU and CUDA 12 Linux builds, and the macOS build).

## Python

A Python wrapper is included in the `python/` directory. Copy `rope.py` to your project or add `python/` to your Python path.

```python
from rope import Rope

r = Rope()
r.forecast("2024-06-01 00:00:00", horizon=24)

with r:
    result = r.get(time="2024-06-01T06:00:00", lst=12.5, lat=45.0, alt_km=400.0)
    print(result)  # {'density': 4.72e-12, 'uncertainty': 3.5e-13}

r.shutdown()
```

The `with` block opens an interpolation handle against the cached grid and closes it on exit. It does not stop the server. Call `r.shutdown()` when you are done to stop the server process. If you forget, the Python wrapper will call it automatically on interpreter exit, and the server will also stop itself after 30 minutes of inactivity. The idle timeout is configurable via `idle_timeout_seconds` in the `[server]` section of `rope.conf`; set it to `0` to disable.

The wrapper requires Python 3.8 or later and no additional packages.

## C API

For use from C or other languages, `include/rope/capi/rope.h` exposes a stable C-compatible API. The shared library is `lib/librope.so` (Linux), `lib/librope.dylib` (macOS), or `bin/rope.dll` (Windows).

## Command Line Interface

All commands are run through the `rope` executable in `bin/`. On Linux and macOS you may need to prefix commands with `./bin/rope` if the directory is not on your PATH.

### 1. Run a forecast

```
rope forecast --start "2024-06-01 00:00:00" --horizon 24
```

This runs a 24-hour forecast starting at the given UTC time and caches the result in memory. A background server process is started automatically on first use and remains running until you exit it.

`--start` accepts `YYYY-MM-DD HH:MM:SS` or `YYYY-MM-DDTHH:MM:SS` in UTC.

### 2. Query a point

```
rope get --mode interp --time "2024-06-01T06:00:00" --lst 12.5 --lat 45.0 --alt 400.0
```

Returns density and uncertainty at the requested position, interpolated from the cached forecast.

| Argument | Description |
|----------|-------------|
| `--mode interp` | Interpolate in time between model hours |
| `--mode hold` | Snap to the next model hour (no time blending) |
| `--time` | UTC time within the forecast window |
| `--lst` | Local Solar Time, hours [0, 24) |
| `--lat` | Geodetic latitude, degrees [-87.5, 87.5] |
| `--alt` | Altitude, km [100, 980] |

Output is a JSON object on stdout:

```json
{"density": 4.72e-12, "uncertainty": 3.5e-13}
```

Density and uncertainty are in kg/m³.

### Batch queries

To query many points at once, prepare a CSV file with the columns `YYYY,MM,DD,HH,MIN,SS,lst,lat,alt_km` and pass it with `--file`:

```
rope get --mode interp --file queries.csv --output results.csv
```

### Stop the server

```
rope exit
```

### Use a non-default config

```
rope forecast --config /path/to/rope.conf --start "2024-06-01 00:00:00" --horizon 24
```

## Demo

The `demo/` directory contains a Jupyter notebook that benchmarks ROPE against NRLMSIS-2.1 atmospheric drag across two scenarios: forecast and interpolation throughput (24-hour forecast, 10,000 random point queries), and a 1-day ISS-like orbit integration comparing altitude decay between the two models.

Running the demo scrips is **not** required to use any of the rope library, but may be beneficial as a sanity check to ensure your system is functioning.

To set it up, install the demo dependencies from the archive root:

```
bash demo/install.sh
```

This requires Python 3.8+ and `pip`. It installs the Python wrapper from `python/` and the demo support library from `demo/`. The notebook fetches space weather data from CelesTrak on first run and caches it locally, so an internet connection is needed the first time.

Then open the notebook:

```
bash demo/run_benchmark.sh
```

The notebook uses `demo/rope.conf`, which is pre-configured to point at the `models/` and `data/` directories in the standard archive layout. If you extracted the archive somewhere else, edit the paths in that file before running.

## License

MIT. See `LICENSE`.

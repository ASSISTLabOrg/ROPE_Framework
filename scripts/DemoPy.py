from time import perf_counter

import numpy as np

from rope import ROPE
from interpolator import DensityInterpolator

print("=== ROPE Python Demo ===\n")

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
t1 = perf_counter()
rope = ROPE(device="cpu")
t2 = perf_counter()
print(f"Load time : {t2 - t1:.4f} s\n")

# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------
res = rope.run("2024-02-09 00:00:00", horizon=120)
t3 = perf_counter()

d = np.asarray(res["meta_density"])
print(f"Run time  : {t3 - t2:.4f} s")
print(
    f"Density   : min={d.min():.3e}  max={d.max():.3e}  mean={d.mean():.3e}\n"
)

# ---------------------------------------------------------------------------
# Grid bounds
# ---------------------------------------------------------------------------
q = DensityInterpolator(res)
b = q.bounds()
print(
    "Grid bounds:\n"
    f"  LST  : [{b['lst_min']:.4g}, {b['lst_max']:.4g}] h\n"
    f"  lat  : [{b['lat_min']:.4g}, {b['lat_max']:.4g}] deg\n"
    f"  alt  : [{b['alt_km_min']:.4g}, {b['alt_km_max']:.4g}] km\n"
    f"  time : [{b['time_min']}  to  {b['time_max']}]\n"
)

# ---------------------------------------------------------------------------
# Point queries
# ---------------------------------------------------------------------------
t4 = perf_counter()

out1 = q.query("2024-02-10 00:30:00", lst=10.5, lat=25.0, alt_km=400.0,
               time_mode="hold_next_hour")
out2 = q.query("2024-02-10 00:30:00", lst=10.5, lat=25.0, alt_km=400.0,
               time_mode="interp_time")

t5 = perf_counter()

print(
    "hold_next_hour:\n"
    f"  requested : {out1['datetime_requested']}\n"
    f"  used      : {out1['datetime_used']}\n"
    f"  density   : {out1['density']:.4e} kg/m³\n"
    f"  t_index   : {out1['t_index']}\n"
)
print(
    "interp_time:\n"
    f"  datetime  : {out2['datetime']}\n"
    f"  density   : {out2['density']:.4e} kg/m³\n"
    f"  left  t={out2['t_index_left']} ({out2['datetime_left']})\n"
    f"  right t={out2['t_index_right']} ({out2['datetime_right']})\n"
    f"  w_right   : {out2['time_weight_right']:.4f}\n"
)

print(f"Interp time : {t5 - t4:.4f} s")

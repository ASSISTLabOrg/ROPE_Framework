"""
sensitivity.py
-----------

Date: 2026-04-13

sensitivity study comparing USSA76 vs NRLMSIS-2.1 using real historical
solar/geomagnetic conditions fetched from CelesTrak for eight well-known
solar cycle phases between 2000 and 2024.

All F10.7, F10.7a, and 3-hourly Ap values come directly from the CelesTrak
SW-All.csv archive — nothing is hard-coded except the epoch dates.

Historical cases
----------------
  SC23 max          2003-10-28  (peak SSN=180.3, strong X-flare period)
  SC23/24 min       2008-12-01  (817 spotless days, deepest min in decades)
  SC24 rising       2012-03-15  (fast rise, first SSN peak)
  SC24 max          2014-04-15  (weakest max since 1906, SSN=116.4)
  SC24/25 min       2019-12-15  (deepest minimum in ~100 years)
  SC25 rising       2022-02-03  (Starlink event, enhanced drag at 210 km)
  SC25 max          2024-05-10  (strongest geomagnetic storm in 20 years, Ap~236)
  SC25 high-F10.7   2024-10-03  (daily F10.7>300 sfu, SC25 smoothed peak)

Figures
-------
  sensitivity_static.png      — density vs altitude (A) + geographic ratio (B)
  sensitivity_performance.png — ODE wall-clock timing (C) + altitude decay (D)
  sensitivity_trajectory.png  — density + altitude time-series along orbit (E)

Usage
-----
    python sensitivity.py
    python sensitivity.py --alt 450 --days 2 --dt 60
    python sensitivity.py --no-cache    # force re-download of CelesTrak data
"""

import argparse
import time
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.integrate import solve_ivp

from demo_lib.perturbations import (
    ussa76, msis_density,
    perturbations_ussa76, perturbations_msis,
    eci_to_geodetic, _MU, _RE,
)
from demo_lib.space_weather import SpaceWeather

# ---------------------------------------------------------------------------
# Historical cases — change dates here only; everything else auto-populates
# ---------------------------------------------------------------------------

CASES = [
    # (short_label,           epoch_str,            color)
    ("SC23 max\n2003-10-28",  "2003-10-28T00:00:00", "#B71C1C"),
    ("SC23/24 min\n2008-12-01","2008-12-01T00:00:00", "#1565C0"),
    ("SC24 rising\n2012-03-15","2012-03-15T00:00:00", "#0277BD"),
    ("SC24 max\n2014-04-15",  "2014-04-15T00:00:00", "#00695C"),
    ("SC24/25 min\n2019-12-15","2019-12-15T00:00:00", "#4527A0"),
    ("SC25 rising\n2022-02-03","2022-02-03T00:00:00", "#558B2F"),
    ("SC25 storm\n2024-05-10","2024-05-10T18:00:00", "#E65100"),
    ("SC25 max\n2024-10-03",  "2024-10-03T00:00:00", "#6A1B9A"),
]

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8.5,
    "figure.titlesize": 13,
    "figure.titleweight": "bold",
})
_GRAY  = "#9E9E9E"
_BLACK = "#212121"


def _ax_style(ax, ylabel):
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", lw=0.35, color=_GRAY, alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------------------
# Load SW data (SW-All needed for 2001)
# ---------------------------------------------------------------------------

def load_sw(force_refresh=False) -> SpaceWeather:
    print("[space weather] loading CelesTrak SW-All.csv ...")
    sw = SpaceWeather(force_refresh=force_refresh, use_all=True)
    # Eagerly fetch and print indices for all cases so user can verify
    print(f"\n  {'Case':<28} {'Date':<12} {'F10.7':>7} {'F10.7a':>7} "
          f"{'Ap(day)':>8} {'ap(now)':>8}")
    print("  " + "-" * 72)
    for label, epoch_str, _ in CASES:
        epoch = datetime.fromisoformat(epoch_str)
        f107, f107a, ap7 = sw.get(epoch)
        print(f"  {label.replace(chr(10),' '):<28} {str(epoch.date()):<12} "
              f"{f107:>7.1f} {f107a:>7.1f} {ap7[0]:>8.0f} {ap7[1]:>8.0f}")
    print()
    return sw


# ---------------------------------------------------------------------------
# ODE helpers
# ---------------------------------------------------------------------------

def make_ic(alt_km, inc_deg=51.6):
    inc = np.radians(inc_deg)
    r0  = np.array([_RE + alt_km, 0.0, 0.0])
    vc  = np.sqrt(_MU / np.linalg.norm(r0))
    v0  = np.array([0.0, vc * np.cos(inc), vc * np.sin(inc)])
    return np.concatenate([r0, v0])


def eom_76(t, state, bstar, rho_log):
    r, v = state[:3], state[3:]
    ag   = -_MU / np.linalg.norm(r)**3 * r
    ap, rho = perturbations_ussa76(r, v, bstar)
    rho_log.append(rho)
    return np.concatenate([v, ag + ap])


def eom_ms(t, state, bstar, epoch, f107, f107a, ap7, rho_log):
    r, v   = state[:3], state[3:]
    ag     = -_MU / np.linalg.norm(r)**3 * r
    date   = epoch + timedelta(seconds=float(t))
    ap_acc, rho = perturbations_msis(r, v, bstar,
                                     date=date, f107=f107,
                                     f107a=f107a, ap7=ap7)
    rho_log.append(rho)
    return np.concatenate([v, ag + ap_acc])


def run_integration(model, x0, t_end, dt, bstar, epoch, f107, f107a, ap7,
                    method="RK45"):
    """Run one ODE integration. Returns (sol, rho_on_grid, wall_s)."""
    t_eval  = np.arange(0.0, t_end + dt, dt)
    rho_log = []

    if model == "ussa76":
        fn, args = eom_76, (bstar, rho_log)
    else:
        fn, args = eom_ms, (bstar, epoch, f107, f107a, ap7, rho_log)

    t0  = time.perf_counter()
    sol = solve_ivp(fn, (0.0, t_end), x0, method=method,
                    t_eval=t_eval, args=args,
                    rtol=1e-9, atol=1e-9, dense_output=False)
    wall = time.perf_counter() - t0

    rho_on_grid = np.interp(
        t_eval,
        np.linspace(0.0, t_end, len(rho_log)),
        np.array(rho_log),
    )
    return sol, rho_on_grid, wall


# ===========================================================================
# Panel A — density vs altitude, all historical epochs + USSA76
# ===========================================================================

def panel_a(ax, sw, alts=None):
    if alts is None:
        alts = np.linspace(150, 700, 200)

    rho_76 = np.array([ussa76(z) for z in alts])
    ax.semilogy(alts, rho_76, color=_BLACK, lw=2.2, ls="-",
                label="USSA76  (no solar dependence)", zorder=10)

    for label, epoch_str, color in CASES:
        epoch = datetime.fromisoformat(epoch_str)
        f107, f107a, ap7 = sw.get(epoch)
        rho = np.array([
            msis_density(z, 0.0, 0.0, epoch, f107, f107a, ap7)
            for z in alts
        ])
        clean = label.replace("\n", " ")
        ax.semilogy(alts, rho, lw=1.3, ls="--", color=color,
                    label=f"MSIS  {clean}  (F10.7={f107:.0f}, Ap={ap7[0]:.0f})")

    ax.set_xlabel("Altitude [km]")
    _ax_style(ax, r"$\rho$  [kg m$^{-3}$]")
    ax.set_title("A — Density vs altitude  (lon=0°, lat=0°)")
    ax.legend(fontsize=8, frameon=False)


# ===========================================================================
# Panel B — geographic MSIS/USSA76 ratio at LEO alt for selected cases
# Uses the two most extreme cases: SC23/24 min and SC25 storm
# ===========================================================================

def panel_b(axes_row, sw, alt):
    """axes_row : list of 2 axes for the two extreme cases."""
    extreme_cases = [
        CASES[1],   # SC23/24 min — quietest
        CASES[6],   # SC25 storm  — most active
    ]

    lats = np.linspace(-90, 90, 37)
    lons = np.linspace(-180, 180, 73)
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    rho_76 = ussa76(alt)

    for ax, (label, epoch_str, color) in zip(axes_row, extreme_cases):
        epoch = datetime.fromisoformat(epoch_str)
        f107, f107a, ap7 = sw.get(epoch)

        ratio = np.zeros_like(LAT)
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                ratio[i, j] = msis_density(
                    alt, lon, lat, epoch, f107, f107a, ap7) / rho_76

        cf = ax.contourf(LON, LAT, ratio,
                         levels=np.linspace(ratio.min(), ratio.max(), 20),
                         cmap="RdBu_r")
        plt.colorbar(cf, ax=ax,
                     label=r"$\rho_{\rm MSIS}\,/\,\rho_{\rm USSA76}$",
                     pad=0.02)
        ax.contour(LON, LAT, ratio, levels=[1.0],
                   colors="white", linewidths=0.9, linestyles="--")
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        clean = label.replace("\n", " ")
        ax.set_title(f"B — Geographic ratio at {alt:.0f} km\n"
                     f"{clean}  F10.7={f107:.0f}  Ap={ap7[0]:.0f}  "
                     f"range {ratio.min():.2f}–{ratio.max():.2f}",
                     pad=12)
        ax.spines[["top", "right"]].set_visible(False)


# ===========================================================================
# Panel C — ODE wall-clock timing
# ===========================================================================

def panel_c(ax, sw, alt, bstar, days, dt, method):
    x0    = make_ic(alt)
    t_end = days * 86400.0

    # USSA76 first (single baseline)
    print("  [C] timing USSA76 ...", flush=True)
    _, _, t_76 = run_integration("ussa76", x0, t_end, dt, bstar,
                                 datetime(2020, 1, 1), 150, 150, [4]*7, method)

    case_labels, case_times = ["USSA76"], [t_76]

    for label, epoch_str, color in CASES:
        clean = label.replace("\n", " ")
        print(f"  [C] timing MSIS {clean} ...", flush=True)
        epoch = datetime.fromisoformat(epoch_str)
        f107, f107a, ap7 = sw.get(epoch)
        _, _, wall = run_integration("msis", x0, t_end, dt, bstar,
                                     epoch, f107, f107a, ap7, method)
        case_labels.append(label)
        case_times.append(wall)

    colors = [_BLACK] + [c for *_, c in CASES]
    bars   = ax.bar(range(len(case_labels)), case_times,
                    color=colors,alpha=0.3,
                    edgecolor="white", linewidth=0.5)

    for bar, t in zip(bars, case_times):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f"{t:.2f}s", ha="center", va="bottom", fontsize=8)

    speedup_min = min(case_times[1:]) / t_76
    speedup_max = max(case_times[1:]) / t_76
    ax.set_xticks(range(len(case_labels)))
    ax.set_xticklabels(case_labels, fontsize=8)
    ax.set_ylabel("Wall-clock time [s]")
    ax.set_ylim(0, max(case_times) * 1.35)
    ax.set_title(f"C — ODE integration time  ({days}d, dt={dt:.0f}s, {method})\n"
                 f"MSIS overhead: ×{speedup_min:.1f}–×{speedup_max:.1f} vs USSA76",
                 pad=12)
    ax.grid(True, axis="y", lw=0.35, color=_GRAY, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)


# ===========================================================================
# Panel D — altitude decay per case (USSA76 fixed, MSIS per epoch)
# ===========================================================================

def panel_d(ax, sw, alt, bstar, days, dt, method):
    x0    = make_ic(alt)
    t_end = days * 86400.0

    print("  [D] USSA76 baseline ...", flush=True)
    sol_76, _, _ = run_integration("ussa76", x0, t_end, dt, bstar,
                                   datetime(2020, 1, 1), 150, 150, [4]*7, method)
    da_76 = np.linalg.norm(sol_76.y[:3, -1]) - _RE - alt

    da_ms_vals = []
    for label, epoch_str, color in CASES:
        clean = label.replace("\n", " ")
        print(f"  [D] MSIS {clean} ...", flush=True)
        epoch = datetime.fromisoformat(epoch_str)
        f107, f107a, ap7 = sw.get(epoch)
        sol_ms, _, _ = run_integration("msis", x0, t_end, dt, bstar,
                                       epoch, f107, f107a, ap7, method)
        da_ms_vals.append(np.linalg.norm(sol_ms.y[:3, -1]) - _RE - alt)

    x        = np.arange(len(CASES))
    diff     = np.array(da_ms_vals) - da_76
    colors   = [c for *_, c in CASES]
    clean_labels = [lbl.replace("\n", " ") for lbl, *_ in CASES]

    bars = ax.bar(x, da_ms_vals, color=colors,alpha=0.3,
                  edgecolor="white", linewidth=0.5, label="MSIS")
    ax.axhline(da_76, color=_BLACK, lw=1.8, ls="--",
               label=f"USSA76  ({da_76:.2f} km/day)")

    ax2 = ax.twinx()
    ax2.plot(x, diff, "D--", color=_GRAY, ms=5, lw=1.0,
             label="ΔMSIS − USSA76")
    ax2.axhline(0, color=_GRAY, lw=0.6, ls=":")
    ax2.set_ylabel("ΔMSIS − USSA76  [km/day]",
                   color=_GRAY, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=_GRAY, labelsize=10)
    ax2.spines[["top"]].set_visible(False)

    ax.set_xticks(x)
    ax.set_xticklabels(clean_labels, fontsize=7.5, rotation=15, ha="right")
    ax.set_ylabel("Altitude decay  Δalt [km/day]")
    ax.set_title(f"D — Altitude decay per historical epoch\n"
                 f"alt={alt:.0f} km, BStar={bstar:.1e}, {days}d",
                 pad=12)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8.5, frameon=False)
    ax.grid(True, axis="y", lw=0.35, color=_GRAY, alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)


# ===========================================================================
# Panel E — density + altitude time-series for subset of cases
# ===========================================================================

def panel_e(axes, sw, alt, bstar, days, dt, method):
    """axes : [ax_rho, ax_alt, ax_ratio]  (sharex)"""
    x0    = make_ic(alt)
    t_end = days * 86400.0

    # Run USSA76 once
    print("  [E] USSA76 ...", flush=True)
    sol_76, rho_76, wall_76 = run_integration(
        "ussa76", x0, t_end, dt, bstar,
        datetime(2020, 1, 1), 150, 150, [4]*7, method)
    t_hr  = sol_76.t / 3600.0
    alt_76 = np.linalg.norm(sol_76.y[:3].T, axis=1) - _RE

    axes[0].semilogy(t_hr, rho_76, color=_BLACK, lw=1.8, ls="-",
                     label=f"USSA76  ({wall_76:.1f}s)")
    axes[1].plot(t_hr, alt_76, color=_BLACK, lw=1.8, ls="-")

    for label, epoch_str, color in CASES:
        clean = label.replace("\n", " ")
        print(f"  [E] MSIS {clean} ...", flush=True)
        epoch = datetime.fromisoformat(epoch_str)
        f107, f107a, ap7 = sw.get(epoch)
        sol_ms, rho_ms, wall_ms = run_integration(
            "msis", x0, t_end, dt, bstar,
            epoch, f107, f107a, ap7, method)

        alt_ms = np.linalg.norm(sol_ms.y[:3].T, axis=1) - _RE
        ratio  = rho_ms / np.where(rho_76 > 0, rho_76, np.nan)

        axes[0].semilogy(t_hr, rho_ms, lw=0.9, ls="--", color=color,
                         label=f"MSIS {clean}  ({wall_ms:.1f}s)")
        axes[1].plot(t_hr, alt_ms, lw=0.9, ls="--", color=color)
        axes[2].plot(t_hr, ratio, lw=0.9, ls="--", color=color,
                     label=clean)

    axes[2].axhline(1.0, color=_GRAY, lw=0.9, ls=":")

    axes[0].legend(fontsize=7.5, frameon=False, ncol=2)
    axes[2].legend(fontsize=7.5, frameon=False, ncol=2)

    _ax_style(axes[0], r"$\rho$  [kg m$^{-3}$]")
    _ax_style(axes[1], "Altitude [km]")
    _ax_style(axes[2], r"$\rho_{\rm MSIS}\,/\,\rho_{\rm USSA76}$")
    axes[0].set_title(f"E — Density and altitude along orbit  "
                      f"(alt={alt:.0f} km, BStar={bstar:.1e})")
    axes[2].set_xlabel("Time [hr]")


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="USSA76 vs MSIS sensitivity — historical solar cycle epochs")
    p.add_argument("--alt",    type=float, default=420.0,
                   help="Initial altitude [km], default 420")
    p.add_argument("--bstar",  type=float, default=2.5e-5)
    p.add_argument("--days",   type=float, default=1.0)
    p.add_argument("--dt",     type=float, default=60.0)
    p.add_argument("--method", type=str,   default="RK45",
                   choices=["RK45", "DOP853", "Radau", "LSODA"])
    p.add_argument("--no-cache", action="store_true",
                   help="Force re-download of CelesTrak SW data")
    p.add_argument("--out-prefix", type=str, default="sensitivity")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print("sensitivity STUDY — USSA76 vs NRLMSIS-2.1")
    print(f"  alt={args.alt} km  BStar={args.bstar:.2e}  "
          f"days={args.days}  dt={args.dt}s  solver={args.method}")
    print("  Epochs: 8 historical solar cycle phases, 2001–2024")
    print("=" * 65 + "\n")

    sw = load_sw(force_refresh=args.no_cache)

    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)

    # -----------------------------------------------------------------------
    # Figure 1 — static: density profile + geographic ratio
    # -----------------------------------------------------------------------
    print("[Figure 1] Static model properties ...")
    fig1 = plt.figure(figsize=(16, 10), layout="constrained")
    gs   = fig1.add_gridspec(2, 2, hspace=0.38, wspace=0.35)
    ax_a    = fig1.add_subplot(gs[:, 0])         # full left column
    ax_b0   = fig1.add_subplot(gs[0, 1])
    ax_b1   = fig1.add_subplot(gs[1, 1])
    fig1.suptitle("USSA76 vs NRLMSIS-2.1 — Static model properties")

    panel_a(ax_a, sw)
    panel_b([ax_b0, ax_b1], sw, alt=args.alt)

    out1 = os.path.join("output", f"{args.out_prefix}_static.png")
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved: {out1}")

    # -----------------------------------------------------------------------
    # Figure 2 — ODE timing + altitude decay
    # -----------------------------------------------------------------------
    print("\n[Figure 2] Integration performance and decay ...")
    fig2, (ax_c, ax_d) = plt.subplots(1, 2, figsize=(16, 6), layout="constrained")
    fig2.suptitle("USSA76 vs NRLMSIS-2.1 — Integration performance & decay")

    panel_c(ax_c, sw, alt=args.alt, bstar=args.bstar,
            days=args.days, dt=args.dt, method=args.method)
    panel_d(ax_d, sw, alt=args.alt, bstar=args.bstar,
            days=args.days, dt=args.dt * 2, method=args.method)

    out2 = os.path.join("output", f"{args.out_prefix}_performance.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {out2}")

    # -----------------------------------------------------------------------
    # Figure 3 — density + altitude time-series
    # -----------------------------------------------------------------------
    print("\n[Figure 3] Trajectory time series ...")
    fig3, axes_e = plt.subplots(3, 1, figsize=(11, 11), sharex=True, layout="constrained")
    fig3.suptitle("USSA76 vs NRLMSIS-2.1 — Density & altitude along orbit\n"
                  f"alt={args.alt:.0f} km  BStar={args.bstar:.1e}  "
                  f"duration={args.days} day(s)")

    panel_e(axes_e, sw, alt=args.alt, bstar=args.bstar,
            days=args.days, dt=args.dt, method=args.method)

    out3 = os.path.join("output", f"{args.out_prefix}_trajectory.png")
    fig3.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved: {out3}")

    print("\nDone.")


if __name__ == "__main__":
    main()
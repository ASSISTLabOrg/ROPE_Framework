"""
integrate_orbit.py
------------------

Date: 2026-04-13

Integrates a perturbed orbit under two drag models (USSA76 and NRLMSIS-2.1)
from the same initial condition.

Space weather (F10.7, Ap) is fetched automatically from CelesTrak for the
propagation epoch, so MSIS gets accurate solar/geomagnetic drivers rather
than fixed guesses.  The data is cached locally after the first download.

Produces two figures:

  Figure 1 — {prefix}_comparison.png
      Altitude, eccentricity, inclination vs time for both models.

  Figure 2 — {prefix}_density.png
      Density at actual satellite position (log), altitude, geographic
      track (lat/lon), and density ratio MSIS/USSA76.

Usage
-----
    python integrate_orbit.py
    python integrate_orbit.py --epoch 2023-10-05T00:00:00 --days 2
    python integrate_orbit.py --ic rx ry rz vx vy vz --bstar 1e-4
    python integrate_orbit.py --no-sw           # fall back to F10.7=150, Ap=4
    python integrate_orbit.py --refresh-sw      # force re-download of SW data
"""

import argparse
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.titleweight': 'bold',
    'font.size': 12,
    'legend.fontsize': 11,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
})
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
from scipy.integrate import solve_ivp

from demo_lib.perturbations import (
    perturbations_ussa76, perturbations_msis, perturbations_rope,
    eci_to_geodetic,
    _MU, _RE,
)
from rope import Rope

# ---------------------------------------------------------------------------
# Equations of motion
# ---------------------------------------------------------------------------

def eom_ussa76(t, state, bstar, rho_log):
    r, v   = state[:3], state[3:]
    a_grav = -_MU / np.linalg.norm(r)**3 * r
    a_pert, rho = perturbations_ussa76(r, v, bstar)
    rho_log.append((t, rho))
    return np.concatenate([v, a_grav + a_pert])


def eom_msis(t, state, bstar, epoch, sw_engine, rho_log, geo_log):
    r, v         = state[:3], state[3:]
    a_grav       = -_MU / np.linalg.norm(r)**3 * r
    current_date = epoch + timedelta(seconds=float(t))

    # FAST LOOKUP: No datetimes, no pandas
    f107, f107a, ap7 = sw_engine.query(t)

    a_pert, rho = perturbations_msis(r, v, bstar,
                                     date=current_date,
                                     f107=f107, f107a=f107a, ap7=ap7)
    lon, lat, alt = eci_to_geodetic(r, current_date)
    rho_log.append((t, rho))
    geo_log.append((t, lon, lat, alt))
    return np.concatenate([v, a_grav + a_pert])


def eom_rope(t, state, bstar, epoch, rp, rho_log):
    r, v   = state[:3], state[3:]
    a_grav = -_MU / np.linalg.norm(r)**3 * r
    current_date = epoch + timedelta(seconds=float(t))

    a_pert, rho = perturbations_rope(r, v, bstar, current_date, rp)
    rho_log.append((t, rho))
    return np.concatenate([v, a_grav + a_pert])


# ---------------------------------------------------------------------------
# Orbital elements
# ---------------------------------------------------------------------------

def rv_to_elements(r, v):
    r_n    = np.linalg.norm(r)
    energy = 0.5 * np.linalg.norm(v)**2 - _MU / r_n
    a      = -_MU / (2.0 * energy)
    h      = np.cross(r, v)
    e      = np.linalg.norm(np.cross(v, h) / _MU - r / r_n)
    inc    = np.degrees(np.arccos(np.clip(h[2] / np.linalg.norm(h), -1, 1)))
    alt    = r_n - _RE
    return a, e, inc, alt


def elements_from_trajectory(r_hist, v_hist):
    out = np.array([rv_to_elements(r_hist[k], v_hist[k])
                    for k in range(len(r_hist))])
    return out[:, 0], out[:, 1], out[:, 2], out[:, 3]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def interp_log(log, t_eval):
    if not log:
        return np.full_like(t_eval, np.nan)
    arr = np.array(log)
    return np.interp(t_eval, arr[:, 0], arr[:, 1])


def interp_geo(geo_log, t_eval):
    if not geo_log:
        return np.zeros_like(t_eval), np.zeros_like(t_eval)
    arr = np.array(geo_log)
    return np.interp(t_eval, arr[:, 0], arr[:, 1]), \
           np.interp(t_eval, arr[:, 0], arr[:, 2])


_BLUE   = "#1565C0"
_ORANGE = "#E65100"
_PURPLE = "#4A148C"
_GRAY   = "#9E9E9E"


def _ax_style(ax, ylabel):
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.grid(True, lw=0.5, color=_GRAY, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------------------
# Space-weather fallback (no-network / --no-sw)
# ---------------------------------------------------------------------------

class _FallbackSW:
    """Returns fixed moderate indices when real data is unavailable."""
    def get(self, dt):
        return 150.0, 150.0, [4.0] * 7

    def summary(self, dt):
        return f"  [fallback]  F10.7=150.0  F10.7a=150.0  Ap=4"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="USSA76 vs MSIS orbit comparison")
    p.add_argument("--ic", nargs=6, type=float,
                   metavar=("RX","RY","RZ","VX","VY","VZ"))
    p.add_argument("--bstar",   type=float, default=2.5e-5)
    p.add_argument("--days",    type=float, default=1.0)
    p.add_argument("--dt",      type=float, default=60.0)
    p.add_argument("--epoch",   type=str,   default="2020-01-01T00:00:00",
                   help="UTC epoch ISO string (default: 2020-01-01T00:00:00)")
    p.add_argument("--method",  type=str,   default="LSODA",
                   choices=["RK45","DOP853","Radau","LSODA"])
    p.add_argument("--rtol",    type=float, default=1e-9)
    p.add_argument("--atol",    type=float, default=1e-9)
    p.add_argument("--no-sw",   action="store_true",
                   help="Skip CelesTrak download; use F10.7=150, Ap=4")
    p.add_argument("--refresh-sw", action="store_true",
                   help="Force re-download of space weather cache")
    p.add_argument("--use-all-sw", action="store_true",
                   help="Use SW-All.csv (back to 1957) instead of last 5 years")
    p.add_argument("--out-prefix", type=str, default="orbit")
    return p.parse_args()


def main():
    args  = parse_args()
    epoch = datetime.fromisoformat(args.epoch)

    # --- Space weather ------------------------------------------------------
    if args.no_sw:
        sw = _FallbackSW()
        print("[space weather] using fallback values (F10.7=150, Ap=4)")
    else:
        from demo_lib.space_weather import SpaceWeather
        print("[space weather] loading CelesTrak data ...")
        sw = SpaceWeather(force_refresh=args.refresh_sw,
                          use_all=args.use_all_sw)
    print(sw.summary(epoch))

    # Report the actual drivers at epoch
    f107_ep, f107a_ep, ap7_ep = sw.get(epoch)
    print(f"  Epoch drivers: F10.7={f107_ep:.1f}  F10.7a={f107a_ep:.1f}  "
          f"Ap(daily)={ap7_ep[0]:.0f}  ap(now)={ap7_ep[1]:.0f}")

    # --- Initial condition ---------------------------------------------------
    if args.ic is not None:
        x0 = np.array(args.ic, dtype=float)
    else:
        inc_rad = np.radians(51.6)
        r0 = np.array([_RE + 420.0, 0.0, 0.0])
        vc = np.sqrt(_MU / np.linalg.norm(r0))
        v0 = np.array([0.0, vc * np.cos(inc_rad), vc * np.sin(inc_rad)])
        x0 = np.concatenate([r0, v0])
        print("\n[default IC] ISS-like orbit, alt~420 km, i~51.6 deg")

    t_end  = args.days * 86400.0
    t_eval = np.arange(0.0, t_end + args.dt, args.dt)

    a0, e0, i0, alt0 = rv_to_elements(x0[:3], x0[3:])
    print(f"  IC: a={a0:.3f} km  e={e0:.6f}  i={i0:.3f} deg  alt={alt0:.3f} km")
    print(f"  Span: {args.days} d  dt={args.dt} s  solver={args.method}")

    ivp_kw = dict(method=args.method, t_eval=t_eval,
                  rtol=args.rtol, atol=args.atol, dense_output=True)

    # --- USSA76 integration -------------------------------------------------
    print("\nIntegrating USSA76 ...", flush=True)
    rho_log_76 = []
    sol_76 = solve_ivp(eom_ussa76, (0.0, t_end), x0,
                       args=(args.bstar, rho_log_76), **ivp_kw)
    if not sol_76.success:
        sys.exit(f"USSA76 failed: {sol_76.message}")
    print(f"  {sol_76.nfev} evals,  {len(sol_76.t)} steps")

    # --- MSIS integration ---------------------------------------------------
    print("Integrating MSIS-2.1 ...", flush=True)
    from demo_lib.space_weather import SpaceWeatherEngine
    sw_engine = SpaceWeatherEngine(sw, epoch, args.days)

    rho_log_ms, geo_log_ms = [], []
    sol_ms = solve_ivp(eom_msis, (0.0, t_end), x0,
                       args=(args.bstar, epoch, sw_engine, rho_log_ms, geo_log_ms),
                       **ivp_kw)
    if not sol_ms.success:
        sys.exit(f"MSIS failed: {sol_ms.message}")
    print(f"  {sol_ms.nfev} evals,  {len(sol_ms.t)} steps")

    # Rope
    print("Integrating ROPE Model ...", flush=True)
    rho_log_rope = []
    rp = Rope(config_path=Path(__file__).parent / "rope.conf")
    rp.forecast(epoch, horizon=int(24.0 * args.days + 12)) # run the forecast
    with rp:
        sol_rope = solve_ivp(eom_rope, (0.0, t_end), x0,
                            args=(args.bstar, epoch, rp, rho_log_rope),
                            **ivp_kw)
    if not sol_rope.success:
        sys.exit(f"ROPE failed: {sol_rope.message}")
    print(f"  {sol_rope.nfev} evals,  {len(sol_rope.t)} steps")

    # --- Derived quantities -------------------------------------------------
    r76, v76 = sol_76.y[:3].T, sol_76.y[3:].T
    rms, vms = sol_ms.y[:3].T, sol_ms.y[3:].T
    rrope, vrope = sol_rope.y[:3].T, sol_rope.y[3:].T
    t_hr     = sol_76.t / 3600.0

    a76, e76, i76, alt76 = elements_from_trajectory(r76, v76)
    ams, ems, ims, altms = elements_from_trajectory(rms, vms)
    arope, erope, irope, altrope = elements_from_trajectory(rrope, vrope)

    rho76 = interp_log(rho_log_76, sol_76.t)
    rhoms = interp_log(rho_log_ms, sol_ms.t)
    rhorope = interp_log(rho_log_rope, sol_rope.t)
    lon_ms, lat_ms = interp_geo(geo_log_ms, sol_ms.t)

    ratio = rhoms / np.where(rho76 > 0, rho76, np.nan)

    print(f"\n  USSA76   final a={a76[-1]:.3f} km  Δa={a76[-1]-a76[0]:.3f} km")
    print(f"  MSIS-2.1 final a={ams[-1]:.3f} km  Δa={ams[-1]-ams[0]:.3f} km")
    print(f"  Density ratio: mean={np.nanmean(ratio):.4f}  "
          f"min={np.nanmin(ratio):.4f}  max={np.nanmax(ratio):.4f}")

    sw_label = (f"F10.7={f107_ep:.0f}  F10.7a={f107a_ep:.0f}  "
                f"Ap={ap7_ep[0]:.0f}  [CelesTrak]")
    title_base = (f"BStar={args.bstar:.2e}  {sw_label}\n"
                  f"epoch={epoch.date()}")

    # =======================================================================
    # Figure 1 — orbit comparison
    # =======================================================================
    fig1, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    fig1.suptitle(f"Perturbed orbit — USSA76 vs NRLMSIS-2.1\n{title_base}",
                  fontsize=14, fontweight="bold")

    # axes[0].plot(t_hr, a76, lw=1.0, color=_BLUE,   label="USSA76")
    axes[0].plot(t_hr, ams, lw=1.0, color=_ORANGE, label="MSIS-2.1", ls="--")
    axes[0].plot(t_hr, arope, lw=1.0, color=_PURPLE, label="ROPE-0.2", ls=":")
    axes[0].legend(fontsize=11, frameon=False)
    _ax_style(axes[0], "Semi-major axis [km]")

    # axes[1].plot(t_hr, e76, lw=1.0, color=_BLUE)
    axes[1].plot(t_hr, ems, lw=1.0, color=_ORANGE, ls="--")
    axes[1].plot(t_hr, erope, lw=1.0, color=_PURPLE, ls=':')
    _ax_style(axes[1], "Eccentricity")

    # axes[2].plot(t_hr, i76, lw=1.0, color=_BLUE)
    axes[2].plot(t_hr, ims, lw=1.0, color=_ORANGE, ls="--")
    axes[2].plot(t_hr, irope, lw=1.0, color=_PURPLE, ls=':')
    axes[2].set_xlabel("Time [hr]", fontsize=12, fontweight="bold")
    _ax_style(axes[2], "Inclination [deg]")

    # --- Save ---------------------------------------------------------------
    os.makedirs("output", exist_ok=True)

    fig1.tight_layout()
    out1 = os.path.join("output", f"{args.out_prefix}_comparison.png")
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"\n  Saved: {out1}")

    # =======================================================================
    # Figure 2 — density at actual position
    # =======================================================================
    fig2, axes2 = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    fig2.suptitle(
        f"Atmospheric density at satellite position — USSA76 vs NRLMSIS-2.1\n"
        f"{title_base}",
        fontsize=14,
        fontweight="bold",
    )

    # axes2[0].semilogy(t_hr, rho76, lw=0.9, color=_BLUE,
    #                   label="USSA76  (altitude only)")
    axes2[0].semilogy(t_hr, rhoms, lw=0.9, color=_ORANGE, ls="--",
                      label="MSIS-2.1  (lon / lat / alt / time, real F10.7 & Ap)")
    axes2[0].semilogy(t_hr, rhorope, lw=0.9, color=_PURPLE, ls=":", label="rope-0.2")
    axes2[0].legend(fontsize=11, frameon=False)
    _ax_style(axes2[0], r"$\rho$  [kg m$^{-3}$]")

    axes2[1].plot(t_hr, alt76, lw=0.9, color=_BLUE)
    axes2[1].plot(t_hr, altms, lw=0.9, color=_ORANGE, ls="--")
    _ax_style(axes2[1], "Altitude [km]")

    axes2[2].plot(t_hr, lat_ms, lw=0.9, color="#00695C", label="latitude")
    ax2r = axes2[2].twinx()
    ax2r.plot(t_hr, lon_ms, lw=0.6, color="#BF360C", alpha=0.6, label="longitude")
    axes2[2].set_ylabel("Geocentric latitude [deg]", fontsize=12, color="#00695C", fontweight="bold")
    ax2r.set_ylabel("Longitude [deg]", fontsize=12, color="#BF360C", fontweight="bold")
    axes2[2].tick_params(axis="y", labelcolor="#00695C", labelsize=11)
    ax2r.tick_params(axis="y", labelcolor="#BF360C",  labelsize=11)
    axes2[2].grid(True, lw=0.5, color=_GRAY, alpha=0.6)
    axes2[2].spines[["top"]].set_visible(False)
    lines  = axes2[2].get_lines() + ax2r.get_lines()
    axes2[2].legend(lines, [l.get_label() for l in lines],
                    fontsize=11, frameon=False, loc="upper right")

    axes2[3].axhline(1.0, color=_GRAY, lw=0.8, ls=":")
    axes2[3].plot(t_hr, ratio, lw=0.9, color=_PURPLE)
    axes2[3].set_xlabel("Time [hr]", fontsize=12, fontweight="bold")
    _ax_style(axes2[3], r"$\rho_{\rm MSIS}\,/\,\rho_{\rm USSA76}$")

    fig2.tight_layout()
    out2 = os.path.join("output", f"{args.out_prefix}_density.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {out2}")

    # --- Save data ----------------------------------------------------------
    out_data = os.path.join("output", f"{args.out_prefix}_data.npz")
    np.savez(
        out_data,
        t=sol_76.t,
        r_76=r76, v_76=v76, alt_76=alt76, e_76=e76, inc_76=i76, rho_76=rho76,
        r_ms=rms, v_ms=vms, alt_ms=altms, e_ms=ems, inc_ms=ims, rho_ms=rhoms,
        lon_ms=lon_ms, lat_ms=lat_ms, ratio=ratio,
        bstar=args.bstar, f107=f107_ep, f107a=f107a_ep, ap=ap7_ep[0], ic=x0,
    )
    print(f"  Saved: {out_data}")


if __name__ == "__main__":
    main()
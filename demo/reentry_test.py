"""
reentry_test.py
---------------

Date: 2026-04-13

Simulates the reentry of a launch vehicle component from 900 km altitude
down to splashdown (altitude = 0 km).

Compares USSA76 vs NRLMSIS-2.1 models.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.integrate import solve_ivp
from demo_lib.perturbations import (
    perturbations_ussa76, perturbations_msis,
    eci_to_geodetic, _MU, _RE
)
from demo_lib.space_weather import SpaceWeather, SpaceWeatherEngine
import os

# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def splashdown_event(t, state, *args):
    """Event function to stop integration at altitude = 0."""
    r = state[:3]
    alt = np.linalg.norm(r) - _RE
    return alt

splashdown_event.terminal = True
splashdown_event.direction = -1

# ---------------------------------------------------------------------------
# Equations of motion (simplified wrappers for reentry)
# ---------------------------------------------------------------------------

def eom_ussa76(t, state, bstar):
    r, v = state[:3], state[3:]
    ag = -_MU / np.linalg.norm(r)**3 * r
    ap, _ = perturbations_ussa76(r, v, bstar)
    return np.concatenate([v, ag + ap])

def eom_msis(t, state, bstar, epoch, sw_engine):
    r, v = state[:3], state[3:]
    ag = -_MU / np.linalg.norm(r)**3 * r
    f107, f107a, ap7 = sw_engine.query(t)
    current_date = epoch + timedelta(seconds=float(t))
    ap_acc, _ = perturbations_msis(r, v, bstar, date=current_date, f107=f107, f107a=f107a, ap7=ap7)
    return np.concatenate([v, ag + ap_acc])

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    epoch = datetime(2024, 5, 10, 12, 0, 0) # Use the SC25 storm date for interesting MSIS results
    bstar = 1.0e-3 # High drag for a reentry vehicle
    
    # Initial Condition: 900 km altitude, sub-orbital velocity
    alt_init = 900.0
    r_mag = _RE + alt_init
    r0 = np.array([r_mag, 0.0, 0.0])
    
    # Circular velocity is ~7.4 km/s. Let's use 6.5 km/s for a steeper reentry.
    v_mag = 6.5 
    v0 = np.array([0.0, v_mag, 0.0])
    x0 = np.concatenate([r0, v0])
    
    print(f"Starting Reentry Simulation from {alt_init} km...")
    print(f"Initial Velocity: {v_mag} km/s")
    
    # Load Space Weather
    sw = SpaceWeather(use_all=True)
    simulation_duration_days = 1.0
    sw_engine = SpaceWeatherEngine(sw, epoch, simulation_duration_days)
    
    # USSA76 Integration
    print("Running USSA76 reentry...")
    sol_76 = solve_ivp(eom_ussa76, (0.0, 10000.0), x0, args=(bstar,),
                       events=splashdown_event, rtol=1e-8, atol=1e-8)
    
    # MSIS Integration
    print("Running MSIS-2.1 reentry...")
    sol_ms = solve_ivp(eom_msis, (0.0, 10000.0), x0, args=(bstar, epoch, sw_engine),
                       events=splashdown_event, rtol=1e-8, atol=1e-8)
    
    # Results
    t76 = sol_76.t[-1]
    tms = sol_ms.t[-1]
    
    r76_f = sol_76.y[:3, -1]
    rms_f = sol_ms.y[:3, -1]
    
    lon76, lat76, alt76 = eci_to_geodetic(r76_f, epoch + timedelta(seconds=t76))
    lonms, latms, altms = eci_to_geodetic(rms_f, epoch + timedelta(seconds=tms))
    
    print("\n--- REENTRY RESULTS ---")
    print(f"USSA76 Splashdown Time: {t76:.2f} s ({t76/60:.2f} min)")
    print(f"USSA76 Location: Lat {lat76:.4f}, Lon {lon76:.4f}")
    
    print(f"MSIS-2.1 Splashdown Time: {tms:.2f} s ({tms/60:.2f} min)")
    print(f"MSIS-2.1 Location: Lat {latms:.4f}, Lon {lonms:.4f}")
    
    print(f"\nDifference in Time: {abs(t76 - tms):.2f} s")
    
    # --- Enhanced Plotting with Zoom ---
    fig, axes = plt.subplots(4, 1, figsize=(10, 18), sharex=False)
    
    # 1. Altitude Plot
    alts_76 = np.linalg.norm(sol_76.y[:3], axis=0) - _RE
    alts_ms = np.linalg.norm(sol_ms.y[:3], axis=0) - _RE
    
    axes[0].plot(sol_76.t, alts_76, label='USSA76', lw=2, color='#1565C0')
    axes[0].plot(sol_ms.t, alts_ms, label='MSIS-2.1', ls='--', lw=2, color='#E65100')
    axes[0].set_ylabel('Altitude [km]', fontweight='bold')
    axes[0].set_title(f'Reentry Profile: 900 km to Splashdown\nBStar={bstar:.1e}, Epoch={epoch.date()}', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Density Plot
    from demo_lib.perturbations import ussa76, msis_density
    rho_76_vals = [ussa76(a) for a in alts_76]
    rho_ms_vals = []
    for i in range(len(sol_ms.t)):
        t_curr = sol_ms.t[i]
        r_curr = sol_ms.y[:3, i]
        date_curr = epoch + timedelta(seconds=float(t_curr))
        lon, lat, alt = eci_to_geodetic(r_curr, date_curr)
        f107, f107a, ap7 = sw_engine.query(t_curr)
        rho_ms_vals.append(msis_density(alt, lon, lat, date_curr, f107, f107a, ap7))

    axes[1].semilogy(sol_76.t, rho_76_vals, label='USSA76', lw=1.5, color='#1565C0')
    axes[1].semilogy(sol_ms.t, rho_ms_vals, label='MSIS-2.1', ls='--', lw=1.5, color='#E65100')
    axes[1].set_ylabel(r'Density [kg/m$^3$]', fontweight='bold')
    axes[1].set_xlabel('Time [s]', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)

    # 3. Full Ground Track
    lons_76, lats_76 = [], []
    for i in range(len(sol_76.t)):
        lon, lat, _ = eci_to_geodetic(sol_76.y[:3, i], epoch + timedelta(seconds=float(sol_76.t[i])))
        lons_76.append(lon)
        lats_76.append(lat)
        
    lons_ms, lats_ms = [], []
    for i in range(len(sol_ms.t)):
        lon, lat, _ = eci_to_geodetic(sol_ms.y[:3, i], epoch + timedelta(seconds=float(sol_ms.t[i])))
        lons_ms.append(lon)
        lats_ms.append(lat)

    axes[2].plot(lons_76, lats_76, label='USSA76', lw=2, color='#1565C0')
    axes[2].plot(lons_ms, lats_ms, label='MSIS-2.1', ls='--', lw=2, color='#E65100')
    axes[2].scatter(lons_76[0], lats_76[0], color='green', marker='o', label='Start', zorder=5)
    axes[2].set_xlabel('Longitude [deg]', fontweight='bold')
    axes[2].set_ylabel('Latitude [deg]', fontweight='bold')
    axes[2].set_title('Full Ground Track', fontweight='bold')
    axes[2].set_xlim([-180, 180])
    axes[2].set_ylim([-90, 90])
    axes[2].grid(True, alpha=0.3)

    # 4. Zoomed Splashdown View
    # Calculate distance in meters (approx at equator: 1 deg ~ 111.32 km)
    d_lon_km = (lonms - lon76) * 111.32 * np.cos(np.radians(latms))
    d_lat_km = (latms - lat76) * 111.32
    dist_m = np.sqrt(d_lon_km**2 + d_lat_km**2) * 1000.0

    axes[3].plot(lons_76, lats_76, marker='o', markersize=4, label='USSA76', color='#1565C0')
    axes[3].plot(lons_ms, lats_ms, marker='s', markersize=4, ls='--', label='MSIS-2.1', color='#E65100')
    
    # Center on the landing points
    center_lon = (lon76 + lonms) / 2
    center_lat = (lat76 + latms) / 2
    # Zoom window: roughly 2km x 2km
    zoom_deg = 2.0 / 111.32 
    
    axes[3].set_xlim([center_lon - zoom_deg, center_lon + zoom_deg])
    axes[3].set_ylim([center_lat - zoom_deg, center_lat + zoom_deg])
    
    axes[3].set_xlabel('Longitude [deg]', fontweight='bold')
    axes[3].set_ylabel('Latitude [deg]', fontweight='bold')
    axes[3].set_title(f'Splashdown Precision Zoom\nSeparation: {dist_m:.1f} meters', fontweight='bold')
    axes[3].text(0.05, 0.95, f'Separation: {dist_m:.1f} m\nTime Diff: {abs(t76-tms):.1f} s', 
                 transform=axes[3].transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[3].legend()
    axes[3].grid(True, alpha=0.5)

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/reentry_test_zoom.png")
    print(f"\nSaved zoomed plot to output/reentry_test_zoom.png")
    print(f"Calculated Separation: {dist_m:.1f} meters")

if __name__ == "__main__":
    main()

"""
Ground-track targeting with J2: Skopinski & Johnson (NASA TN D-233) + numerical propagation.

Finds the burnout azimuth / orbital elements so that a satellite launched from a known burnout
point passes over Austin, TX after n revolutions, then verifies the result by integrating
two-body + J2 dynamics.

Usage:
    python orbit_project.py            # Austin case + plots
    python orbit_project.py --case-a   # validate solver against the NASA TN D-233 Case A values
"""
import sys
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
except ImportError:
    HAVE_CARTOPY = False

# ----------------------------------------------------------------------------
# Constants (project statement values)
# ----------------------------------------------------------------------------
MU = 398600.4415                 # km^3/s^2
RE = 6378.1363                   # km
J2 = 0.0010826267
OMEGA_E = 2 * np.pi / 86164.0    # rad/s
D2R, R2D = np.pi / 180.0, 180.0 / np.pi


# ----------------------------------------------------------------------------
# Kepler / J2 helpers
# ----------------------------------------------------------------------------
def kepler_time(theta, a, e):
    """Time since perigee for true anomaly theta (rad); valid for any theta >= 0 (multi-revolution)."""
    n = np.sqrt(MU / a**3)
    T = 2 * np.pi / n
    k = np.floor(theta / (2 * np.pi))
    th = theta - 2 * np.pi * k
    E = 2 * np.arctan2(np.sqrt((1 - e) / (1 + e)) * np.sin(th / 2), np.cos(th / 2))
    return (E - e * np.sin(E)) / n + k * T


def j2_rates(a, e, inc):
    """Secular J2 rates of RAAN and argument of perigee (rad/s)."""
    p = a * (1 - e**2)
    n = np.sqrt(MU / a**3)
    f = J2 * n * (RE / p) ** 2
    return -1.5 * f * np.cos(inc), 0.75 * f * (5 * np.cos(inc) ** 2 - 1)


def j2_target_corrections(a, e, inc, argp, theta2e, phi2, dt):
    """Appendix A oblateness corrections to target latitude/longitude over time dt."""
    Odot, wdot = j2_rates(a, e, inc)
    dO, dw = Odot * dt, wdot * dt
    arg = argp + theta2e
    dphi = np.sin(inc) * np.cos(arg) / np.cos(phi2) * dw
    dlam = np.cos(inc) / np.cos(arg) ** 2 / (1 + np.cos(inc) ** 2 * np.tan(arg) ** 2) * dw + dO
    return dphi, dlam, dw, dO


# ----------------------------------------------------------------------------
# Skopinski & Johnson solver
# ----------------------------------------------------------------------------
def skopinski(a, e, theta1, phi1, phi2, lon1, lon2, n_rev, n_iter=4, verbose=False):
    """
    Returns dict with burnout azimuth A1, inclination, RAAN, argument of perigee,
    equivalent true anomaly at the pass, and predicted time of the pass (s after burnout).
    Angles in radians; longitudes east-positive in [0, 2*pi).
    """
    T = 2 * np.pi * np.sqrt(a**3 / MU)
    t1 = kepler_time(theta1, a, e)
    lam2e = np.mod(lon2 + n_rev * OMEGA_E * T, 2 * np.pi)          # equivalent longitude (Eq. 18)
    dt21 = np.mod(lam2e - lon1, 2 * np.pi) * T / (2 * np.pi)       # initial time guess (Eq. 21)
    phi_eff, lam_eff = phi2, lam2e

    for k in range(n_iter):
        # Eastward longitude difference, accounting for Earth rotation during the final partial orbit
        dl = np.mod(lam_eff - lon1 + OMEGA_E * dt21, 2 * np.pi)
        c = np.clip(np.sin(phi_eff) * np.sin(phi1)
                    + np.cos(phi_eff) * np.cos(phi1) * np.cos(dl), -1, 1)
        dth = np.arccos(c)                                          # short-arc central angle
        psi = np.arctan2(np.sin(dl) * np.cos(phi_eff),
                         np.cos(phi1) * np.sin(phi_eff) - np.sin(phi1) * np.cos(phi_eff) * np.cos(dl))
        if psi < 0:
            # The short arc heads west. A prograde orbit must take the long arc the other way round.
            psi += np.pi
            dth = 2 * np.pi - dth
        theta2e = theta1 + dth

        dt_prev = dt21
        dt21 = kepler_time(theta2e, a, e) - t1                      # updated from theta2e (not Eq. 21)

        inc = np.arccos(np.clip(np.cos(phi1) * np.sin(psi), -1, 1))
        u1 = np.arcsin(np.clip(np.sin(phi1) / np.sin(inc), -1, 1))   # argument of latitude at burnout
        if np.cos(psi) < 0:
            u1 = np.pi - u1                                          # southbound at burnout
        argp = u1 - theta1

        if verbose:
            print(f"  iter {k+1}: dlam={dl*R2D:9.4f}  theta2e={theta2e*R2D:9.4f}  "
                  f"t(theta2e)={kepler_time(theta2e, a, e):10.4f}  psi={psi*R2D:8.4f}  i={inc*R2D:8.4f}")

        if k == 0:
            # Oblateness corrections: computed once after iteration 1, applied from iteration 2 on
            dphi, dlam, dw, dO = j2_target_corrections(a, e, inc, argp, theta2e, phi2, n_rev * T + dt_prev)
            phi_eff, lam_eff = phi2 - dphi, lam2e - dlam
            if verbose:
                print(f"     J2: d_omega={dw*R2D:.5f}  d_Omega={dO*R2D:.5f}  d_phi2={dphi*R2D:.5f}  "
                      f"d_lam2={dlam*R2D:.5f}  corrected phi2={phi_eff*R2D:.4f}")

    raan = np.mod(lon1 - np.arctan2(np.cos(inc) * np.sin(u1), np.cos(u1)), 2 * np.pi)
    return dict(A1=psi, inc=inc, RAAN=raan, argp=argp, theta2e=theta2e, T=T,
                t_pass=n_rev * T + dt21)


# ----------------------------------------------------------------------------
# Elements -> state, dynamics, frames
# ----------------------------------------------------------------------------
def oe2cart(a, e, inc, raan, argp, nu):
    """Classical elements -> ECI position/velocity (perifocal -> ECI, standard 3-1-3 rotation)."""
    p = a * (1 - e**2)
    r_pf = np.array([p * np.cos(nu) / (1 + e * np.cos(nu)),
                     p * np.sin(nu) / (1 + e * np.cos(nu)), 0.0])
    v_pf = np.array([-np.sqrt(MU / p) * np.sin(nu),
                     np.sqrt(MU / p) * (e + np.cos(nu)), 0.0])
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(argp), np.sin(argp)
    R3_O = np.array([[cO, -sO, 0], [sO, cO, 0], [0, 0, 1]])
    R1_i = np.array([[1, 0, 0], [0, ci, -si], [0, si, ci]])
    R3_w = np.array([[cw, -sw, 0], [sw, cw, 0], [0, 0, 1]])
    Q = R3_O @ R1_i @ R3_w
    return Q @ r_pf, Q @ v_pf


def two_body_j2(t, y):
    r, v = y[:3], y[3:]
    x, yy, z = r
    rn = np.linalg.norm(r)
    r2 = rn**2
    a_c = -MU * r / rn**3
    f = 1.5 * J2 * MU * RE**2 / rn**5
    a_j2 = np.array([f * x * (5 * z**2 / r2 - 1),
                     f * yy * (5 * z**2 / r2 - 1),
                     f * z * (5 * z**2 / r2 - 3)])
    return np.hstack((v, a_c + a_j2))


def eci_to_ecef(r_eci, t, theta0=0.0):
    """GMST(t) = theta0 + omega_E * t, with theta0 = 0 at burnout per the project statement."""
    th = theta0 + OMEGA_E * t
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) @ r_eci


def geodetic_to_ecef(lat, lon, h=0.0):
    """Spherical-Earth position (km)."""
    return (RE + h) * np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def ecef_to_enu(r_ecef, r_obs, lat, lon):
    sl, cl, sn, cn = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    R = np.array([[-sn, cn, 0], [-cn * sl, -sn * sl, cl], [cn * cl, sn * cl, sl]])
    return R @ (r_ecef - r_obs)


# ----------------------------------------------------------------------------
# Validation against NASA TN D-233 Case A (values from the course notes)
# ----------------------------------------------------------------------------
def validate_case_a():
    P = 5495.211501135044
    E1, t1 = 0.40952370835562396, 350.5348557563641
    a = 6730.525196784
    e = (E1 - 2 * np.pi / P * t1) / np.sin(E1)
    theta1 = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E1 / 2))
    print("Case A (NASA TN D-233, eastward launch) -- solver output vs. reference")
    s = skopinski(a, e, theta1, 28.5 * D2R, 34.0 * D2R, 279.45 * D2R, 241.0 * D2R, 3, n_iter=3, verbose=True)
    print(f"\n  RAAN = {s['RAAN']*R2D:.4f} deg   (reference 225.9553)")
    print(f"  omega = {s['argp']*R2D:.4f} deg   (reference 34.5103)")
    assert abs(s['RAAN'] * R2D - 225.9553) < 1e-3 and abs(s['argp'] * R2D - 34.5103) < 1e-3
    print("  PASS")


# ----------------------------------------------------------------------------
# Main: Austin case
# ----------------------------------------------------------------------------
def main(save_dir=None):
    a, e = 6500.0, 0.001
    theta1 = 20.0 * D2R
    phi1, lon1 = 30.3 * D2R, (360.0 - 120.6) * D2R       # burnout: 30.3 N, 120.6 W
    phi2, lon2 = 30.3 * D2R, (360.0 - 97.7) * D2R        # Austin
    n_rev = 10

    print("Skopinski & Johnson solution (Austin target):")
    s = skopinski(a, e, theta1, phi1, phi2, lon1, lon2, n_rev, n_iter=4, verbose=True)
    print(f"\n  Burnout azimuth A1 = {s['A1']*R2D:.4f} deg")
    print(f"  Inclination i      = {s['inc']*R2D:.4f} deg")
    print(f"  RAAN               = {s['RAAN']*R2D:.4f} deg")
    print(f"  Arg. of perigee    = {s['argp']*R2D:.4f} deg")
    print(f"  Predicted pass     = {s['t_pass']/3600:.4f} hr after burnout "
          f"(10 periods = {n_rev*s['T']/3600:.4f} hr)\n")

    r0, v0 = oe2cart(a, e, s['inc'], s['RAAN'], s['argp'], theta1)
    lat0 = np.arcsin(r0[2] / np.linalg.norm(r0)) * R2D
    lon0 = np.arctan2(r0[1], r0[0]) * R2D
    print(f"  Burnout check from state: lat={lat0:.3f}, lon={lon0:.3f} (target 30.300, -120.600)")

    # Propagate past the predicted pass (NOT just 10 periods: the pass is the partial orbit after them)
    tf = s['t_pass'] + 1800.0
    t_eval = np.arange(0.0, tf, 10.0)
    sol = solve_ivp(two_body_j2, (0, tf), np.hstack((r0, v0)), t_eval=t_eval, rtol=1e-11, atol=1e-11)
    t, r_eci = sol.t, sol.y[:3].T

    lat_o, lon_o = 30.3 * D2R, -97.7 * D2R
    r_obs = geodetic_to_ecef(lat_o, lon_o)
    lats, lons, elev, rng = [], [], [], []
    for ti, ri in zip(t, r_eci):
        r_ecef = eci_to_ecef(ri, ti)
        lats.append(np.arctan2(r_ecef[2], np.hypot(r_ecef[0], r_ecef[1])) * R2D)
        lons.append((np.arctan2(r_ecef[1], r_ecef[0]) * R2D + 180) % 360 - 180)
        enu = ecef_to_enu(r_ecef, r_obs, lat_o, lon_o)
        rng.append(np.linalg.norm(enu))
        elev.append(np.arcsin(enu[2] / rng[-1]) * R2D)
    elev, rng = np.array(elev), np.array(rng)
    k = int(np.argmax(elev))
    print(f"\nVerification (two-body + J2 propagation):")
    print(f"  Max elevation over Austin : {elev[k]:.2f} deg at t = {t[k]/3600:.4f} hr")
    print(f"  Min range to Austin       : {rng.min():.1f} km")
    print(f"  Predicted vs. actual pass : {s['t_pass']/3600:.4f} hr vs {t[k]/3600:.4f} hr "
          f"({(t[k]-s['t_pass']):+.0f} s)")

    # ------------------------------ plots ------------------------------
    fig = plt.figure(figsize=(11, 5))
    if HAVE_CARTOPY:
        ax = plt.axes(projection=ccrs.PlateCarree())
        try:
            ax.add_feature(cfeature.COASTLINE, linewidth=1)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        except Exception:
            print("  (could not load map features; plotting without coastlines)")
        ax.gridlines(draw_labels=True)
        kw = dict(transform=ccrs.PlateCarree())
    else:
        ax = plt.gca(); ax.grid(True); ax.set_xlim(-180, 180); ax.set_ylim(-60, 60); kw = {}
    # Integration step is 10 s; plot every 60 s so the individual dots stay visible
    ax.plot(lons[::6], lats[::6], '.', markersize=3, label="Ground track", **kw)
    ax.plot(-97.7, 30.3, 'r*', markersize=12, label="Austin, TX", **kw)
    ax.plot(lon0, lat0, 'g^', markersize=9, label="Burnout", **kw)
    plt.title("Ground Track (Two-Body + J2), burnout to just after the predicted pass")
    plt.legend()
    if save_dir: plt.savefig(f"{save_dir}/ground_track.png", dpi=200, bbox_inches="tight")

    plt.figure(figsize=(9, 4))
    plt.plot(t / 3600, elev); plt.axhline(0, color='r', ls='--')
    plt.xlabel("Time since burnout (hours)"); plt.ylabel("Elevation (deg)")
    plt.title("Elevation Angle as Seen From Austin, TX"); plt.grid(True)
    if save_dir: plt.savefig(f"{save_dir}/elevation_full.png", dpi=200, bbox_inches="tight")

    win = (t > t[k] - 900) & (t < t[k] + 900)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].plot((t[win] - t[k]) / 60, elev[win]); axs[0].axhline(0, color='r', ls='--')
    axs[0].set_xlabel("Minutes from closest approach"); axs[0].set_ylabel("Elevation (deg)")
    axs[0].set_title("Elevation near the pass"); axs[0].grid(True)
    axs[1].plot((t[win] - t[k]) / 60, rng[win])
    axs[1].set_xlabel("Minutes from closest approach"); axs[1].set_ylabel("Range (km)")
    axs[1].set_title("Range near the pass"); axs[1].grid(True)
    plt.tight_layout()
    if save_dir: plt.savefig(f"{save_dir}/pass_closeup.png", dpi=200, bbox_inches="tight")

    if not save_dir:
        plt.show()


if __name__ == "__main__":
    if "--case-a" in sys.argv:
        validate_case_a()
    else:
        main()

"""
Regression tests for orbit_project.py.

Case A is the eastward-launch example in NASA TN D-233 (Skopinski & Johnson, 1960).
The reference values below come from a Python implementation of that algorithm using the
project's constants (mu, J2, R_earth, omega_earth) and three iterations.

NOTE: the Case A *inputs* (burnout lat/lon, target lat/lon, number of orbits) are reconstructed
from the reference outputs, not copied from the technical note: phi1 = 28.5 N, lambda1 = 279.45 E,
phi2 = 34.0 N, lambda2 = 241.0 E, n = 3 orbits, with e and theta1 recovered from the reference
eccentric anomaly and time since perigee. If every reference value below is reproduced, those
inputs are consistent with the published case.

Run from the repository root:  pytest -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import orbit_project as op  # noqa: E402

D2R, R2D = op.D2R, op.R2D

# ----------------------------------------------------------------------------------------
# Reference values (degrees / seconds), NASA TN D-233 Case A
# ----------------------------------------------------------------------------------------
REF_ITER = [
    dict(dlam=32.3688834639212, theta2e=52.0613502641232, t_theta2e=764.765562378691,
         psi=70.4839727473241, inc=34.072698947011),
    dict(dlam=31.1274822452164, theta2e=51.0290050490009, t_theta2e=749.439240858738,
         psi=70.5556118911761, inc=34.0352057122414),
    dict(dlam=31.063447662941, theta2e=50.9759409918365, t_theta2e=748.651697397749,
         psi=70.5494476695351, inc=34.0384280486144),
]
REF_J2 = dict(d_omega=1.96995545413727, d_Omega=-1.34267017302985,
              d_phi2=0.08152276927977, d_lam2=1.03148989639591,
              phi2_corrected=33.9184772307202, lon2_corrected=239.968510103604)
REF_RAAN, REF_OMEGA = 225.955331131408, 34.5103270086054


@pytest.fixture(scope="module")
def case_a():
    a = 6730.525196784
    period, E1, t1 = 5495.211501135044, 0.40952370835562396, 350.5348557563641
    e = (E1 - 2 * np.pi / period * t1) / np.sin(E1)
    theta1 = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E1 / 2))
    return op.skopinski(a, e, theta1, 28.5 * D2R, 34.0 * D2R, 279.45 * D2R, 241.0 * D2R,
                        3, n_iter=3)


@pytest.mark.parametrize("k", [0, 1, 2])
def test_case_a_iterations(case_a, k):
    got, ref = case_a["history"][k], REF_ITER[k]
    assert got["dlam"] * R2D == pytest.approx(ref["dlam"], abs=2e-3)
    assert got["theta2e"] * R2D == pytest.approx(ref["theta2e"], abs=2e-3)
    assert got["t_theta2e"] == pytest.approx(ref["t_theta2e"], abs=0.05)
    assert got["psi"] * R2D == pytest.approx(ref["psi"], abs=2e-3)
    assert got["inc"] * R2D == pytest.approx(ref["inc"], abs=2e-3)


def test_case_a_j2_corrections(case_a):
    j2 = case_a["j2"]
    for key in ("d_omega", "d_Omega", "d_phi2", "d_lam2", "phi2_corrected", "lon2_corrected"):
        assert j2[key] * R2D == pytest.approx(REF_J2[key], abs=1e-3), key


def test_case_a_final_elements(case_a):
    assert case_a["RAAN"] * R2D == pytest.approx(REF_RAAN, abs=1e-3)
    assert case_a["argp"] * R2D == pytest.approx(REF_OMEGA, abs=1e-3)


# ----------------------------------------------------------------------------------------
# Regression tests for bugs found while debugging
# ----------------------------------------------------------------------------------------
def test_oe2cart_roundtrip():
    """Elements -> state -> elements must recover i, RAAN, omega (catches transposed rotations)."""
    a, e = 6500.0, 0.001
    inc, raan, argp, nu = 50 * D2R, 100 * D2R, 30 * D2R, 20 * D2R
    r, v = op.oe2cart(a, e, inc, raan, argp, nu)
    h = np.cross(r, v)
    node = np.cross([0, 0, 1], h)
    ecc = ((v @ v - op.MU / np.linalg.norm(r)) * r - (r @ v) * v) / op.MU
    assert np.arccos(h[2] / np.linalg.norm(h)) * R2D == pytest.approx(50, abs=1e-6)
    assert np.arctan2(node[1], node[0]) % (2 * np.pi) * R2D == pytest.approx(100, abs=1e-6)
    assert np.arccos(node @ ecc / (np.linalg.norm(node) * np.linalg.norm(ecc))) * R2D \
        == pytest.approx(30, abs=1e-6)


@pytest.fixture(scope="module")
def austin():
    a, e, theta1 = 6500.0, 0.001, 20 * D2R
    sol = op.skopinski(a, e, theta1, 30.3 * D2R, 30.3 * D2R, (360 - 120.6) * D2R,
                       (360 - 97.7) * D2R, 10, n_iter=4)
    return a, e, theta1, sol


def test_austin_solution_is_prograde_long_arc(austin):
    _, _, theta1, s = austin
    assert s["inc"] < np.pi / 2                       # prograde
    assert s["A1"] * R2D == pytest.approx(120.14, abs=0.05)
    assert s["inc"] * R2D == pytest.approx(41.70, abs=0.05)
    assert (s["theta2e"] - theta1) > np.pi            # the pass is on the long arc
    assert s["t_pass"] > 10 * s["T"]                  # pass occurs after the 10 full periods


def test_austin_burnout_state_and_pass(austin):
    """The element set must put the satellite at the burnout point and, propagated with J2,
    pass almost directly over Austin near the predicted time."""
    a, e, theta1, s = austin
    r0, v0 = op.oe2cart(a, e, s["inc"], s["RAAN"], s["argp"], theta1)
    assert np.arcsin(r0[2] / np.linalg.norm(r0)) * R2D == pytest.approx(30.3, abs=1e-3)
    assert np.arctan2(r0[1], r0[0]) * R2D == pytest.approx(-120.6, abs=1e-3)

    tf = s["t_pass"] + 600.0
    t_eval = np.arange(s["t_pass"] - 600.0, tf, 2.0)
    sol = solve_ivp(op.two_body_j2, (0, tf), np.hstack((r0, v0)), t_eval=t_eval,
                    rtol=1e-11, atol=1e-11)
    lat_o, lon_o = 30.3 * D2R, -97.7 * D2R
    r_obs = op.geodetic_to_ecef(lat_o, lon_o)
    elev, rng = [], []
    for t, r in zip(sol.t, sol.y[:3].T):
        enu = op.ecef_to_enu(op.eci_to_ecef(r, t), r_obs, lat_o, lon_o)
        rng.append(np.linalg.norm(enu))
        elev.append(np.arcsin(enu[2] / rng[-1]) * R2D)
    elev, rng = np.array(elev), np.array(rng)
    assert elev.max() > 75.0                          # near-overhead pass
    assert rng.min() < 150.0                          # km (orbit altitude is ~120 km)
    assert abs(sol.t[elev.argmax()] - s["t_pass"]) < 120.0   # within 2 min of the prediction

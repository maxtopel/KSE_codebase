"""Lorenz-63: 20 stationary observables across a wide KSE range."""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from common import run_pipeline

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
SNRS = (np.inf, 30, 20, 15, 10)


def lorenz(t, s):
    x, y, z = s
    return [SIGMA * (y - x), x * (RHO - z) - y, x * y - BETA * z]


def smooth(x, w):
    return np.convolve(x, np.ones(w) / w, mode="same")


def main():
    dt = 0.01
    sol = solve_ivp(lorenz, (0.0, 200.0), [1.0, 1.0, 1.0],
                    t_eval=np.arange(0.0, 200.0, dt),
                    rtol=1e-10, atol=1e-12, method="DOP853")
    x, y, z = (a[5000:] for a in sol.y)
    truth = np.stack([x, y, z], axis=1)
    obs = {
        "x": x, "y": y, "z": z,
        **{f"x_smooth_{w}": smooth(x, w) for w in (5, 25, 100)},
        **{f"z_smooth_{w}": smooth(z, w) for w in (5, 25, 100)},
        "dxdt": np.gradient(x, dt), "dzdt": np.gradient(z, dt),
        "d2xdt2": np.gradient(np.gradient(x, dt), dt),
        "x_squared": x ** 2, "y_squared": y ** 2, "z_squared": z ** 2,
        "log1p_z": np.log1p(np.maximum(z, 0)),
        "x_plus_y": x + y, "radial": np.sqrt(x ** 2 + y ** 2 + z ** 2),
        "x_times_y": x * y, "x_times_z": x * z,
    }
    run_pipeline("lorenz63", truth, obs, SNRS, dim=7, int_dim=3)


if __name__ == "__main__":
    main()

"""Hastings-Powell three-species food chain: 20 stationary observables."""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from common import run_pipeline

A1, A2, B1, B2, D1, D2 = 5.0, 0.1, 3.0, 2.0, 0.4, 0.01
SNRS = (np.inf, 30, 20, 15, 10)


def hp(t, s):
    x, y, z = s
    return [x * (1 - x) - A1 * x * y / (1 + B1 * x),
            A1 * x * y / (1 + B1 * x) - A2 * y * z / (1 + B2 * y) - D1 * y,
            A2 * y * z / (1 + B2 * y) - D2 * z]


def smooth(x, w):
    return np.convolve(x, np.ones(w) / w, mode="same")


def safe_log(x):
    return np.log(np.clip(x, 1e-9, None))


def main():
    dt = 1.0
    sol = solve_ivp(hp, (0.0, 20000.0), [0.7, 0.2, 8.0],
                    t_eval=np.arange(0.0, 20000.0, dt),
                    rtol=1e-10, atol=1e-12, method="DOP853")
    x, y, z = (a[5000:] for a in sol.y)
    truth = np.stack([x, y, z], axis=1)
    biomass = x + y + z
    obs = {
        "resource": x, "consumer": y, "predator": z,
        "log_resource": safe_log(x), "log_consumer": safe_log(y), "log_predator": safe_log(z),
        **{f"resource_smooth_{w}": smooth(x, w) for w in (5, 25, 100)},
        "consumer_smooth_25": smooth(y, 25), "predator_smooth_25": smooth(z, 25),
        "dresource_dt": np.gradient(x, dt), "dconsumer_dt": np.gradient(y, dt),
        "dpredator_dt": np.gradient(z, dt),
        "total_biomass": biomass, "log_total_biomass": safe_log(biomass),
        "resource_squared": x ** 2,
        "predator_minus_resource": z - x,
        "resource_times_consumer": x * y,
        "predator_over_consumer": z / np.clip(y, 1e-9, None),
    }
    run_pipeline("hastings_powell", truth, obs, SNRS, dim=7, int_dim=3)


if __name__ == "__main__":
    main()

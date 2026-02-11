"""Geometric Brownian motion path simulation utilities."""

import numpy as np


def simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, rng=None, shocks=None):
    if S0 <= 0 or sigma <= 0 or T <= 0:
        raise ValueError("S0, sigma, and T must be positive")
    if steps <= 0 or n_paths <= 0:
        raise ValueError("steps and n_paths must be positive integers")

    if shocks is not None:
        shocks = np.asarray(shocks, dtype=float)  # allow external draws
        expected_shape = (n_paths, steps)
        if shocks.shape != expected_shape:
            raise ValueError(f"shocks must have shape {expected_shape}")
    else:
        if rng is None:
            raise ValueError("rng is required when shocks are not provided")
        shocks = rng.normal(size=(n_paths, steps))  # default iid normals

    dt = T / steps
    drift = (r - 0.5 * sigma * sigma) * dt
    diffusion = sigma * np.sqrt(dt)

    paths = np.empty((n_paths, steps + 1), dtype=float)
    paths[:, 0] = S0  # initial level at t=0 for every path

    log_returns = drift + diffusion * shocks  # Euler log-return increments
    log_levels = np.cumsum(log_returns, axis=1)  # cumulative log path
    paths[:, 1:] = S0 * np.exp(log_levels)  # convert back to price space

    return paths

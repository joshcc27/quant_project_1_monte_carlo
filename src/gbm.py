"""Geometric Brownian motion path simulation utilities.

The module exposes a vectorised simulator for risk-neutral GBM paths:
``dS_t / S_t = r dt + sigma dW_t``, implemented in log-space on a 
discrete grid for numerical stability and speed.
"""

import numpy as np
from .validation import validate_positive, validate_positive_int


def simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, rng=None, shocks=None):
    """Simulate GBM price paths on an equally spaced monitoring grid.

    Parameters
    ----------
    S0 : float
        Initial spot price at time ``t=0``.
    r : float
        Continuously compounded risk-free drift used under pricing measure.
    sigma : float
        Volatility (annualized, decimal).
    T : float
        Time horizon in years.
    steps : int
        Number of time increments between ``0`` and ``T``.
    n_paths : int
        Number of Monte Carlo paths to generate.
    rng : object, optional
        Random number source exposing ``normal(size=...)``.
        Required when ``shocks`` is not provided.
    shocks : array-like, optional
        Pre-generated standard-normal shocks with shape ``(n_paths, steps)``.
        When passed, ``rng`` is ignored.

    Returns
    -------
    numpy.ndarray
        Simulated price matrix with shape ``(n_paths, steps + 1)``.
        Column ``0`` is the initial value ``S0`` and columns ``1:`` are
        simulated levels at monitoring times.

    Raises
    ------
    ValueError
        If required inputs are not strictly positive, if neither ``rng`` nor
        valid ``shocks`` are supplied, or if ``shocks`` has wrong shape.
    """
    # Basic parameter guards 
    # Under GBM, S0 > 0, sigma > 0, and T > 0 are required
    validate_positive(S0, "S0")
    validate_positive(sigma, "sigma")
    validate_positive(T, "T")
    # Simulation dimensions must be strictly positive so arrays are well-defined
    validate_positive_int(steps, "steps")
    validate_positive_int(n_paths, "n_paths")

    # The simulator supports two modes:
    # 1) Caller-provided shocks: deterministic/reproducible experiments
    # 2) RNG-generated shocks: standard Monte Carlo
    if shocks is not None:
        shocks = np.asarray(shocks, dtype=float)
        expected_shape = (n_paths, steps)
        if shocks.shape != expected_shape:
            # Shape convention:
            # rows    -> independent paths
            # columns -> one normal shock per time increment
            raise ValueError(f"shocks must have shape {expected_shape}")
    else:
        if rng is None:
            # If shocks are not injected, we must have an RNG object exposing .normal(...)
            raise ValueError("rng is required when shocks are not provided")
        # iid N(0,1) innovations for each path/time step
        shocks = rng.normal(size=(n_paths, steps))

    dt = T / steps
    # Discrete log-return drift:
    # d log S = (r - 0.5*sigma^2) dt + sigma sqrt(dt) Z
    drift = (r - 0.5 * sigma * sigma) * dt
    # Per-step volatility scale applied to standard normal shocks
    diffusion = sigma * np.sqrt(dt)

    # Output array stores full paths including initial value at t=0
    paths = np.empty((n_paths, steps + 1), dtype=float)
    paths[:, 0] = S0

    # Vectorised path construction:
    # 1) Build all per-step log returns for all paths in one operation.
    log_returns = drift + diffusion * shocks
    # 2) Cumulative sum across time gives log(S_t / S0) for each time index.
    log_levels = np.cumsum(log_returns, axis=1)
    # 3) Exponentiate and scale by S0 to recover price levels in standard space.
    paths[:, 1:] = S0 * np.exp(log_levels)

    return paths

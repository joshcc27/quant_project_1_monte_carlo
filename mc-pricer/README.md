# Monte Carlo Pricer

## Overview
Monte Carlo-based pricing and Greek estimation toolkit for European and Asian options under geometric Brownian motion. Includes analytic Black-Scholes benchmarks, variance-reduction techniques, and extensive pytest coverage.

## Repository structure
- src/
  - bs_analytics.py: Black-Scholes pricing and analytic Greeks
  - gbm.py: GBM path simulation (accepts custom shock matrices)
  - mc.py: Monte Carlo pricing helpers returning price, stderr, CI, runtime
  - payoffs.py: European and Asian arithmetic payoff utilities
  - variance_reduction.py: Antithetic path simulators + control variate helper
  - greeks.py: Monte Carlo European delta/vega (pathwise + finite differences)
  - asian_geometric.py: Closed-form geometric Asian pricing
  - rng.py: NumPy default_rng wrapper for reproducible draws
- tests/: Pytest suite covering analytics, convergence, Greeks, and variance reduction
- requirements.txt: Python dependencies (numpy, scipy, pytest, matplotlib)

## Requirements
- Python 3.10+
- Install dependencies: pip install -r requirements.txt

## Running tests
`bash
cd mc-pricer
python -m pytest
`

## Usage sketch
`python
from rng import RNG
from gbm import simulate_gbm_paths
from mc import mc_price_european

S0, K, r, T, sigma = 100, 105, 0.02, 1.0, 0.2
rng = RNG(seed=42)
paths = simulate_gbm_paths(S0, r, sigma, T, steps=128, n_paths=50_000, rng=rng)
ST = paths[:, -1]
result = mc_price_european(ST, K, r, T, 'call')
print(result)
`

## Variance reduction
- **Antithetic variates**: mc_price_european_antithetic / mc_price_asian_arithmetic_antithetic
- **Control variate**: control_variate(X, Y, EY) for custom pairings (e.g., geometric Asian CV for arithmetic Asian payoffs)

## Greeks
mc_european_greeks returns pathwise + finite-difference delta/vega alongside analytic Black-Scholes values, using common random numbers to reduce noise.


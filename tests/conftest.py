"""Shared pytest configuration and fixtures for the test suite.

This file:
- ensures project imports are resolvable during test execution,
- provides reusable market/simulation fixtures,
- provides a small RNG factory fixture for deterministic test seeding.
"""

from pathlib import Path
import sys
import pytest
from src.rng import RNG

ROOT = Path(__file__).resolve().parents[1]
# Add project root so imports like ``from src...`` work regardless of where
# pytest is launched from.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def european_market():
    """Standard European-option parameter set reused across tests."""
    return {"S0": 100.0, "K": 100.0, "r": 0.02, "T": 1.0, "sigma": 0.25}


@pytest.fixture
def asian_market():
    """Standard Asian-option parameter set reused across tests."""
    return {"S0": 100.0, "K": 95.0, "r": 0.015, "T": 1.0, "sigma": 0.25}


@pytest.fixture
def sim_medium():
    """Medium simulation budget balancing stability and runtime."""
    return {"steps": 64, "n_paths": 200_000}


@pytest.fixture
def make_rng():
    """Factory fixture returning deterministic RNG instances by seed."""
    def _make(seed):
        return RNG(seed=seed)
    return _make


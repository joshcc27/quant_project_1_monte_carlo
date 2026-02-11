"""Wrapper around NumPy's Generator for reproduciblity."""
import numpy as np


class RNG:
    def __init__(self, seed=None):
        self.seed(seed)

    def seed(self, seed=None):
        """Reset the underlying generator with a new seed."""
        self._generator = np.random.default_rng(seed)

    def normal(self, size, mean=0.0, std=1.0):
        """Draw normal variates with broadcasting-compatible size."""
        return self._generator.normal(loc=mean, scale=std, size=size)

    def uniform(self, size=None):
        return self._generator.random(size=size)

"""Random number generator wrapper used across the project.

This module provides a small interface over ``numpy.random.Generator`` so
simulation code can depend on a consistent RNG contract and deterministic
seeding behaviour.
"""
import numpy as np


class RNG:
    """Lightweight wrapper around ``numpy.random.default_rng``.

    Parameters
    ----------
    seed : int | None, optional
        Seed for reproducible random streams. If ``None``, NumPy entropy is
        used to initialise the generator.
    """

    def __init__(self, seed=None):
        self.seed(seed)

    def seed(self, seed=None):
        """Reset the underlying generator with a new seed.

        Parameters
        ----------
        seed : int | None, optional
            Seed value passed to ``numpy.random.default_rng``.
        """
        self._generator = np.random.default_rng(seed)

    def normal(self, size, mean=0.0, std=1.0):
        """Draw normal random variates.

        Parameters
        ----------
        size : int | tuple[int, ...]
            Output shape of the draw.
        mean : float, optional
            Mean of the normal distribution (default ``0.0``).
        std : float, optional
            Standard deviation of the normal distribution (default ``1.0``).

        Returns
        -------
        numpy.ndarray
            Array of normal random variates with requested shape.
        """
        return self._generator.normal(loc=mean, scale=std, size=size)

    def uniform(self, size=None):
        """Draw uniform random variates on ``[0, 1)``.

        Parameters
        ----------
        size : int | tuple[int, ...] | None, optional
            Output shape. If ``None``, returns a scalar float.

        Returns
        -------
        float | numpy.ndarray
            Uniform draw(s) in the half-open interval ``[0, 1)``.
        """
        return self._generator.random(size=size)
